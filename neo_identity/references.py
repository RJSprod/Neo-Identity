"""The references of one generation: the pixels the host captured, and what the
recipe makes of them.

The host already collects reference images for Krea 2. ImageStitch encodes its
gallery through ``encode_first_stage`` and the engine keeps each image's pixels
in ``ref_latents``; an img2img input (or Model Chain's Stage 1 image) lands in
``ini_latent`` the same way. Native conditioning reads them as
``[ini_latent, *ref_latents]``: the img2img input first, then the gallery in its
order. That order is kept here, which is what lets a scene-and-person edit be
"the scene in the gallery first, the person second", or "the Stage 1 scene, then
the supplemental person" in Model Chain, without a second set of controls.

A ``ReferenceSet`` holds those pixels for the job and derives two things from
them on demand, each cached: the grounding images for Qwen3-VL (one size per
grounding setting) and the VAE latents fitted to a target size (one set per
target size and geometry). The VAE runs once per reference per target size,
before sampling when the conditioning tells us the size, and never inside the
step loop unless the sampled size was never announced.
"""

from __future__ import annotations

import logging

import torch

from . import geometry
from . import settings as S

logger = logging.getLogger("neo_identity")


def to_bchw(image: torch.Tensor) -> torch.Tensor:
    """A captured image, ``(1, H, W, C)`` in [0, 1], as ``(1, 3, H, W)``.

    The engine stores its references channels-last, as the VAE takes them. A
    channels-first tensor is accepted too, for callers that already converted.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"a reference image is (1, H, W, C), got {tuple(image.shape)}")
    if image.shape[-1] in (1, 3, 4):
        image = image.movedim(-1, 1)
    elif image.shape[1] not in (1, 3, 4):
        raise ValueError(f"a reference image needs 1, 3 or 4 channels, got {tuple(image.shape)}")
    image = image[:, :3]
    if image.shape[1] == 1:
        image = image.expand(-1, 3, -1, -1)
    return image.float()


def to_bhwc(pixels: torch.Tensor) -> torch.Tensor:
    return pixels.movedim(1, -1)


class ReferenceSet:
    def __init__(self, pixels: list[torch.Tensor], fingerprint: tuple):
        if not pixels:
            raise ValueError("a ReferenceSet needs at least one image")
        self.pixels = [to_bchw(image) for image in pixels]
        self.fingerprint = fingerprint
        self._vision: dict[int, list[torch.Tensor]] = {}
        self._encoded: dict[tuple, tuple[list[torch.Tensor], list[geometry.FitPlan]]] = {}

    def __len__(self) -> int:
        return len(self.pixels)

    @property
    def sizes(self) -> list[tuple[int, int]]:
        return [(int(p.shape[-2]), int(p.shape[-1])) for p in self.pixels]

    def vision(self, grounding_px: int) -> list[torch.Tensor]:
        """The images for Qwen3-VL, ``(1, h, w, 3)`` in [0, 1], capped at
        ``grounding_px`` on the longest side."""
        key = int(grounding_px or 0)
        if key not in self._vision:
            self._vision[key] = [to_bhwc(geometry.grounding_pixels(p, key)) for p in self.pixels]
        return self._vision[key]

    def has_encoded(self, target_h: int, target_w: int, geometry_name: str) -> bool:
        return (int(target_h), int(target_w), geometry_name) in self._encoded

    def encoded(self, engine, target_h: int, target_w: int, geometry_name: str = S.FIT) -> tuple[list[torch.Tensor], list[geometry.FitPlan]]:
        """The references as latents for a ``target_h`` x ``target_w`` pixel
        output: fitted, VAE-encoded through the engine's own VAE, normalised
        the way the engine normalises every latent, in the engine's latent
        layout ``(1, C, h, w)``."""
        key = (int(target_h), int(target_w), geometry_name)
        if key not in self._encoded:
            latents, plans = [], []
            for pixels in self.pixels:
                fitted, fit = geometry.fit_pixels(pixels, key[0], key[1], geometry_name)
                latents.append(encode_pixels(engine, fitted))
                plans.append(fit)
            self._encoded[key] = (latents, plans)
            logger.info("references for %dx%d: %s", key[1], key[0], "; ".join(p.describe() for p in plans))
        return self._encoded[key]

    def forget_encoded(self) -> None:
        self._encoded.clear()


def encode_pixels(engine, pixels: torch.Tensor) -> torch.Tensor:
    """``pixels`` (1, 3, H, W) in [0, 1] -> the engine's latent ``(1, C, h, w)``.

    The same two calls the engine makes for its own references: the VAE's
    ``encode`` on channels-last pixels, then ``process_in`` so the latent is in
    the space the transformer sees. A Wan-style VAE answers with a time axis of
    one, which is squeezed, as the engine squeezes it.
    """
    vae = engine.forge_objects.vae
    sample = vae.encode(to_bhwc(pixels).contiguous())
    latent = vae.first_stage_model.process_in(sample)
    if latent.ndim == 5:
        latent = latent.squeeze(2)
    return latent


def snapshot(engine, previous: ReferenceSet | None) -> ReferenceSet | None:
    """The references the engine holds right now, in native order, or ``None``.

    Consumes ``ini_latent`` the way the native conditioning does, so an img2img
    input is used once per generation and cannot leak into the next. The
    gallery's list is compared by the identity of its tensors: ImageStitch
    leaves it alone between two jobs with the same gallery, replaces its
    contents when the gallery changes, and clears it when its accordion closes.
    A set already built for this job is kept while the gallery is the same and
    no new img2img input has arrived, which is what makes the second batch, the
    hires pass and the negative-after-positive order see the same images.
    """
    gallery = list(getattr(engine, "ref_latents", None) or ())
    tail = tuple(id(t) for t in gallery)
    pending = getattr(engine, "ini_latent", None)

    if pending is not None:
        engine.ini_latent = None
        return ReferenceSet([pending, *gallery], (id(pending), tail))

    if previous is not None and previous.fingerprint[1] == tail:
        return previous

    if not gallery:
        return None
    return ReferenceSet(gallery, (None, tail))
