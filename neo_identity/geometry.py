"""Reference geometry: how a source image fits the target grid, and where its
tokens sit in position space.

Ported from the Krea2Edit nodes (lbouaraba/comfyui-krea2edit, Apache-2.0) and
checked against the trainer's geometry contract (krea2edit-trainer,
``docs/GEOMETRY.md``). Geometry is the thing that misregisters: a reference whose
grid or offset differs from training's by a pixel shows up as blur, ghosting or
a doubled seam, so these numbers are reproduced exactly rather than approximated.

Everything here is plain torch on tensors the caller already has; nothing
touches the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import settings as S

PATCH_PX = 16
"""One DiT token covers 16 pixels: the VAE compresses 8x and the transformer
patchifies 2x2 on top of it."""

CROP_TOLERANCE = 0.08
"""Within 8 % of the target's aspect ratio the source is centre-cropped to the
exact ratio and fills the grid completely. Fit-inside margins of a token or two
are not harmless: target edge columns with no reference correspondence get
filled by repeating adjacent reference content (upstream's 2026-07-14
edge-duplication finding). It also keeps the promise that *fit* equals *crop*
at a matched aspect ratio."""


@dataclass(frozen=True)
class FitPlan:
    """How one source image becomes reference pixels for one target size.

    Pixels throughout. ``crop`` is the window taken from the source before the
    resize, ``output`` the size the window is resampled to. The output is always
    a multiple of 16 on both sides and never larger than the target.
    """

    source: tuple[int, int]
    """(height, width) of the source."""
    target: tuple[int, int]
    """(height, width) of the target, as handed in."""
    crop: tuple[int, int, int, int]
    """(y0, x0, height, width) of the window taken from the source."""
    output: tuple[int, int]
    """(height, width) the window is resampled to."""
    geometry: str

    @property
    def grid(self) -> tuple[int, int]:
        """The reference's token grid (rows, columns)."""
        return self.output[0] // PATCH_PX, self.output[1] // PATCH_PX

    @property
    def fills_target(self) -> bool:
        return self.output == self.target

    def describe(self) -> str:
        sh, sw = self.source
        y0, x0, ch, cw = self.crop
        oh, ow = self.output
        crop = "" if (ch, cw) == (sh, sw) else f", crop {cw}x{ch} at ({x0}, {y0})"
        return f"{sw}x{sh} -> {ow}x{oh}{crop}, grid {self.grid[1]}x{self.grid[0]} ({self.geometry.lower()})"


def plan(source_h: int, source_w: int, target_h: int, target_w: int, geometry: str = S.FIT) -> FitPlan:
    """The plan for ``geometry`` (``settings.FIT`` or ``settings.CROP``)."""
    if geometry == S.CROP:
        return plan_crop(source_h, source_w, target_h, target_w)
    return plan_fit(source_h, source_w, target_h, target_w)


def plan_fit(ih: int, iw: int, px_h: int, px_w: int) -> FitPlan:
    """The v1.2+ *fit* geometry, as the trainer's ``_fit_prep`` and the node's
    ``_fit_encode_image`` compute it.

    Scale the source to fit inside the target. Near the target's aspect ratio,
    centre-crop to that ratio and fill the whole grid. Otherwise snap the fitted
    size down to the 16 px grid (never above the target's own snapped size) and
    centre-crop the source so that the fitted axis lands on that grid at the
    same scale exactly: a resize from ``ih * sc`` down to the snapped size would
    squash content by up to 15 px, and that misregistration peaks at the
    reference band's edges, where it renders as a doubled seam.
    """
    _check(ih, iw, px_h, px_w)
    sc = min(px_h / ih, px_w / iw)
    if ih * sc >= px_h * (1 - CROP_TOLERANCE) and iw * sc >= px_w * (1 - CROP_TOLERANCE):
        y0, x0, ch, cw = _crop_to_ratio(ih, iw, px_h, px_w)
        return FitPlan((ih, iw), (px_h, px_w), (y0, x0, ch, cw), (px_h, px_w), S.FIT)

    nh = min(max(PATCH_PX, int(ih * sc) // PATCH_PX * PATCH_PX), max(PATCH_PX, px_h // PATCH_PX * PATCH_PX))
    nw = min(max(PATCH_PX, int(iw * sc) // PATCH_PX * PATCH_PX), max(PATCH_PX, px_w // PATCH_PX * PATCH_PX))
    ch = min(ih, max(1, int(round(nh / sc))))
    cw = min(iw, max(1, int(round(nw / sc))))
    y0, x0 = (ih - ch) // 2, (iw - cw) // 2
    return FitPlan((ih, iw), (px_h, px_w), (y0, x0, ch, cw), (nh, nw), S.FIT)


def plan_crop(ih: int, iw: int, px_h: int, px_w: int) -> FitPlan:
    """The v1 and v1.1 *crop* geometry: centre-crop to the target's aspect
    ratio, then resize to the target."""
    _check(ih, iw, px_h, px_w)
    y0, x0, ch, cw = _crop_to_ratio(ih, iw, px_h, px_w)
    return FitPlan((ih, iw), (px_h, px_w), (y0, x0, ch, cw), (px_h, px_w), S.CROP)


def _crop_to_ratio(ih: int, iw: int, px_h: int, px_w: int) -> tuple[int, int, int, int]:
    s = max(px_h / ih, px_w / iw)
    ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
    return (ih - ch) // 2, (iw - cw) // 2, ch, cw


def _check(ih: int, iw: int, px_h: int, px_w: int) -> None:
    if min(ih, iw, px_h, px_w) <= 0:
        raise ValueError(f"geometry needs positive sizes, got source {iw}x{ih} and target {px_w}x{px_h}")


def apply_plan(pixels: torch.Tensor, fit: FitPlan) -> torch.Tensor:
    """Crop and resample ``pixels`` (B, C, H, W) as ``fit`` says.

    Bicubic with antialiasing in float, as upstream resamples; the result is
    clamped to the image range, because bicubic overshoots at edges and the VAE
    is only defined on [0, 1].
    """
    if tuple(pixels.shape[-2:]) != fit.source:
        raise ValueError(f"plan is for a {fit.source} source, pixels are {tuple(pixels.shape[-2:])}")
    y0, x0, ch, cw = fit.crop
    window = pixels[..., y0 : y0 + ch, x0 : x0 + cw]
    if tuple(window.shape[-2:]) == fit.output:
        return window.float().clamp(0, 1)
    return F.interpolate(window.float(), size=fit.output, mode="bicubic", antialias=True).clamp(0, 1)


def fit_pixels(pixels: torch.Tensor, target_h: int, target_w: int, geometry: str = S.FIT) -> tuple[torch.Tensor, FitPlan]:
    """``pixels`` (B, C, H, W) prepared for a ``target_h`` x ``target_w`` output."""
    fit = plan(int(pixels.shape[-2]), int(pixels.shape[-1]), int(target_h), int(target_w), geometry)
    return apply_plan(pixels, fit), fit


def grounding_pixels(pixels: torch.Tensor, grounding_px: int) -> torch.Tensor:
    """``pixels`` (B, C, H, W) for Qwen3-VL: the longest side capped at
    ``grounding_px``, never enlarged, untouched when ``grounding_px`` is 0.

    Area interpolation, as upstream's ``common_upscale(..., "area")``. The
    semantic path only needs the gist; fine detail reaches the DiT through the
    VAE tokens, so capping here trims vision tokens by about (native/cap)^2.
    """
    grounding_px = int(grounding_px or 0)
    h, w = int(pixels.shape[-2]), int(pixels.shape[-1])
    if grounding_px <= 0 or max(h, w) <= grounding_px:
        return pixels
    scale = grounding_px / max(h, w)
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    return F.interpolate(pixels.float(), size=size, mode="area")


# --------------------------------------------------------------------------- #
# Position ids
# --------------------------------------------------------------------------- #


def target_ids(batch: int, rows: int, cols: int, device) -> torch.Tensor:
    """Frame 0, stride-1 (h, w): the target's positions, as native Krea 2 lays them."""
    return _grid_ids(batch, 0, rows, cols, 0.0, 0.0, device)


def centered_offset(target_tokens: int, reference_tokens: int) -> float:
    """Where a reference grid starts inside the target grid along one axis.

    Fractional on purpose: an integer floor put odd-gap references half a token
    (8 px) off their true centre, and RoPE is continuous, so a half-token
    position is exact. Never negative: a fitted reference is never larger than
    the target.
    """
    return max(0.0, (target_tokens - reference_tokens) / 2)


def reference_ids(batch: int, frame: int, rows: int, cols: int, target_rows: int, target_cols: int, device) -> torch.Tensor:
    """Frame ``frame`` (1 for the first reference), stride-1 (h, w) at the
    centred offset inside the target grid."""
    if frame < 1:
        raise ValueError("references take frames from 1; frame 0 is the target")
    return _grid_ids(batch, frame, rows, cols, centered_offset(target_rows, rows), centered_offset(target_cols, cols), device)


def _grid_ids(batch: int, frame: int, rows: int, cols: int, off_h: float, off_w: float, device) -> torch.Tensor:
    ids = torch.zeros(rows, cols, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(rows, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(cols, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, rows * cols, 3).repeat(batch, 1, 1)


def place(reference_ids_list: list[torch.Tensor], placements: list[str], target_rows: int, target_cols: int) -> list[torch.Tensor]:
    """Apply each reference's placement to its ids. The frame axis is untouched.

    ``Aligned`` keeps the ids as they are. ``Beside`` shifts the reference to
    the right of the target canvas, and of every reference placed beside before
    it, so no two share a column; its vertical placement and internal layout are
    unchanged. ``Bag of features`` gives every token the target's centre.
    """
    if len(placements) != len(reference_ids_list):
        raise ValueError(f"{len(placements)} placements for {len(reference_ids_list)} references")
    out = []
    next_column = float(target_cols)
    for ids, placement in zip(reference_ids_list, placements):
        if placement == S.ALIGNED:
            out.append(ids)
            continue
        ids = ids.clone()
        if placement == S.BESIDE:
            ids[..., 2] = ids[..., 2] - ids[..., 2].min() + next_column
            next_column = float(ids[..., 2].max()) + 1.0
        elif placement == S.BAG:
            ids[..., 1] = (target_rows - 1) / 2.0
            ids[..., 2] = (target_cols - 1) / 2.0
        else:
            raise ValueError(f"unknown placement {placement!r}")
        out.append(ids)
    return out


def padded_grid(height: int, width: int, patch: int) -> tuple[int, int]:
    """The token grid of a latent of ``height`` x ``width`` after the DiT pads
    it to its patch size: what the references are fitted against."""
    rows = math.ceil(height / patch)
    cols = math.ceil(width / patch)
    return rows, cols
