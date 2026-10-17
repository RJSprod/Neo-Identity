"""Where the recipe meets the host: three wrapped methods and one job's state.

Forge Neo runs Krea 2 through two classes. The engine
(``backend.diffusion_engine.krea.Krea2``) captures reference pixels in
``encode_first_stage`` and turns prompts into conditioning in
``get_learned_conditioning``; the transformer (``backend.nn.krea.SingleStreamDiT``)
assembles the token sequence in ``forward``. This module wraps those three
methods on the *classes*, once, with dispatchers that do nothing unless a
generation has switched the recipe on. Off, every call goes straight to the
original, so the host behaves as if the extension were not installed; on, the
dispatchers read the references the host captured, ground both prompts on them,
and run the edit forward with the V2 geometry, placement and boost.

Why the classes and not the generation's own objects: Model Chain refines its
Stage 2 through a processing object with the scripts switched off, on a model
it loads itself. A wrapper hung on the first stage's model or installed from a
script hook would never see that pass. A class-level dispatcher sees every Krea
2 engine the host creates, and the generation's own accordion decides whether
it acts.

The state is one job at a time, which is what the host runs, and it lives
from one generation's ``before_process`` to the next one's. ``activate`` is
called from ``before_process``; the script's ``postprocess`` only *releases*
the job's tensors and keeps the recipe armed, because Model Chain's Stage 2
runs inside the generation's ``postprocess`` phase and Forge runs the scripts'
``postprocess`` hooks in load order -- this extension's may well run before
Model Chain's. The next generation's ``before_process`` resets or re-arms the
state before anything samples, so nothing leaks across generations; a
generation that dies half-way is reset the same way.
"""

from __future__ import annotations

import logging
import weakref

import torch

from . import compat
from . import forward as edit
from . import geometry as G
from . import references as R
from .settings import Settings

logger = logging.getLogger("neo_identity")

TAG = "__neo_identity__"
ORIGINAL = "__neo_identity_original__"


class State:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.active: bool = False
        self.settings: Settings | None = None
        self.references: R.ReferenceSet | None = None
        self._engine = None
        self.notes: list[str] = []
        self.used: int = 0
        """How many references the last grounded conditioning used."""
        self.serial: int = getattr(self, "serial", 0)

    @property
    def engine(self):
        return self._engine() if self._engine is not None else None

    def bind(self, engine) -> None:
        self._engine = weakref.ref(engine)

    def note(self, text: str) -> None:
        """Something the user should read once, on the console now and on the
        result later."""
        if text not in self.notes:
            self.notes.append(text)
            logger.warning(text)


STATE = State()
_originals: dict = {}


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #


def install() -> compat.Contract:
    """Wrap the host's three methods, once. Idempotent, and safe across a reload
    of this module: a dispatcher left on a class by an earlier copy of the module
    is recognised by its tag and replaced, with its original kept."""
    contract = compat.check()
    if not contract.ok:
        return contract
    if _originals:
        return contract

    krea2, dit = compat.classes()
    _wrap(krea2, "get_learned_conditioning", _conditioning)
    _wrap(krea2, "encode_first_stage", _encode_first_stage)
    _wrap(dit, "forward", _forward)
    logger.info("installed on %s and %s", krea2.__name__, dit.__name__)
    return contract


def _wrap(owner, name: str, dispatcher) -> None:
    current = owner.__dict__.get(name)
    original = getattr(current, ORIGINAL, None) if getattr(current, TAG, False) else current
    if original is None:
        raise RuntimeError(f"{owner.__name__}.{name} is not defined on the class")
    _originals[name] = original
    setattr(dispatcher, TAG, True)
    setattr(dispatcher, ORIGINAL, original)
    setattr(owner, name, dispatcher)


def uninstall() -> None:
    """Put the originals back. Used by the tests and by a script reload."""
    if not _originals:
        return
    try:
        krea2, dit = compat.classes()
    except Exception:
        _originals.clear()
        return
    for owner, name in ((krea2, "get_learned_conditioning"), (krea2, "encode_first_stage"), (dit, "forward")):
        original = _originals.get(name)
        if original is not None and getattr(owner.__dict__.get(name), TAG, False):
            setattr(owner, name, original)
    _originals.clear()
    STATE.reset()


def installed() -> bool:
    return bool(_originals)


def original(name: str):
    return _originals[name]


# --------------------------------------------------------------------------- #
# The job
# --------------------------------------------------------------------------- #


def activate(settings: Settings) -> None:
    """Switch the recipe on for the generation that is starting."""
    STATE.reset()
    STATE.serial += 1
    STATE.active = True
    STATE.settings = settings


def deactivate() -> list[str]:
    """Switch the recipe off and hand back what it wanted to say."""
    notes = list(STATE.notes)
    STATE.reset()
    return notes


def release() -> list[str]:
    """The generation's own passes are over: drop its tensors and hand back its
    notes, but stay armed with its settings for a Krea 2 pass that another
    script still runs inside this generation (Model Chain's Stage 2). The next
    generation's ``before_process`` resets or re-arms the state."""
    notes = list(STATE.notes)
    STATE.references = None
    STATE._engine = None
    STATE.notes = []
    STATE.used = 0
    return notes


def references_for(engine) -> R.ReferenceSet | None:
    """The engine's references for this job, snapshotted in native order."""
    refs = R.snapshot(engine, STATE.references)
    STATE.references = refs
    if refs is not None:
        STATE.bind(engine)
    return refs


def owns(dit) -> bool:
    """Whether ``dit`` is the transformer of the engine whose conditioning this
    job grounded. Anything else -- another model, a transformer this job never
    conditioned -- runs natively."""
    engine = STATE.engine
    if engine is None:
        return False
    model = compat.diffusion_model_of(engine)
    return model is not None and (model is dit or compat.unwrap(dit) is model)


def target_size(pixel_h: int, pixel_w: int) -> tuple[int, int]:
    """The pixel size the forward will fit references to for an output of
    ``pixel_h`` x ``pixel_w``: the transformer pads the latent to its patch grid,
    so a size that is not a multiple of 16 rounds up."""
    rows, cols = G.padded_grid(max(1, int(pixel_h)) // 8, max(1, int(pixel_w)) // 8, 2)
    return rows * G.PATCH_PX, cols * G.PATCH_PX


def preencode(engine, pixel_h: int, pixel_w: int) -> bool:
    """Encode this job's references for an output of ``pixel_h`` x ``pixel_w``
    now, before sampling, so the VAE never runs inside the step loop. True when
    something was encoded or already there."""
    refs = STATE.references
    if not (STATE.active and STATE.settings is not None and refs is not None):
        return False
    th, tw = target_size(pixel_h, pixel_w)
    refs.encoded(engine, th, tw, STATE.settings.geometry)
    return True


def invalidate_prompt_cache() -> None:
    """Make the host recompute its conditioning.

    The host caches conditioning on the prompt text, the steps and the extra
    networks -- never on the images the Krea 2 engine grounds it with. When a
    new reference is captured the cached conditioning describes the old one.
    ImageStitch clears the cache through its processing object; the engine has
    no processing object in hand, so the class-level lists every new processing
    object shares are emptied in place. A processing object that already
    replaced its own lists is cleared by the script, which has it.
    """
    try:
        from modules import processing
    except Exception:
        return
    for owner_name in ("StableDiffusionProcessing", "StableDiffusionProcessingTxt2Img"):
        owner = getattr(processing, owner_name, None)
        if owner is None:
            continue
        for name in ("cached_c", "cached_uc", "cached_hr_c", "cached_hr_uc"):
            cache = vars(owner).get(name)
            if isinstance(cache, list) and cache:
                cache[:] = [None] * len(cache)


# --------------------------------------------------------------------------- #
# Dispatchers
# --------------------------------------------------------------------------- #


def _engaged(engine) -> bool:
    return STATE.active and STATE.settings is not None and compat.is_krea2(engine)


@torch.inference_mode()
def _encode_first_stage(engine, x):
    """The engine's capture, then: a new reference means the cached conditioning
    and this job's fitted latents are stale."""
    if not _engaged(engine):
        return _originals["encode_first_stage"](engine, x)
    before = (len(getattr(engine, "ref_latents", ())), getattr(engine, "ini_latent", None) is not None)
    out = _originals["encode_first_stage"](engine, x)
    after = (len(getattr(engine, "ref_latents", ())), getattr(engine, "ini_latent", None) is not None)
    if after != before:
        invalidate_prompt_cache()
        STATE.references = None
    return out


@torch.inference_mode()
def _conditioning(engine, prompt):
    """Both prompts grounded on the references, in native order, at the
    grounding size; the fitted latents encoded for the announced output size."""
    if not _engaged(engine):
        return _originals["get_learned_conditioning"](engine, prompt)

    negative = bool(getattr(prompt, "is_negative_prompt", False))
    refs = references_for(engine)
    if refs is None:
        if not negative:
            STATE.note("no reference image reached Krea 2, so this generation ran the native text-only path: add references with ImageStitch, an img2img image, or Model Chain's Stage 2 references")
        return _originals["get_learned_conditioning"](engine, prompt)

    try:
        wrapped = engine.forge_objects.unet.model.diffusion_model
    except AttributeError:
        wrapped = None
    if compat.is_compiled(wrapped):
        STATE.note("the transformer is compiled (torch.compile), which Identity Edit does not drive; this generation ran the native path")
        STATE.references = None
        return _originals["get_learned_conditioning"](engine, prompt)

    from backend import memory_management
    from backend.args import dynamic_args

    memory_management.load_model_gpu(engine.forge_objects.clip.patcher)
    # The native forward injects whatever this list holds. Ours ignores it, but
    # nothing from an earlier job may wait there for a pass that runs natively.
    dynamic_args.ref_latents.clear()

    settings = STATE.settings
    if negative and not settings.ground_negative:
        conds = engine.text_processing_engine_qwen(prompt)
    else:
        conds = engine.text_processing_engine_qwen(prompt, images=refs.vision(settings.grounding_px))
    STATE.used = len(refs)
    if not negative:
        logger.info("%d reference(s), %s: %s", len(refs), ", ".join(f"{w}x{h}" for h, w in refs.sizes), settings.describe(len(refs)))

    width, height = getattr(prompt, "width", None), getattr(prompt, "height", None)
    if width and height:
        try:
            preencode(engine, int(height), int(width))
        except Exception:
            logger.warning("could not encode the references before sampling; they will be encoded at the first step", exc_info=True)
    return conds


def _forward(dit, x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
    """The edit forward for the transformer this job conditioned; the native
    forward for everything else."""
    if transformer_options is None:
        transformer_options = {}
    refs = STATE.references
    if not (STATE.active and STATE.settings is not None and refs is not None and owns(dit)):
        return _originals["forward"](dit, x, timesteps, context, attention_mask, transformer_options, **kwargs)

    engine = STATE.engine
    rows, cols = G.padded_grid(int(x.shape[-2]), int(x.shape[-1]), dit.patch)
    latents, _ = refs.encoded(engine, rows * G.PATCH_PX, cols * G.PATCH_PX, STATE.settings.geometry)
    count = len(latents)
    return edit.edit_forward(
        dit,
        x,
        timesteps,
        context,
        latents,
        STATE.settings.placements(count),
        STATE.settings.boosts(count),
        transformer_options,
        honours_scale=compat.attention_honours_scale(),
    )
