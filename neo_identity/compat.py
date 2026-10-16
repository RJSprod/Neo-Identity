"""What this extension needs from the host, checked before anything is changed.

The recipe lives in two of Forge Neo's classes: the Krea 2 engine
(``backend.diffusion_engine.krea.Krea2``), whose conditioning and first-stage
encode are wrapped, and the transformer (``backend.nn.krea.SingleStreamDiT``),
whose forward is wrapped and whose submodules the edit forward drives. Both are
read against the shapes this extension was written for, pinned to Forge Neo
``46af880c``. A host that disagrees fails here, once, with its reasons, and the
extension stays inert: a wrong guess about a method's arguments would otherwise
surface as a crash in the middle of somebody's generation.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass

logger = logging.getLogger("neo_identity")

PINNED_HOST = "Haoming02/sd-webui-forge-classic@46af880c003d31846bbea984ce00fc7be247ce5b"

DIT_ATTRIBUTES = ("first", "blocks", "tmlp", "tproj", "txtfusion", "txtmlp", "last", "pe_embedder")
"""The submodules the edit forward drives, and the names the pinned host gives them."""
DIT_FIELDS = ("patch", "channels", "tdim")

FORWARD_PARAMETERS = ("self", "x", "timesteps", "context", "attention_mask", "transformer_options")
CONDITIONING_PARAMETERS = ("self", "prompt")
ENCODE_PARAMETERS = ("self", "x")


@dataclass(frozen=True)
class Contract:
    ok: bool
    reasons: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        if self.ok:
            return "host contract satisfied"
        return "; ".join(self.reasons)


_checked: Contract | None = None


def classes():
    """``(Krea2, SingleStreamDiT)`` from the host. Raises ImportError off Forge Neo."""
    from backend.diffusion_engine.krea import Krea2
    from backend.nn.krea import SingleStreamDiT

    return Krea2, SingleStreamDiT


def check(force: bool = False) -> Contract:
    """Whether the host has the interfaces this extension wraps. Cached."""
    global _checked
    if _checked is not None and not force:
        return _checked

    reasons: list[str] = []
    try:
        krea2, dit = classes()
    except Exception as error:  # ImportError, or anything a half-present host raises
        _checked = Contract(False, (f"Forge Neo's Krea 2 engine is not importable ({type(error).__name__}: {error})",))
        return _checked

    _check_signature(reasons, krea2, "get_learned_conditioning", CONDITIONING_PARAMETERS)
    _check_signature(reasons, krea2, "encode_first_stage", ENCODE_PARAMETERS)
    _check_signature(reasons, dit, "forward", FORWARD_PARAMETERS)

    try:
        source = inspect.getsource(dit.__init__)
    except (OSError, TypeError):
        source = ""
    for name in DIT_ATTRIBUTES + DIT_FIELDS:
        if f"self.{name} =" not in source and f"self.{name}=" not in source:
            reasons.append(f"SingleStreamDiT no longer defines {name}")

    try:
        from backend.text_processing.krea2_engine import Qwen3VL4BEngine

        parameters = list(inspect.signature(Qwen3VL4BEngine.__call__).parameters)
        if parameters[:3] != ["self", "texts", "images"]:
            reasons.append(f"Qwen3VL4BEngine.__call__ takes {parameters}, not (texts, images)")
    except Exception as error:
        reasons.append(f"the Qwen3-VL text engine is not importable ({type(error).__name__}: {error})")

    _checked = Contract(not reasons, tuple(reasons))
    return _checked


def _check_signature(reasons: list[str], owner, name: str, expected: tuple[str, ...]) -> None:
    method = getattr(owner, name, None)
    if method is None:
        reasons.append(f"{owner.__name__} has no {name}")
        return
    try:
        parameters = list(inspect.signature(method).parameters)
    except (TypeError, ValueError):
        reasons.append(f"{owner.__name__}.{name} has no readable signature")
        return
    if parameters[: len(expected)] != list(expected):
        reasons.append(f"{owner.__name__}.{name} takes {parameters}, not {list(expected)}")


def is_krea2(model) -> bool:
    """Whether ``model`` is a loaded Krea 2 engine.

    By class name first, so a test double or a reloaded module still counts,
    then by isinstance for a host whose class has been subclassed.
    """
    if model is None:
        return False
    if type(model).__name__ == "Krea2":
        return True
    try:
        krea2, _ = classes()
    except Exception:
        return False
    return isinstance(model, krea2)


def attention_name() -> str:
    """The name of the attention function Forge selected at start-up."""
    try:
        from backend import attention

        return getattr(attention.attention_function, "__name__", type(attention.attention_function).__name__)
    except Exception:
        return "unknown"


def attention_honours_scale() -> bool:
    from .boost import HONOURS_SCALE

    return attention_name() in HONOURS_SCALE


def unwrap(module):
    """A module compiled by ``torch.compile`` stands in for the original; the
    original is what the edit forward needs and what the wrapper compares."""
    seen = 0
    while module is not None and hasattr(module, "_orig_mod") and seen < 8:
        module = module._orig_mod
        seen += 1
    return module


def is_compiled(module) -> bool:
    return module is not None and hasattr(module, "_orig_mod")


def diffusion_model_of(engine):
    """The engine's transformer, through its patcher and any compile wrapper."""
    try:
        return unwrap(engine.forge_objects.unet.model.diffusion_model)
    except AttributeError:
        return None
