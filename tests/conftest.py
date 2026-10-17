"""Test harness: a stand-in for the Forge Neo host, with the real Krea 2
transformer when a Forge checkout is in reach.

The extension is normally imported by the WebUI, which supplies ``modules.*``,
``backend.*`` and ``gradio``. These tests install minimal fakes of those into
``sys.modules`` before anything of the extension is imported. The fakes are
faithful where the extension depends on the host's behaviour -- the Krea 2
engine's capture of references and its native conditioning, the processing
object's class-level prompt cache, the attention function's override protocol --
and stubs everywhere else.

``backend.nn.krea`` is the host's own module, loaded from the Forge checkout
``tests/forge_source.py`` finds, so the parity tests run the real transformer;
without a checkout it is a small fake with the same interface and the tests that
need the real one skip.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))

import forge_source  # noqa: E402

FORGE_ROOT = forge_source.forge_root()


# --------------------------------------------------------------------------- #
# gradio
# --------------------------------------------------------------------------- #


class Component:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.value = kwargs.get("value", args[0] if args else None)
        self.label = kwargs.get("label")
        self.elem_id = kwargs.get("elem_id")
        self.choices = kwargs.get("choices")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _gradio():
    gr = types.ModuleType("gradio")
    for name in ("HTML", "Row", "Column", "Radio", "Slider", "Checkbox", "Markdown", "Textbox", "Image", "Gallery", "Button", "Accordion"):
        setattr(gr, name, type(name, (Component,), {}))
    return gr


# --------------------------------------------------------------------------- #
# modules.*
# --------------------------------------------------------------------------- #


class Options:
    def __init__(self):
        self.krea2_do_reference = False
        self.img2img_background_color = "#ffffff"


class ShoutingReport(Exception):
    pass


def _modules(host):
    modules = types.ModuleType("modules")

    errors = types.ModuleType("modules.errors")
    errors.reports = []

    def report(message, *, exc_info=False):
        errors.reports.append(message)

    errors.report = report

    shared = types.ModuleType("modules.shared")
    shared.opts = host.opts
    shared.sd_model = None
    shared.device = torch.device("cpu")

    scripts = types.ModuleType("modules.scripts")
    scripts.AlwaysVisible = object()

    class Script:
        sorting_priority = 0
        alwayson = True
        is_img2img = False
        infotext_fields = None
        paste_field_names = None

        def __init__(self):
            pass

        def elem_id(self, item_id):
            tab = "img2img" if self.is_img2img else "txt2img"
            return f"{tab}_neo_identity_{item_id}"

        def title(self):
            raise NotImplementedError

    scripts.Script = Script

    ui_components = types.ModuleType("modules.ui_components")

    class InputAccordion(Component):
        def __init__(self, value=None, **kwargs):
            super().__init__(value=value, **kwargs)

    ui_components.InputAccordion = InputAccordion

    processing = types.ModuleType("modules.processing")

    class StableDiffusionProcessing:
        cached_uc = [None, None, None]
        cached_c = [None, None, None]

        def __init__(self, prompt="a prompt", width=1024, height=1024, sd_model=None, batch_size=1, n_iter=1):
            self.prompt = prompt
            self.negative_prompt = ""
            self.width = width
            self.height = height
            self.sd_model = sd_model
            self.batch_size = batch_size
            self.n_iter = n_iter
            self.override_settings = {}
            self.extra_generation_params = {}
            self.comments = {}
            self.cached_uc = StableDiffusionProcessing.cached_uc
            self.cached_c = StableDiffusionProcessing.cached_c
            self.scripts = None
            self.cleared = 0

        def clear_prompt_cache(self):
            self.cleared += 1
            self.cached_c = [None, None, None]
            self.cached_uc = [None, None, None]
            StableDiffusionProcessing.cached_c = [None, None, None]
            StableDiffusionProcessing.cached_uc = [None, None, None]

        def comment(self, text):
            self.comments[text] = 1

    class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
        cached_hr_uc = [None, None, None]
        cached_hr_c = [None, None, None]

        def __init__(self, *args, enable_hr=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.enable_hr = enable_hr
            self.cached_hr_uc = StableDiffusionProcessingTxt2Img.cached_hr_uc
            self.cached_hr_c = StableDiffusionProcessingTxt2Img.cached_hr_c

    class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
        def __init__(self, *args, init_images=None, denoising_strength=1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.init_images = init_images or []
            self.denoising_strength = denoising_strength

    class Processed:
        def __init__(self):
            self.comments = ""

    processing.StableDiffusionProcessing = StableDiffusionProcessing
    processing.StableDiffusionProcessingTxt2Img = StableDiffusionProcessingTxt2Img
    processing.StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2Img
    processing.Processed = Processed

    prompt_parser = types.ModuleType("modules.prompt_parser")

    class SdConditioning(list):
        def __init__(self, prompts, is_negative_prompt=False, width=None, height=None):
            super().__init__(prompts)
            self.is_negative_prompt = is_negative_prompt
            self.width = width
            self.height = height

    prompt_parser.SdConditioning = SdConditioning

    modules.errors = errors
    modules.shared = shared
    modules.scripts = scripts
    modules.ui_components = ui_components
    modules.processing = processing
    modules.prompt_parser = prompt_parser
    return {
        "modules": modules,
        "modules.errors": errors,
        "modules.shared": shared,
        "modules.scripts": scripts,
        "modules.ui_components": ui_components,
        "modules.processing": processing,
        "modules.prompt_parser": prompt_parser,
    }


# --------------------------------------------------------------------------- #
# backend.*
# --------------------------------------------------------------------------- #


def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    """Forge's PyTorch attention, as far as the Krea blocks exercise it."""
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.reshape(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    extra = {key: value for key, value in kwargs.items() if key in ("scale",)}
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, **extra)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def wrap_attn(func):
    """Forge's ``wrap_attn``: the override gets the undecorated function."""

    def wrapper(*args, **kwargs):
        options = kwargs.get("transformer_options", {}) or {}
        override = options.get("optimized_attention_override")
        if override is not None:
            return override(func, *args, **kwargs)
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__wrapped__ = func
    return wrapper


def apply_rope(xq, xk, freqs_cis):
    """Flux's rotary application (black-forest-labs/flux, Apache-2.0), which
    comfy-kitchen's ``ck.apply_rope`` implements for the host."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def _pad_to_patch_size_fallback(img, patch_size=(2, 2), padding_mode="circular"):
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad
    return torch.nn.functional.pad(img, pad, mode=padding_mode)


def _timestep_embedding_fallback(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(t) if torch.is_floating_point(t) else embedding


def _backend(host):
    backend = types.ModuleType("backend")

    args = types.ModuleType("backend.args")

    class dynamic_args:
        krea2 = True
        ref_latents: list = []
        is_referencing = False
        last_extra_generation_params: dict = {}

        @classmethod
        def can_reference(cls):
            return cls.krea2

    args.dynamic_args = dynamic_args

    memory_management = types.ModuleType("backend.memory_management")
    memory_management.loaded = []
    memory_management.load_model_gpu = lambda patcher: memory_management.loaded.append(patcher)
    memory_management.cast_to = lambda weight, dtype=None, device=None, **_: weight.to(dtype=dtype, device=device)
    memory_management.is_device_mps = lambda device: False
    memory_management.is_intel_xpu = lambda: False
    memory_management.is_directml_enabled = lambda: False
    memory_management.soft_empty_cache = lambda *a, **k: None

    attention = types.ModuleType("backend.attention")
    attention.attention_pytorch = wrap_attn(attention_pytorch)
    attention.attention_function = attention.attention_pytorch

    quant_ops = types.ModuleType("backend.quant_ops")
    quant_ops.ck = types.SimpleNamespace(apply_rope=apply_rope)

    utils = types.ModuleType("backend.utils")
    flux = types.ModuleType("backend.nn.flux")
    nn = types.ModuleType("backend.nn")
    misc = types.ModuleType("backend.misc")
    image_resize = types.ModuleType("backend.misc.image_resize")

    if FORGE_ROOT is not None:
        ns = {"torch": torch, "__name__": "backend.utils"}
        forge_source.load_definitions(FORGE_ROOT, "backend/utils.py", ("pad_to_patch_size",), ns)
        utils.pad_to_patch_size = ns["pad_to_patch_size"]

        from einops import rearrange, repeat

        ns = {"torch": torch, "math": math, "nn": torch.nn, "rearrange": rearrange, "repeat": repeat, "memory_management": memory_management, "__name__": "backend.nn.flux"}
        forge_source.load_definitions(FORGE_ROOT, "backend/nn/flux.py", ("rope", "timestep_embedding", "EmbedND"), ns)
        flux.rope, flux.timestep_embedding, flux.EmbedND = ns["rope"], ns["timestep_embedding"], ns["EmbedND"]

        import numpy as np
        from PIL import Image

        ns = {"torch": torch, "np": np, "Image": Image, "__name__": "backend.misc.image_resize"}
        forge_source.load_definitions(FORGE_ROOT, "backend/misc/image_resize.py", ("bislerp", "lanczos", "adaptive_resize"), ns)
        image_resize.adaptive_resize = ns["adaptive_resize"]
    else:
        utils.pad_to_patch_size = _pad_to_patch_size_fallback
        flux.timestep_embedding = _timestep_embedding_fallback

        class EmbedND(torch.nn.Module):
            def __init__(self, dim, theta, axes_dim):
                super().__init__()
                self.dim, self.theta, self.axes_dim = dim, theta, axes_dim

            def forward(self, ids):
                parts = []
                for i in range(ids.shape[-1]):
                    dim = self.axes_dim[i]
                    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
                    omega = 1.0 / (self.theta**scale)
                    out = torch.einsum("...n,d->...nd", ids[..., i].to(torch.float32), omega.to(torch.float32))
                    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
                    parts.append(out.reshape(*out.shape[:-1], 2, 2))
                return torch.cat(parts, dim=-3).unsqueeze(1)

        flux.EmbedND = EmbedND

        def adaptive_resize(samples, width, height, upscale_method, crop):
            if crop == "center":
                old_w, old_h = samples.shape[-1], samples.shape[-2]
                old_aspect, new_aspect = old_w / old_h, width / height
                x = y = 0
                if old_aspect > new_aspect:
                    x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
                elif old_aspect < new_aspect:
                    y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)
                samples = samples.narrow(-2, y, old_h - y * 2).narrow(-1, x, old_w - x * 2)
            return torch.nn.functional.interpolate(samples, size=(height, width), mode=upscale_method)

        image_resize.adaptive_resize = adaptive_resize

    backend.args = args
    backend.memory_management = memory_management
    backend.attention = attention
    backend.quant_ops = quant_ops
    backend.utils = utils
    backend.nn = nn
    nn.flux = flux
    backend.misc = misc
    misc.image_resize = image_resize
    return {
        "backend": backend,
        "backend.args": args,
        "backend.memory_management": memory_management,
        "backend.attention": attention,
        "backend.quant_ops": quant_ops,
        "backend.utils": utils,
        "backend.nn": nn,
        "backend.nn.flux": flux,
        "backend.misc": misc,
        "backend.misc.image_resize": image_resize,
    }


def _krea_module(stubs):
    """``backend.nn.krea``: the host's own file when a checkout is in reach."""
    if FORGE_ROOT is not None:
        return forge_source.load_module_from(FORGE_ROOT, "backend/nn/krea.py", "backend.nn.krea", sys.modules)

    krea = types.ModuleType("backend.nn.krea")

    class _Attn(torch.nn.Module):
        def __init__(self, headdim):
            super().__init__()
            self.headdim = headdim

    class _Block(torch.nn.Module):
        def __init__(self, features, heads):
            super().__init__()
            self.attn = _Attn(features // heads)
            self.lin = torch.nn.Linear(features, features)

        def forward(self, x, vec, freqs, mask=None, transformer_options={}):
            return x + self.lin(x)

    class SingleStreamDiT(torch.nn.Module):
        """A stand-in with the pinned interface; its forward records calls."""

        def __init__(self, features=16, tdim=8, txtdim=8, heads=2, kvheads=1, multiplier=1, layers=1, patch=2, channels=4, txtlayers=3, **kwargs):
            super().__init__()
            # One assignment per field, as the host writes them: the contract
            # check reads the constructor's source for ``self.<name> =``.
            self.patch = patch
            self.channels = channels
            self.tdim = tdim
            self.first = torch.nn.Linear(channels * patch**2, features)
            self.blocks = torch.nn.ModuleList([_Block(features, heads) for _ in range(layers)])
            self.tmlp = torch.nn.Linear(tdim, features)
            self.tproj = torch.nn.Linear(features, features * 6)
            self.txtfusion = torch.nn.Identity()
            self.txtmlp = torch.nn.Linear(txtdim, features)
            self.last = torch.nn.Linear(features, patch * patch * channels)
            self.pe_embedder = torch.nn.Identity()
            self.calls = []

        def forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
            self.calls.append(("native", tuple(x.shape)))
            return x * 2.0

    krea.SingleStreamDiT = SingleStreamDiT
    return krea


def _engine_module(stubs, krea):
    """``backend.diffusion_engine.krea``: a Krea2 with the host's capture and
    conditioning behaviour, around a fake VAE and text engine."""
    dynamic_args = stubs["backend.args"].dynamic_args
    opts = stubs["modules.shared"].opts
    memory_management = stubs["backend.memory_management"]

    class FakeVAE:
        def __init__(self):
            self.encodes = []
            self.first_stage_model = types.SimpleNamespace(process_in=lambda x: x * 1.0)
            self.patcher = object()

        def encode(self, pixels):
            self.encodes.append(tuple(pixels.shape))
            b, h, w, c = pixels.shape
            latent = torch.zeros(1, 16, b, h // 8, w // 8)
            latent[0, 0, :, 0, 0] = h
            latent[0, 1, :, 0, 0] = w
            return latent

    class FakeQwen:
        def __init__(self):
            self.calls = []

        def __call__(self, texts, images=[]):
            self.calls.append((list(texts), [tuple(i.shape) for i in images], bool(getattr(texts, "is_negative_prompt", False))))
            return [torch.zeros(4 + len(images), 12, 8) for _ in texts]

    class Krea2:
        matched_guesses = []

        def __init__(self, dit=None):
            self.ref_latents = []
            self.ini_latent = None
            self.text_processing_engine_qwen = FakeQwen()
            self.forge_objects = types.SimpleNamespace(
                vae=FakeVAE(),
                clip=types.SimpleNamespace(patcher=object()),
                unet=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dit)),
            )
            self.native_conditioning = []
            self.encoded = []

        @torch.inference_mode()
        def get_learned_conditioning(self, prompt):
            memory_management.load_model_gpu(self.forge_objects.clip.patcher)
            self.native_conditioning.append(list(prompt))
            if not prompt.is_negative_prompt:
                _references = [*self.ref_latents]
                if self.ini_latent is not None:
                    _references.insert(0, self.ini_latent)
                    self.ini_latent = None
                if opts.krea2_do_reference and bool(_references):
                    return self.get_learned_conditioning_with_image(prompt, _references)
                else:
                    dynamic_args.ref_latents.clear()
            return self.text_processing_engine_qwen(prompt)

        @torch.inference_mode()
        def get_learned_conditioning_with_image(self, prompt, images):
            dynamic_args.ref_latents = [torch.zeros(1, 16, 8, 8) for _ in images]
            return self.text_processing_engine_qwen(prompt, images=images)

        @torch.inference_mode()
        def encode_first_stage(self, x):
            if opts.krea2_do_reference:
                start_image = x[0].movedim(0, -1).mul(0.5).add(0.5).unsqueeze(0)
                if dynamic_args.is_referencing:
                    self.ref_latents.append(start_image.cpu())
                else:
                    self.ini_latent = start_image.cpu()
            self.encoded.append(tuple(x.shape))
            return torch.zeros(1, 16, 1, x.shape[-2] // 8, x.shape[-1] // 8)

        def clear_references(self):
            self.ref_latents.clear()

    diffusion_engine = types.ModuleType("backend.diffusion_engine")
    engine = types.ModuleType("backend.diffusion_engine.krea")
    engine.Krea2 = Krea2
    engine.FakeVAE = FakeVAE
    diffusion_engine.krea = engine

    text_processing = types.ModuleType("backend.text_processing")
    krea2_engine = types.ModuleType("backend.text_processing.krea2_engine")

    class Qwen3VL4BEngine:
        def __call__(self, texts, images=[]):
            return []

    krea2_engine.Qwen3VL4BEngine = Qwen3VL4BEngine
    text_processing.krea2_engine = krea2_engine
    return {
        "backend.diffusion_engine": diffusion_engine,
        "backend.diffusion_engine.krea": engine,
        "backend.text_processing": text_processing,
        "backend.text_processing.krea2_engine": krea2_engine,
    }


class Host:
    """Everything the tests reach into."""

    def __init__(self):
        self.opts = Options()


def install_host() -> Host:
    host = Host()
    stubs = {"gradio": _gradio()}
    stubs.update(_modules(host))
    stubs.update(_backend(host))
    for name, module in stubs.items():
        sys.modules[name] = module
    krea = _krea_module(stubs)
    sys.modules["backend.nn.krea"] = krea
    stubs["backend.nn"].krea = krea
    stubs.update(_engine_module(stubs, krea))
    for name in ("backend.diffusion_engine", "backend.diffusion_engine.krea", "backend.text_processing", "backend.text_processing.krea2_engine"):
        sys.modules[name] = stubs[name]
    stubs["backend"].diffusion_engine = stubs["backend.diffusion_engine"]
    stubs["backend"].text_processing = stubs["backend.text_processing"]
    host.stubs = stubs
    host.krea = krea
    host.engine_module = stubs["backend.diffusion_engine.krea"]
    host.dynamic_args = stubs["backend.args"].dynamic_args
    host.memory_management = stubs["backend.memory_management"]
    host.processing = stubs["modules.processing"]
    host.prompt_parser = stubs["modules.prompt_parser"]
    host.errors = stubs["modules.errors"]
    host.attention = stubs["backend.attention"]
    return host


HOST = install_host()


@pytest.fixture
def host():
    """The stub host, reset between tests."""
    HOST.opts.krea2_do_reference = False
    HOST.dynamic_args.ref_latents.clear()
    HOST.dynamic_args.is_referencing = False
    HOST.memory_management.loaded.clear()
    HOST.errors.reports.clear()
    HOST.attention.attention_function = HOST.attention.attention_pytorch
    P = HOST.processing.StableDiffusionProcessing
    T = HOST.processing.StableDiffusionProcessingTxt2Img
    P.cached_c = [None, None, None]
    P.cached_uc = [None, None, None]
    T.cached_hr_c = [None, None, None]
    T.cached_hr_uc = [None, None, None]
    from neo_identity import lifecycle

    lifecycle.uninstall()
    lifecycle.STATE.reset()
    yield HOST
    lifecycle.uninstall()
    lifecycle.STATE.reset()


@pytest.fixture
def forge_root():
    if FORGE_ROOT is None:
        pytest.skip("no Forge Neo checkout: set NEO_IDENTITY_FORGE_ROOT or run from <forge>/extensions/<repo>")
    return FORGE_ROOT


def tiny_dit(seed: int = 0):
    """A small real ``SingleStreamDiT`` (or the fake) with deterministic weights."""
    torch.manual_seed(seed)
    model = HOST.krea.SingleStreamDiT(features=64, tdim=16, txtdim=32, heads=4, kvheads=2, multiplier=1, layers=2, patch=2, channels=16, bias=False, theta=1e3, txtlayers=3, txtheads=2, txtkvheads=2).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
    return model


def image(h: int, w: int, seed: int = 0) -> torch.Tensor:
    """A captured reference as the engine stores it: (1, H, W, 3) in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, h, w, 3, generator=g)
