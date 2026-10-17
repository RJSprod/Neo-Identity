"""Krea 2 Identity Edit V2 for Forge Neo: the always-on script.

One accordion on txt2img and img2img. Switched on, the generation's Krea 2 pass
runs the V2 recipe -- the fit geometry, per-reference placement and boost,
grounding of both prompts -- on the references the host already collects:
ImageStitch's gallery, the img2img image, or Model Chain's Stage 1 image and
supplemental references. Switched off, or uninstalled, the host is exactly
itself.
"""

from __future__ import annotations

import re

import gradio as gr

from modules import errors, scripts
from modules.ui_components import InputAccordion

from neo_identity import compat, lifecycle
from neo_identity import settings as S
from neo_identity.log import ensure_handler

logger = ensure_handler()

INFO = """
<b>Krea 2 Identity Edit V2</b> runs Krea 2's edit path the way the V2 LoRAs were trained:
references fitted to the output grid, each placed <i>aligned</i> onto the layout or <i>beside</i> it,
an optional attention <i>boost</i>, and both prompts grounded on the images.<br>
<b>References</b> come from what you already use: <b>ImageStitch Integrated</b> (scene first, person last),
the <b>img2img</b> image, or <b>Model Chain</b>'s Stage 1 image and Stage 2 references.
Add the LoRA as usual: <code>&lt;lora:krea2_identity_edit_v2:1&gt;</code>. Turbo: 8 steps, CFG 1.
Only a <b>Krea 2</b> pass is affected; every other model runs untouched.
"""

LORA_TAG = re.compile(r"<lora:", re.IGNORECASE)
LOW_DENOISE = 0.9

IMAGE_STITCH_PRIORITY = 529
"""ImageStitch's sorting priority. A larger number places this accordion below
ImageStitch's in the tab, which is where its references are chosen. Forge runs
the hooks themselves in load order, not by this number, so nothing here depends
on it: the recipe stays armed until the next generation starts (see
``lifecycle.release``)."""


class NeoIdentityEdit(scripts.Script):
    sorting_priority = 1000

    def __init__(self):
        super().__init__()
        self._contract = None

    def title(self):
        return "Krea 2 Identity Edit V2"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label=self.title(), elem_id=self.elem_id("enable")) as enabled:
            gr.HTML(INFO)
            with gr.Row():
                subject_placement = gr.Radio(
                    choices=list(S.PLACEMENTS),
                    value=S.ALIGNED,
                    label="Subject placement (last reference)",
                    info="Aligned pins the reference onto the output layout, as trained. Beside puts it next to the canvas so the person restages freely. Bag of features is experimental.",
                    elem_id=self.elem_id("subject_placement"),
                )
                scene_placement = gr.Radio(
                    choices=list(S.PLACEMENTS),
                    value=S.ALIGNED,
                    label="Scene placement (first reference, with two or more)",
                    info="For a person into a scene: scene Aligned, subject Beside.",
                    elem_id=self.elem_id("scene_placement"),
                )
            with gr.Row():
                subject_boost = gr.Slider(
                    minimum=S.BOOST_MIN,
                    maximum=S.BOOST_MAX,
                    step=S.BOOST_STEP,
                    value=S.BOOST_DEFAULT,
                    label="Subject boost",
                    info="How hard the output attends to the subject reference. 1 is neutral and costs nothing; the model card suggests 2 for identity. Above 1 the attention carries a bias, about 6 % more work per step; under SageAttention the boosted attention runs on PyTorch's kernel instead.",
                    elem_id=self.elem_id("subject_boost"),
                )
                scene_boost = gr.Slider(
                    minimum=S.BOOST_MIN,
                    maximum=S.BOOST_MAX,
                    step=S.BOOST_STEP,
                    value=S.BOOST_DEFAULT,
                    label="Scene boost",
                    info="The same dial for the first reference when there are two or more.",
                    elem_id=self.elem_id("scene_boost"),
                )
            with gr.Row():
                grounding_px = gr.Slider(
                    minimum=0,
                    maximum=S.GROUNDING_MAX,
                    step=S.GROUNDING_STEP,
                    value=S.GROUNDING_DEFAULT,
                    label="Grounding resolution",
                    info="Longest side of each reference as Qwen3-VL sees it. Lower favours the instruction, higher favours likeness; 768 is what the LoRAs trained at, 0 is the image's own size.",
                    elem_id=self.elem_id("grounding"),
                )
                geometry = gr.Radio(
                    choices=list(S.GEOMETRIES),
                    value=S.FIT,
                    label="Reference geometry",
                    info="Fit is the v1.2 and V2 training geometry. Crop is the v1 and v1.1 one.",
                    elem_id=self.elem_id("geometry"),
                )
                ground_negative = gr.Checkbox(
                    value=True,
                    label="Ground the negative prompt",
                    info="At CFG above 1 the negative prompt sees the same references, which is the trained unconditional. No effect at CFG 1.",
                    elem_id=self.elem_id("ground_negative"),
                )

        self.infotext_fields = [
            (enabled, S.INFOTEXT_ENABLED),
            (subject_placement, S.INFOTEXT_SUBJECT_PLACEMENT),
            (scene_placement, S.INFOTEXT_SCENE_PLACEMENT),
            (subject_boost, S.INFOTEXT_SUBJECT_BOOST),
            (scene_boost, S.INFOTEXT_SCENE_BOOST),
            (grounding_px, S.INFOTEXT_GROUNDING),
            (geometry, S.INFOTEXT_GEOMETRY),
            (ground_negative, S.INFOTEXT_NEGATIVE),
        ]
        self.paste_field_names = [name for _, name in self.infotext_fields]
        return [enabled, subject_placement, scene_placement, subject_boost, scene_boost, grounding_px, geometry, ground_negative]

    # -- hooks ------------------------------------------------------------- #

    def before_process(self, p, enabled=False, subject_placement=S.ALIGNED, scene_placement=S.ALIGNED, subject_boost=S.BOOST_DEFAULT, scene_boost=S.BOOST_DEFAULT, grounding_px=S.GROUNDING_DEFAULT, geometry=S.FIT, ground_negative=True, *args):
        """Switch the recipe on for this generation, before the host applies
        its override settings and loads the model."""
        if not enabled:
            lifecycle.deactivate()
            return

        contract = self._install()
        if not contract.ok:
            lifecycle.deactivate()
            message = f"Krea 2 Identity Edit V2 cannot run on this host ({contract.summary}); the generation runs natively"
            logger.warning(message)
            p.comment(f"Neo Identity: {message}")
            return

        settings = S.Settings.normalized(
            subject_placement=subject_placement,
            scene_placement=scene_placement,
            subject_boost=subject_boost,
            scene_boost=scene_boost,
            grounding_px=grounding_px,
            geometry=geometry,
            ground_negative=ground_negative,
        )
        lifecycle.activate(settings)

        # The engine only captures references while the host's own Krea 2
        # toggle is on. Asked for through the generation's override settings,
        # the way Model Chain asks for its Stage 2, so the host applies it now
        # and restores the user's value when the generation ends. An override
        # already present -- the user's own, or Model Chain's -- wins.
        p.override_settings.setdefault("krea2_do_reference", True)
        p.extra_generation_params.update(settings.infotext())
        self._warn(p, settings)

    def process(self, p, enabled=False, *args):
        """The model is loaded: say what will happen, and forget cached
        conditioning that could describe another generation's references."""
        if not enabled or not lifecycle.STATE.active:
            return
        p.clear_prompt_cache()
        model = getattr(p, "sd_model", None)
        if not compat.is_krea2(model):
            name = type(model).__name__ if model is not None else "none"
            logger.info("the loaded model is %s, not Krea 2; Identity Edit waits for a Krea 2 pass (Model Chain's Stage 2, for one)", name)
            return
        if compat.is_compiled(_transformer_of(model)):
            lifecycle.STATE.note("the transformer is compiled (torch.compile), which Identity Edit does not drive; turn compile off for Identity Edit")
        logger.info("on for this generation: %s", lifecycle.STATE.settings.describe())

    def process_batch(self, p, enabled=False, *args, **kwargs):
        if enabled and lifecycle.STATE.active:
            # The img2img image is captured in init(), after process(): the
            # conditioning computed next must not come from the cache.
            p.clear_prompt_cache()

    def process_before_every_sampling(self, p, enabled=False, *args, x=None, **kwargs):
        """Encode the references for the size about to be sampled, before the
        sampler loads the transformer, so no VAE encode lands mid-step."""
        if not (enabled and lifecycle.STATE.active and x is not None):
            return
        model = getattr(p, "sd_model", None)
        if not compat.is_krea2(model) or lifecycle.STATE.references is None:
            return
        try:
            latent = x.squeeze(2) if x.ndim == 5 else x
            lifecycle.preencode(model, int(latent.shape[-2]) * 8, int(latent.shape[-1]) * 8)
        except Exception:
            errors.report("Neo Identity: could not encode the references before sampling", exc_info=True)

    def postprocess(self, p, processed, enabled=False, *args):
        """Drop the generation's tensors and put its notes on the result. The
        recipe stays armed: Model Chain's Stage 2 runs inside this same phase,
        possibly after this hook, and the next generation's ``before_process``
        resets the state before anything else samples."""
        if not enabled:
            return
        used = lifecycle.STATE.used
        notes = lifecycle.release()
        if used:
            p.extra_generation_params[S.INFOTEXT_REFERENCES] = str(used)
        for note in notes:
            processed.comments += f"\nNeo Identity: {note}"

    # -- helpers ----------------------------------------------------------- #

    def _install(self):
        if self._contract is None or not self._contract.ok:
            try:
                self._contract = lifecycle.install()
            except Exception as error:
                errors.report("Neo Identity: installing on the host failed", exc_info=True)
                self._contract = compat.Contract(False, (f"{type(error).__name__}: {error}",))
        return self._contract

    @staticmethod
    def _warn(p, settings) -> None:
        prompt = p.prompt if isinstance(p.prompt, str) else " ".join(str(x) for x in (p.prompt or []))
        if not LORA_TAG.search(prompt or ""):
            logger.warning("no <lora:...> tag in the prompt: Krea 2's edit behaviour comes from the Identity Edit LoRA, so add <lora:krea2_identity_edit_v2:1> (Model Chain's Stage 2 prompt carries its own)")
        denoise = getattr(p, "denoising_strength", None)
        if getattr(p, "init_images", None) and denoise is not None and denoise < LOW_DENOISE:
            logger.warning("img2img denoise is %.2f: in edit mode the image is a reference, not a starting point, so 1.0 is the expected value", denoise)
        if settings.boosted and compat.attention_name() == "attention_sage":
            logger.info("boost under SageAttention: the boosted attention runs on PyTorch's kernel for this generation")


def _transformer_of(model):
    """The loaded transformer as the patcher holds it, compile wrapper included."""
    try:
        return model.forge_objects.unet.model.diffusion_model
    except AttributeError:
        return None
