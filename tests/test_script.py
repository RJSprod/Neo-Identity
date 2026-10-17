"""The always-on script: its controls, its hooks, and what it writes down."""

import logging

import pytest
import torch

import conftest
import neo_identity_edit as script_module
from neo_identity import compat
from neo_identity import lifecycle as L
from neo_identity import settings as S

Script = script_module.NeoIdentityEdit

ARGS = dict(subject_placement=S.ALIGNED, scene_placement=S.ALIGNED, subject_boost=1.0, scene_boost=1.0, grounding_px=768, geometry=S.FIT, ground_negative=True)


def arguments(enabled=True, **overrides):
    values = dict(ARGS, **overrides)
    return [enabled, values["subject_placement"], values["scene_placement"], values["subject_boost"], values["scene_boost"], values["grounding_px"], values["geometry"], values["ground_negative"]]


@pytest.fixture
def script(host):
    return Script()


@pytest.fixture
def p(host):
    engine = host.engine_module.Krea2(conftest.tiny_dit())
    return host.processing.StableDiffusionProcessingTxt2Img(prompt="a portrait <lora:krea2_identity_edit_v2:1>", sd_model=engine)


class TestUI:
    def test_the_accordion_returns_one_control_per_setting(self, script):
        controls = script.ui(False)
        assert len(controls) == 8
        assert controls[0].value is False  # off until asked for
        assert controls[1].choices == list(S.PLACEMENTS) and controls[1].value == S.ALIGNED
        assert controls[3].value == S.BOOST_DEFAULT and controls[5].value == S.GROUNDING_DEFAULT
        assert controls[6].choices == list(S.GEOMETRIES) and controls[7].value is True

    def test_every_control_pastes_from_its_infotext_key(self, script):
        controls = script.ui(False)
        keys = [key for _, key in script.infotext_fields]
        assert keys == [S.INFOTEXT_ENABLED, S.INFOTEXT_SUBJECT_PLACEMENT, S.INFOTEXT_SCENE_PLACEMENT, S.INFOTEXT_SUBJECT_BOOST, S.INFOTEXT_SCENE_BOOST, S.INFOTEXT_GROUNDING, S.INFOTEXT_GEOMETRY, S.INFOTEXT_NEGATIVE]
        assert [c for c, _ in script.infotext_fields] == controls
        assert script.paste_field_names == keys

    def test_it_shows_on_both_tabs_below_image_stitch(self, script, host):
        assert script.show(False) is host.stubs["modules.scripts"].AlwaysVisible
        assert script.show(True) is host.stubs["modules.scripts"].AlwaysVisible
        assert script.sorting_priority > script_module.IMAGE_STITCH_PRIORITY  # UI placement only

    def test_the_title_names_the_recipe(self, script):
        assert script.title() == "Krea 2 Identity Edit V2"


class TestBeforeProcess:
    def test_off_leaves_the_host_alone(self, script, p):
        L.activate(S.Settings())  # a previous job that never ended
        script.before_process(p, *arguments(enabled=False))
        assert not L.STATE.active
        assert p.override_settings == {} and p.extra_generation_params == {}

    def test_on_switches_the_recipe_on_with_the_hosts_toggle(self, script, p):
        script.before_process(p, *arguments(subject_placement=S.BESIDE, subject_boost=2.0, grounding_px=512))
        assert L.installed() and L.STATE.active
        assert L.STATE.settings == S.Settings(subject_placement=S.BESIDE, subject_boost=2.0, grounding_px=512)
        assert p.override_settings == {"krea2_do_reference": True}
        assert p.extra_generation_params[S.INFOTEXT_ENABLED] == "v2"
        assert p.extra_generation_params[S.INFOTEXT_SUBJECT_PLACEMENT] == "Beside"
        assert p.extra_generation_params[S.INFOTEXT_SUBJECT_BOOST] == "2"

    def test_an_override_already_there_wins(self, script, p):
        p.override_settings["krea2_do_reference"] = False
        script.before_process(p, *arguments())
        assert p.override_settings["krea2_do_reference"] is False

    def test_a_host_that_fails_the_contract_gets_a_comment_and_nothing_else(self, script, p, monkeypatch):
        monkeypatch.setattr(L, "install", lambda: compat.Contract(False, ("Forge is not Neo",)))
        script.before_process(p, *arguments())
        assert not L.STATE.active
        assert any("Forge is not Neo" in text for text in p.comments)
        assert p.override_settings == {}

    def test_a_missing_lora_tag_is_warned_about(self, script, p, caplog):
        p.prompt = "a portrait"
        with caplog.at_level(logging.WARNING, logger="neo_identity"):
            script.before_process(p, *arguments())
        assert any("<lora:" in record.getMessage() for record in caplog.records)

    def test_a_low_img2img_denoise_is_warned_about(self, script, host, caplog):
        engine = host.engine_module.Krea2(conftest.tiny_dit())
        p = host.processing.StableDiffusionProcessingImg2Img(prompt="<lora:x:1>", sd_model=engine, init_images=[object()], denoising_strength=0.5)
        with caplog.at_level(logging.WARNING, logger="neo_identity"):
            script.before_process(p, *arguments())
        assert any("denoise" in record.getMessage() for record in caplog.records)


class TestProcess:
    def test_process_forgets_the_prompt_cache_and_says_what_runs(self, script, p, caplog):
        script.before_process(p, *arguments())
        with caplog.at_level(logging.INFO, logger="neo_identity"):
            script.process(p, *arguments())
        assert p.cleared == 1
        assert any("on for this generation" in r.getMessage() for r in caplog.records)

    def test_another_model_is_left_for_a_later_krea_pass(self, script, p, caplog):
        p.sd_model = object()
        script.before_process(p, *arguments())
        with caplog.at_level(logging.INFO, logger="neo_identity"):
            script.process(p, *arguments())
        assert L.STATE.active  # still on: Model Chain's Stage 2 may be Krea 2
        assert any("not Krea 2" in r.getMessage() for r in caplog.records)

    def test_a_compiled_transformer_is_noted(self, script, p):
        import types

        dit = p.sd_model.forge_objects.unet.model.diffusion_model
        p.sd_model.forge_objects.unet.model.diffusion_model = types.SimpleNamespace(_orig_mod=dit)
        script.before_process(p, *arguments())
        script.process(p, *arguments())
        assert any("compiled" in note for note in L.STATE.notes)

    def test_process_batch_forgets_the_prompt_cache_again(self, script, p):
        script.before_process(p, *arguments())
        script.process_batch(p, *arguments(), batch_number=0)
        assert p.cleared == 1
        script.process_batch(p, *arguments(enabled=False), batch_number=0)
        assert p.cleared == 1


class TestSampling:
    def test_the_references_are_encoded_for_the_sampled_size(self, script, p, host):
        script.before_process(p, *arguments())
        engine = p.sd_model
        host.opts.krea2_do_reference = True
        host.dynamic_args.is_referencing = True
        engine.encode_first_stage(torch.rand(1, 3, 768, 1024) * 2 - 1)
        host.dynamic_args.is_referencing = False
        engine.get_learned_conditioning(host.prompt_parser.SdConditioning(["a"], width=None, height=None))
        script.process_before_every_sampling(p, *arguments(), x=torch.zeros(1, 16, 1, 64, 64))
        assert engine.forge_objects.vae.encodes == [(1, 384, 512, 3)]  # fitted to 512x512
        script.process_before_every_sampling(p, *arguments(), x=torch.zeros(1, 16, 1, 128, 128))
        assert engine.forge_objects.vae.encodes[-1] == (1, 768, 1024, 3)  # the hires pass, fitted to 1024

    def test_nothing_is_encoded_without_references_or_off(self, script, p, host):
        script.process_before_every_sampling(p, *arguments(enabled=False), x=torch.zeros(1, 16, 1, 64, 64))
        script.before_process(p, *arguments())
        script.process_before_every_sampling(p, *arguments(), x=torch.zeros(1, 16, 1, 64, 64))
        assert p.sd_model.forge_objects.vae.encodes == []

    def test_an_encode_failure_is_reported_not_raised(self, script, p, host):
        script.before_process(p, *arguments())
        engine = p.sd_model
        host.opts.krea2_do_reference = True
        host.dynamic_args.is_referencing = True
        engine.encode_first_stage(torch.rand(1, 3, 512, 512) * 2 - 1)
        host.dynamic_args.is_referencing = False
        engine.get_learned_conditioning(host.prompt_parser.SdConditioning(["a"], width=None, height=None))
        engine.forge_objects.vae.encode = lambda pixels: (_ for _ in ()).throw(RuntimeError("no"))
        script.process_before_every_sampling(p, *arguments(), x=torch.zeros(1, 16, 1, 64, 64))
        assert host.errors.reports


class TestPostprocess:
    def test_postprocess_releases_the_tensors_and_writes_the_notes_and_the_count(self, script, p, host):
        script.before_process(p, *arguments(subject_boost=2.0))
        L.STATE.used = 2
        L.STATE.note("something to say")
        processed = host.processing.Processed()
        script.postprocess(p, processed, *arguments())
        assert L.STATE.active and L.STATE.settings.subject_boost == 2.0  # armed for a Stage 2 in this generation
        assert L.STATE.references is None and L.STATE.engine is None and L.STATE.notes == []
        assert p.extra_generation_params[S.INFOTEXT_REFERENCES] == "2"
        assert "Neo Identity: something to say" in processed.comments
        script.before_process(p, *arguments(enabled=False))  # the next generation, off
        assert not L.STATE.active

    def test_a_krea_pass_after_postprocess_still_runs_the_recipe(self, script, p, host):
        """Model Chain's Stage 2 runs inside the generation's postprocess phase,
        and Forge orders the hooks by load order: this script's postprocess may
        run first. The Stage 2 pass must still get the recipe."""
        script.before_process(p, *arguments())
        script.postprocess(p, host.processing.Processed(), *arguments())
        engine = p.sd_model
        host.opts.krea2_do_reference = True  # Model Chain's Enable override
        engine.encode_first_stage(torch.rand(1, 3, 512, 512) * 2 - 1)  # the Stage 1 image, through init()
        engine.get_learned_conditioning(host.prompt_parser.SdConditioning(["refine"], width=512, height=512))
        assert engine.native_conditioning == []
        assert engine.text_processing_engine_qwen.calls[-1][1] == [(1, 512, 512, 3)]
        assert L.STATE.references is not None and L.STATE.used == 1

    def test_postprocess_off_is_a_no_op(self, script, p, host):
        L.activate(S.Settings())
        processed = host.processing.Processed()
        script.postprocess(p, processed, *arguments(enabled=False))
        assert L.STATE.active  # not this script's job to end what it did not start
        assert processed.comments == ""


class TestAWholeGeneration:
    def test_txt2img_with_image_stitch_runs_the_recipe_end_to_end(self, script, p, host, forge_root):
        """ImageStitch captures, the host conditions, the sampler calls the
        transformer: the recipe runs, and the host's own state is left clean."""
        host.opts.krea2_do_reference = True  # the override the host would have applied
        script.before_process(p, *arguments(subject_placement=S.BESIDE, subject_boost=2.0))
        script.process(p, *arguments())
        engine = p.sd_model
        host.dynamic_args.is_referencing = True
        engine.encode_first_stage(torch.rand(1, 3, 256, 256) * 2 - 1)
        engine.encode_first_stage(torch.rand(1, 3, 128, 256) * 2 - 1)
        host.dynamic_args.is_referencing = False
        script.process_batch(p, *arguments(), batch_number=0)
        uc = engine.get_learned_conditioning(host.prompt_parser.SdConditioning([""], is_negative_prompt=True, width=256, height=256))
        c = engine.get_learned_conditioning(host.prompt_parser.SdConditioning([p.prompt], width=256, height=256))
        assert len(uc) == len(c) == 1
        assert engine.forge_objects.vae.encodes == [(1, 256, 256, 3), (1, 128, 256, 3)]
        script.process_before_every_sampling(p, *arguments(), x=torch.zeros(1, 16, 1, 32, 32))
        assert len(engine.forge_objects.vae.encodes) == 2
        dit = engine.forge_objects.unet.model.diffusion_model
        x, t, ctx = torch.randn(1, 16, 1, 32, 32), torch.tensor([0.5]), torch.randn(1, 5, 3, 32)
        with torch.inference_mode():
            out = dit(x, t, ctx.clone())
            native = L.original("forward")(dit, x, t, ctx.clone())
        assert out.shape == x.shape and not torch.allclose(out, native, atol=1e-5)
        processed = host.processing.Processed()
        script.postprocess(p, processed, *arguments())
        assert p.extra_generation_params[S.INFOTEXT_REFERENCES] == "2"
        assert L.STATE.references is None and host.dynamic_args.ref_latents == []
        with torch.inference_mode():
            assert torch.equal(dit(x, t, ctx.clone()), native)  # nothing to edit with: native
        script.before_process(p, *arguments(enabled=False))
        assert not L.STATE.active
