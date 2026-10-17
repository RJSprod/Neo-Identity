"""The three dispatchers: inert off, the recipe on, and the state of one job."""

import types

import pytest
import torch

import conftest
from neo_identity import geometry as G
from neo_identity import lifecycle as L
from neo_identity import settings as S


def engine_with(host, dit=None):
    return host.engine_module.Krea2(dit if dit is not None else conftest.tiny_dit())


def prompt(host, texts=("a prompt",), negative=False, width=1024, height=1024):
    return host.prompt_parser.SdConditioning(list(texts), is_negative_prompt=negative, width=width, height=height)


def captured(host, engine, h, w, *, gallery: bool, seed=0):
    """Run a reference through the engine the way ImageStitch (gallery) or
    img2img's init (not gallery) does, with the host's toggle on."""
    host.opts.krea2_do_reference = True
    host.dynamic_args.is_referencing = gallery
    pixels = conftest.image(h, w, seed).movedim(-1, 1) * 2 - 1  # the encoder takes [-1, 1] NCHW
    engine.encode_first_stage(pixels)
    host.dynamic_args.is_referencing = False


class TestInstall:
    def test_install_wraps_once_and_uninstall_restores(self, host):
        krea2, dit = host.engine_module.Krea2, host.krea.SingleStreamDiT
        originals = (krea2.get_learned_conditioning, krea2.encode_first_stage, dit.forward)
        assert L.install().ok and L.installed()
        assert krea2.get_learned_conditioning is L._conditioning
        assert krea2.encode_first_stage is L._encode_first_stage
        assert dit.forward is L._forward
        L.install()  # a second time changes nothing
        assert L.original("forward") is originals[2]
        L.uninstall()
        assert (krea2.get_learned_conditioning, krea2.encode_first_stage, dit.forward) == originals
        assert not L.installed()

    def test_a_reloaded_module_replaces_the_dispatcher_and_keeps_the_original(self, host):
        krea2 = host.engine_module.Krea2
        original = krea2.get_learned_conditioning
        L.install()

        def newer(self, prompt):
            return "newer"

        L._originals.clear()
        L._wrap(krea2, "get_learned_conditioning", newer)
        assert krea2.get_learned_conditioning is newer
        assert L.original("get_learned_conditioning") is original
        assert getattr(newer, L.ORIGINAL) is original
        L._wrap(krea2, "get_learned_conditioning", L._conditioning)  # so uninstall finds its tag

    def test_a_failed_contract_installs_nothing(self, host, monkeypatch):
        from neo_identity import compat

        monkeypatch.setattr(compat, "check", lambda force=False: compat.Contract(False, ("no",)))
        assert not L.install().ok
        assert not L.installed()


class TestOff:
    def test_every_dispatcher_passes_through_when_no_job_is_on(self, host):
        L.install()
        engine = engine_with(host)
        host.opts.krea2_do_reference = True
        host.dynamic_args.is_referencing = True
        engine.encode_first_stage(torch.zeros(1, 3, 64, 64))
        assert len(engine.ref_latents) == 1
        conds = engine.get_learned_conditioning(prompt(host))
        assert engine.native_conditioning == [["a prompt"]]
        assert engine.text_processing_engine_qwen.calls[-1][1] == [(1, 64, 64, 3)]  # the native grounding
        assert len(conds) == 1

    def test_the_forward_passes_through_when_no_job_is_on(self, host):
        L.install()
        dit = conftest.tiny_dit()
        x, t, ctx = torch.randn(1, 16, 1, 8, 8), torch.tensor([0.5]), torch.randn(1, 5, 3, 32)
        with torch.inference_mode():
            out = dit(x, t, ctx.clone())
            expected = L.original("forward")(dit, x, t, ctx.clone())
        assert torch.equal(out, expected)

    def test_a_job_on_another_models_engine_leaves_it_alone(self, host):
        L.install()
        L.activate(S.Settings())
        other = types.SimpleNamespace()  # not a Krea2
        assert L._engaged(other) is False


class TestCapture:
    def test_a_new_reference_empties_the_shared_prompt_cache_in_place(self, host):
        L.install()
        L.activate(S.Settings())
        engine = engine_with(host)
        p = host.processing.StableDiffusionProcessing()  # shares the class-level lists
        p.cached_c[0] = ("stale", "params")
        host.processing.StableDiffusionProcessingTxt2Img.cached_hr_c[0] = "stale"
        captured(host, engine, 64, 64, gallery=True)
        assert p.cached_c == [None, None, None]
        assert host.processing.StableDiffusionProcessingTxt2Img.cached_hr_c == [None, None, None]
        assert len(engine.ref_latents) == 1

    def test_the_capture_itself_is_the_hosts(self, host):
        L.install()
        L.activate(S.Settings())
        engine = engine_with(host)
        host.opts.krea2_do_reference = False  # the host's toggle decides, as natively
        host.dynamic_args.is_referencing = True
        engine.encode_first_stage(torch.zeros(1, 3, 64, 64))
        assert engine.ref_latents == [] and engine.encoded == [(1, 3, 64, 64)]

    def test_a_capture_forgets_the_jobs_snapshot(self, host):
        L.install()
        L.activate(S.Settings())
        engine = engine_with(host)
        captured(host, engine, 64, 64, gallery=True)
        engine.get_learned_conditioning(prompt(host))
        assert L.STATE.references is not None
        captured(host, engine, 96, 64, gallery=True, seed=3)
        assert L.STATE.references is None


class TestConditioning:
    def test_both_prompts_are_grounded_on_the_references_in_native_order(self, host):
        L.install()
        L.activate(S.Settings(grounding_px=768))
        engine = engine_with(host)
        captured(host, engine, 1536, 1024, gallery=True)   # the gallery: a scene
        captured(host, engine, 800, 600, gallery=True, seed=1)
        captured(host, engine, 512, 512, gallery=False, seed=2)  # the img2img image
        qwen = engine.text_processing_engine_qwen
        host.dynamic_args.ref_latents.append(torch.zeros(1, 16, 8, 8))  # left by an earlier native pass
        negative = engine.get_learned_conditioning(prompt(host, ("",), negative=True))
        positive = engine.get_learned_conditioning(prompt(host))
        assert engine.native_conditioning == []  # the native path never ran
        assert len(negative) == 1 and len(positive) == 1
        # img2img image first, then the gallery in order; the long side capped at 768
        expected = [(1, 512, 512, 3), (1, 768, 512, 3), (1, 768, 576, 3)]
        assert qwen.calls[0] == ([""], expected, True)
        assert qwen.calls[1] == (["a prompt"], expected, False)
        assert engine.ini_latent is None  # consumed once, as natively
        assert L.STATE.used == 3
        assert host.dynamic_args.ref_latents == []
        assert host.memory_management.loaded == [engine.forge_objects.clip.patcher] * 2

    def test_the_negative_can_stay_text_only(self, host):
        L.install()
        L.activate(S.Settings(ground_negative=False))
        engine = engine_with(host)
        captured(host, engine, 512, 512, gallery=True)
        engine.get_learned_conditioning(prompt(host, ("",), negative=True))
        engine.get_learned_conditioning(prompt(host))
        calls = engine.text_processing_engine_qwen.calls
        assert calls[0][1] == [] and calls[1][1] == [(1, 512, 512, 3)]

    def test_the_references_are_encoded_for_the_announced_size_before_sampling(self, host):
        L.install()
        L.activate(S.Settings())
        engine = engine_with(host)
        captured(host, engine, 768, 1024, gallery=True)
        engine.get_learned_conditioning(prompt(host, width=1024, height=1024))
        vae = engine.forge_objects.vae
        assert vae.encodes == [(1, 768, 1024, 3)]  # fitted: 48 rows centred in 64, no resize needed
        engine.get_learned_conditioning(prompt(host, ("",), negative=True, width=1024, height=1024))
        assert len(vae.encodes) == 1  # the same size is not encoded twice
        assert L.STATE.references.has_encoded(1024, 1024, S.FIT)

    def test_the_second_batch_keeps_the_jobs_set(self, host):
        L.install()
        L.activate(S.Settings())
        engine = engine_with(host)
        captured(host, engine, 512, 512, gallery=True)
        captured(host, engine, 640, 640, gallery=False)
        engine.get_learned_conditioning(prompt(host))
        first = L.STATE.references
        engine.get_learned_conditioning(prompt(host))  # batch two: ini_latent is gone, the gallery is the same
        assert L.STATE.references is first
        assert engine.text_processing_engine_qwen.calls[1][1] == [(1, 640, 640, 3), (1, 512, 512, 3)]

    def test_without_a_reference_the_native_path_runs_and_the_job_says_so(self, host):
        L.install()
        L.activate(S.Settings())
        engine = engine_with(host)
        host.opts.krea2_do_reference = True
        engine.get_learned_conditioning(prompt(host))
        assert engine.native_conditioning == [["a prompt"]]
        assert any("no reference image reached" in note for note in L.STATE.notes)
        assert L.STATE.used == 0

    def test_a_compiled_transformer_runs_natively_with_a_note(self, host):
        L.install()
        L.activate(S.Settings())
        dit = conftest.tiny_dit()
        engine = engine_with(host, dit)
        engine.forge_objects.unet.model.diffusion_model = types.SimpleNamespace(_orig_mod=dit)
        captured(host, engine, 512, 512, gallery=True)
        engine.get_learned_conditioning(prompt(host))
        assert engine.native_conditioning == [["a prompt"]]
        assert any("compiled" in note for note in L.STATE.notes)
        assert L.STATE.references is None

    def test_a_failed_pre_encode_is_not_fatal(self, host):
        L.install()
        L.activate(S.Settings())
        engine = engine_with(host)
        captured(host, engine, 512, 512, gallery=True)
        engine.forge_objects.vae.encode = lambda pixels: (_ for _ in ()).throw(RuntimeError("out of memory"))
        conds = engine.get_learned_conditioning(prompt(host))
        assert len(conds) == 1 and L.STATE.references is not None


class TestForwardDispatch:
    def test_the_owned_transformer_runs_the_recipe(self, host, forge_root):
        L.install()
        L.activate(S.Settings(subject_placement=S.BESIDE))
        dit = conftest.tiny_dit()
        engine = engine_with(host, dit)
        captured(host, engine, 128, 128, gallery=True)
        engine.get_learned_conditioning(prompt(host, width=128, height=128))
        x, t, ctx = torch.randn(1, 16, 1, 16, 16), torch.tensor([0.5]), torch.randn(1, 5, 3, 32)
        with torch.inference_mode():
            out = dit(x, t, ctx.clone())
            native = L.original("forward")(dit, x, t, ctx.clone())
            latents, _ = L.STATE.references.encoded(engine, 128, 128, S.FIT)  # a 16x16 latent is 128 px
            from neo_identity import forward as F

            expected = F.edit_forward(dit, x, t, ctx.clone(), latents, [S.BESIDE], [1.0], {})
        assert torch.allclose(out, expected, atol=1e-5)
        assert not torch.allclose(out, native, atol=1e-5)

    def test_a_size_never_announced_is_encoded_at_the_first_step(self, host, forge_root):
        L.install()
        L.activate(S.Settings())
        dit = conftest.tiny_dit()
        engine = engine_with(host, dit)
        captured(host, engine, 128, 128, gallery=True)
        engine.get_learned_conditioning(prompt(host, width=None, height=None))
        assert engine.forge_objects.vae.encodes == []
        with torch.inference_mode():
            dit(torch.randn(1, 16, 1, 16, 16), torch.tensor([0.5]), torch.randn(1, 5, 3, 32))
        assert engine.forge_objects.vae.encodes == [(1, 128, 128, 3)]

    def test_another_transformer_runs_natively(self, host):
        L.install()
        L.activate(S.Settings())
        dit = conftest.tiny_dit()
        engine = engine_with(host, dit)
        captured(host, engine, 128, 128, gallery=True)
        engine.get_learned_conditioning(prompt(host, width=128, height=128))
        other = conftest.tiny_dit(seed=5)
        x, t, ctx = torch.randn(1, 16, 1, 16, 16), torch.tensor([0.5]), torch.randn(1, 5, 3, 32)
        with torch.inference_mode():
            out = other(x, t, ctx.clone())
            native = L.original("forward")(other, x, t, ctx.clone())
        assert torch.equal(out, native)

    def test_without_a_snapshot_the_forward_is_native(self, host):
        L.install()
        L.activate(S.Settings())
        dit = conftest.tiny_dit()
        x, t, ctx = torch.randn(1, 16, 1, 16, 16), torch.tensor([0.5]), torch.randn(1, 5, 3, 32)
        with torch.inference_mode():
            assert torch.equal(dit(x, t, ctx.clone()), L.original("forward")(dit, x, t, ctx.clone()))


class TestJob:
    def test_activate_resets_and_deactivate_hands_back_the_notes(self, host):
        L.activate(S.Settings())
        L.STATE.note("one")
        L.STATE.note("one")
        L.STATE.used = 2
        assert L.STATE.notes == ["one"]
        notes = L.deactivate()
        assert notes == ["one"]
        assert not L.STATE.active and L.STATE.settings is None and L.STATE.used == 0
        L.activate(S.Settings(subject_boost=2.0))
        assert L.STATE.active and L.STATE.notes == [] and L.STATE.settings.subject_boost == 2.0

    def test_target_size_rounds_an_output_up_to_the_token_grid(self):
        assert L.target_size(1024, 1024) == (1024, 1024)
        assert L.target_size(1032, 1024) == (1040, 1024)
        assert L.target_size(8, 8) == (16, 16)

    def test_preencode_runs_once_per_size(self, host):
        L.install()
        L.activate(S.Settings())
        engine = engine_with(host)
        captured(host, engine, 2000, 1500, gallery=True)
        engine.get_learned_conditioning(prompt(host, width=None, height=None))
        assert L.preencode(engine, 1024, 1024)
        assert engine.forge_objects.vae.encodes == [(1, 1024, 768, 3)]
        assert L.preencode(engine, 1024, 1024)
        assert len(engine.forge_objects.vae.encodes) == 1
        assert L.preencode(engine, 512, 512)
        assert engine.forge_objects.vae.encodes[-1] == (1, 512, 384, 3)

    def test_preencode_without_a_job_does_nothing(self, host):
        engine = engine_with(host)
        assert not L.preencode(engine, 1024, 1024)
        assert engine.forge_objects.vae.encodes == []

    def test_the_prompt_cache_invalidation_survives_a_host_without_processing(self, host, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "modules.processing", None)
        L.invalidate_prompt_cache()  # must not raise
