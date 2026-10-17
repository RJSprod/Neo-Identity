"""The host contract: satisfied by the pinned shapes, refused by anything else."""

import types

from neo_identity import compat


class TestCheck:
    def test_the_stub_host_satisfies_the_contract(self, host):
        contract = compat.check(force=True)
        assert contract.ok, contract.reasons
        assert contract.summary == "host contract satisfied"

    def test_an_added_trailing_argument_is_tolerated(self, host, monkeypatch):
        krea2 = host.engine_module.Krea2

        def get_learned_conditioning(self, prompt, extra=None):
            pass

        monkeypatch.setattr(krea2, "get_learned_conditioning", get_learned_conditioning)
        assert compat.check(force=True).ok
        monkeypatch.undo()
        compat.check(force=True)

    def test_a_changed_signature_is_named(self, host, monkeypatch):
        krea2 = host.engine_module.Krea2

        def get_learned_conditioning(self, texts):  # the argument renamed: not the pinned shape
            pass

        monkeypatch.setattr(krea2, "get_learned_conditioning", get_learned_conditioning)
        contract = compat.check(force=True)
        assert not contract.ok
        assert any("get_learned_conditioning" in reason for reason in contract.reasons)
        compat.check(force=True)  # restored by monkeypatch at teardown; recomputed here for the cache

    def test_a_missing_engine_is_a_refusal_not_an_exception(self, host, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "backend.diffusion_engine.krea", None)
        contract = compat.check(force=True)
        assert not contract.ok and "not importable" in contract.summary
        monkeypatch.undo()
        assert compat.check(force=True).ok

    def test_a_renamed_submodule_is_named(self, host, monkeypatch):
        dit = host.krea.SingleStreamDiT
        monkeypatch.setattr(compat, "DIT_ATTRIBUTES", compat.DIT_ATTRIBUTES + ("nonesuch",))
        contract = compat.check(force=True)
        assert not contract.ok and any("nonesuch" in r for r in contract.reasons)
        monkeypatch.undo()
        assert compat.check(force=True).ok
        assert dit is host.krea.SingleStreamDiT


class TestHelpers:
    def test_is_krea2_by_class_name_and_instance(self, host):
        assert compat.is_krea2(host.engine_module.Krea2())
        assert not compat.is_krea2(object())
        assert not compat.is_krea2(None)

        class Krea2:  # another module's class of the same name still counts
            pass

        assert compat.is_krea2(Krea2())

    def test_attention_names_that_honour_a_scale(self, host):
        assert compat.attention_name() == "attention_pytorch"
        assert compat.attention_honours_scale()
        host.attention.attention_function = types.SimpleNamespace(__name__="attention_flash")
        assert not compat.attention_honours_scale()
        host.attention.attention_function = types.SimpleNamespace(__name__="attention_sage")
        assert compat.attention_honours_scale()

    def test_unwrap_sees_through_a_compile_wrapper(self):
        inner = object()
        wrapper = types.SimpleNamespace(_orig_mod=inner)
        assert compat.unwrap(wrapper) is inner
        assert compat.is_compiled(wrapper) and not compat.is_compiled(inner)
        engine = types.SimpleNamespace(forge_objects=types.SimpleNamespace(unet=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=wrapper))))
        assert compat.diffusion_model_of(engine) is inner
        assert compat.diffusion_model_of(object()) is None
