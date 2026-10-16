"""The edit forward against the host's own transformer.

These run the real ``backend/nn/krea.py`` from a Forge Neo checkout at a tiny
size. The one that matters most holds the recipe to its promise: with the
references at the target's size, aligned and unboosted, the edit forward and the
native forward are the same computation.
"""

import pytest
import torch

import conftest
from neo_identity import forward as F
from neo_identity import settings as S


@pytest.fixture
def real_dit(host, forge_root):
    dit = conftest.tiny_dit().double()
    assert type(dit).__module__ == "backend.nn.krea"
    return dit


def inputs(dtype=torch.float64, batch=1, h=16, w=16, text=7):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 16, 1, h, w, generator=g).to(dtype)
    t = torch.full((batch,), 0.5, dtype=dtype)
    context = torch.randn(batch, text, 3, 32, generator=g).to(dtype)
    return x, t, context


def latents(*sizes, dtype=torch.float64):
    g = torch.Generator().manual_seed(2)
    return [torch.randn(1, 16, h, w, generator=g).to(dtype) for h, w in sizes]


class TestParity:
    def test_same_sized_aligned_unboosted_references_are_the_native_forward(self, host, real_dit):
        x, t, context = inputs()
        refs = latents((16, 16), (16, 16))
        with torch.inference_mode():
            host.dynamic_args.ref_latents = list(refs)
            native = real_dit(x, t, context.clone())
            host.dynamic_args.ref_latents = []
            ours = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED, S.ALIGNED], [1.0, 1.0], {})
        assert ours.shape == native.shape == x.shape
        assert torch.allclose(ours, native, atol=1e-9, rtol=0)

    def test_without_references_the_native_forward_is_text_to_image(self, host, real_dit):
        # A control: the native forward without references differs from the
        # edit forward with them, so the parity above is not a tautology.
        x, t, context = inputs()
        refs = latents((16, 16))
        with torch.inference_mode():
            host.dynamic_args.ref_latents = []
            plain = real_dit(x, t, context.clone())
            ours = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED], [1.0], {})
        assert not torch.allclose(ours, plain, atol=1e-4)

    def test_a_neutral_boost_changes_nothing(self, host, real_dit):
        x, t, context = inputs()
        refs = latents((16, 16), (8, 12))
        with torch.inference_mode():
            plain = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED, S.BESIDE], [1.0, 1.0], {})
            neutral = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED, S.BESIDE], [1.0, 1.0], {}, honours_scale=False)
        assert torch.equal(plain, neutral)


class TestRecipe:
    def test_references_of_other_sizes_keep_their_grid_and_the_output_keeps_its_shape(self, host, real_dit):
        x, t, context = inputs()
        refs = latents((12, 16), (16, 8))
        with torch.inference_mode():
            out = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED, S.ALIGNED], [1.0, 1.0], {})
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_placement_changes_the_result(self, host, real_dit):
        x, t, context = inputs()
        refs = latents((16, 16), (16, 16))
        with torch.inference_mode():
            aligned = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED, S.ALIGNED], [1.0, 1.0], {})
            beside = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED, S.BESIDE], [1.0, 1.0], {})
            bag = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.BAG, S.ALIGNED], [1.0, 1.0], {})
        assert not torch.allclose(aligned, beside, atol=1e-6)
        assert not torch.allclose(aligned, bag, atol=1e-6)
        assert not torch.allclose(beside, bag, atol=1e-6)

    def test_boost_changes_the_result_the_same_way_under_both_scalings(self, host, real_dit):
        x, t, context = inputs()
        refs = latents((16, 16), (12, 12))
        with torch.inference_mode():
            plain = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED, S.ALIGNED], [1.0, 1.0], {})
            scaled = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED, S.ALIGNED], [1.0, 2.0], {}, honours_scale=True)
            rescaled = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.ALIGNED, S.ALIGNED], [1.0, 2.0], {}, honours_scale=False)
        assert not torch.allclose(plain, scaled, atol=1e-6)
        assert torch.allclose(scaled, rescaled, atol=1e-9, rtol=0)

    def test_a_batch_shares_the_references_and_each_item_is_its_own(self, host, real_dit):
        x, t, context = inputs(batch=2)
        refs = latents((16, 16))
        with torch.inference_mode():
            both = F.edit_forward(real_dit, x, t, context.clone(), refs, [S.BESIDE], [2.0], {})
            first = F.edit_forward(real_dit, x[:1], t[:1], context[:1].clone(), refs, [S.BESIDE], [2.0], {})
        assert both.shape == x.shape
        assert torch.allclose(both[:1], first, atol=1e-9, rtol=0)

    def test_a_four_dimensional_latent_comes_back_in_four(self, host, real_dit):
        x, t, context = inputs()
        with torch.inference_mode():
            out = F.edit_forward(real_dit, x.squeeze(2), t, context.clone(), latents((16, 16)), [S.ALIGNED], [1.0], {})
        assert out.shape == x.squeeze(2).shape

    def test_an_odd_latent_is_padded_and_cropped_back(self, host, real_dit):
        x, t, context = inputs(h=15, w=16)
        with torch.inference_mode():
            out = F.edit_forward(real_dit, x, t, context.clone(), latents((16, 16)), [S.ALIGNED], [1.0], {})
        assert out.shape == x.shape

    def test_a_reference_larger_than_the_target_is_refused(self, host, real_dit):
        x, t, context = inputs()
        with pytest.raises(ValueError):
            F.edit_forward(real_dit, x, t, context.clone(), latents((18, 16)), [S.ALIGNED], [1.0], {})

    def test_one_placement_and_boost_per_reference(self, host, real_dit):
        x, t, context = inputs()
        with pytest.raises(ValueError):
            F.edit_forward(real_dit, x, t, context.clone(), latents((16, 16)), [], [1.0], {})


class Recorder(torch.nn.Module):
    """Stands in for the position embedder and keeps every ``pos`` it was given."""

    def __init__(self, inner, seen):
        super().__init__()
        self.inner, self.seen = inner, seen

    def forward(self, pos):
        self.seen.append(pos.clone())
        return self.inner(pos)


class TestPositions:
    def test_the_embedder_sees_the_centred_offset_and_the_placement(self, host, real_dit):
        """The reference's ids: frame 1, its own grid at (target - reference) / 2,
        or shifted right of the canvas when placed beside; the target at frame 0."""
        seen = []
        embedder = real_dit.pe_embedder
        real_dit.pe_embedder = Recorder(embedder, seen)
        try:
            x, t, context = inputs()  # 16x16 latent: an 8x8 token grid
            ref = latents((12, 16))   # 6x8 tokens
            with torch.inference_mode():
                F.edit_forward(real_dit, x, t, context.clone(), ref, [S.ALIGNED], [1.0], {})
                F.edit_forward(real_dit, x, t, context.clone(), ref, [S.BESIDE], [1.0], {})
        finally:
            real_dit.pe_embedder = embedder
        text = context.shape[1]
        aligned, beside = seen
        ref_ids = aligned[0, text : text + 48]
        assert torch.all(ref_ids[:, 0] == 1)
        assert ref_ids[0].tolist() == [1.0, 1.0, 0.0]        # (8 - 6) / 2 = 1 row down, column 0
        assert ref_ids[-1].tolist() == [1.0, 6.0, 7.0]
        target = aligned[0, text + 48 :]
        assert torch.all(target[:, 0] == 0) and target[0].tolist() == [0.0, 0.0, 0.0] and target[-1].tolist() == [0.0, 7.0, 7.0]
        assert torch.all(aligned[0, :text] == 0)
        beside_ids = beside[0, text : text + 48]
        assert beside_ids[0].tolist() == [1.0, 1.0, 8.0]      # right of the 8-wide canvas
        assert beside_ids[-1].tolist() == [1.0, 6.0, 15.0]

    def test_a_half_token_offset_reaches_the_embedder(self, host, real_dit):
        seen = []
        embedder = real_dit.pe_embedder
        real_dit.pe_embedder = Recorder(embedder, seen)
        try:
            x, t, context = inputs()
            with torch.inference_mode():
                F.edit_forward(real_dit, x, t, context.clone(), latents((14, 16)), [S.ALIGNED], [1.0], {})  # 7 rows in 8
        finally:
            real_dit.pe_embedder = embedder
        ref_ids = seen[0][0, context.shape[1] : context.shape[1] + 56]
        assert ref_ids[0].tolist() == [1.0, 0.5, 0.0]
