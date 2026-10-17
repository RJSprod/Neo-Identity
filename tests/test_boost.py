"""The boost as padded head dimensions equals the boost as a dense mask."""

import math

import pytest
import torch

import conftest
from neo_identity import boost as B


def reference_attention(q, k, v, bias):
    """Attention under an additive logit bias, in float32, the long way."""
    scale = q.shape[-1] ** -0.5
    logits = torch.einsum("bhid,bhjd->bhij", q, k) * scale + bias
    return torch.einsum("bhij,bhjd->bhid", logits.softmax(-1), v)


def tensors(seed=0, batch=2, heads=3, d=16, text=5, refs=(6, 4), target=9):
    g = torch.Generator().manual_seed(seed)
    length = text + sum(refs) + target
    q, k, v = (torch.randn(batch, heads, length, d, generator=g) for _ in range(3))
    return q, k, v, text, list(refs), target


class TestPlan:
    def test_spans_cover_only_the_boosted_references(self):
        plan = B.plan_bias([2.0, 1.0, 0.5], 5, [6, 4, 3], 9)
        assert plan.spans == ((5, 11, math.log(2.0)), (15, 18, math.log(0.5)))
        assert plan.rows_from == 18 and plan.length == 27
        assert plan.active

    def test_a_neutral_plan_is_inactive(self):
        assert not B.plan_bias([1.0, 1.0], 5, [6, 4], 9).active
        assert not B.active([1.0, 1.0]) and B.active([1.0, 2.0])

    def test_a_zero_boost_is_floored_rather_than_minus_infinity(self):
        plan = B.plan_bias([0.0], 2, [3], 4)
        assert plan.spans[0][2] == math.log(B.FLOOR)

    def test_one_boost_per_reference(self):
        with pytest.raises(ValueError):
            B.plan_bias([2.0], 5, [6, 4], 9)

    def test_the_dense_bias_is_upstreams_mask(self):
        plan = B.plan_bias([2.0, 0.5], 5, [6, 4], 9)
        bias = B.dense_bias(plan)
        assert tuple(bias.shape) == (1, 1, 24, 24)
        assert torch.all(bias[:, :, :15, :] == 0)  # text and reference queries: no bias
        assert torch.all(bias[:, :, 15:, 5:11] == math.log(2.0))
        assert torch.all(bias[:, :, 15:, 11:15] == math.log(0.5))
        assert torch.all(bias[:, :, 15:, :5] == 0) and torch.all(bias[:, :, 15:, 15:] == 0)


class TestPadding:
    @pytest.mark.parametrize("honours_scale", [True, False])
    def test_padded_attention_equals_the_dense_bias(self, honours_scale):
        q, k, v, text, refs, target = tensors()
        plan = B.plan_bias([2.0, 0.5], text, refs, target)
        expected = reference_attention(q, k, v, B.dense_bias(plan))
        q2, k2, v2, extra = B.augment(q, k, v, plan, honours_scale)
        assert q2.shape[-1] == q.shape[-1] + B.PAD
        out = conftest.attention_pytorch(q2, k2, v2, q.shape[1], mask=None, skip_reshape=True, skip_output_reshape=True, **extra)
        out = B.strip(out, q.shape[1], q.shape[-1], True)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_the_scale_keyword_is_passed_only_when_honoured(self):
        q, k, v, text, refs, target = tensors()
        plan = B.plan_bias([2.0, 1.0], text, refs, target)
        _, _, _, extra = B.augment(q, k, v, plan, True)
        assert extra == {"scale": q.shape[-1] ** -0.5}
        q2, _, _, extra = B.augment(q, k, v, plan, False)
        assert extra == {}
        assert torch.allclose(q2[..., : q.shape[-1]], q * math.sqrt((q.shape[-1] + B.PAD) / q.shape[-1]))

    def test_the_padded_values_are_zero_and_stripped(self):
        q, k, v, text, refs, target = tensors()
        plan = B.plan_bias([3.0, 1.0], text, refs, target)
        _, _, v2, _ = B.augment(q, k, v, plan, True)
        assert torch.all(v2[..., q.shape[-1] :] == 0)
        flat = torch.randn(2, 24, 3 * (16 + B.PAD))
        stripped = B.strip(flat, 3, 16, False)
        assert tuple(stripped.shape) == (2, 24, 48)
        assert torch.equal(stripped[0, 0, :16], flat[0, 0, :16])
        assert torch.equal(stripped[0, 0, 16:32], flat[0, 0, 24:40])

    def test_a_sequence_of_the_wrong_length_is_refused(self):
        q, k, v, text, refs, target = tensors()
        plan = B.plan_bias([2.0, 1.0], text, refs, target + 1)
        with pytest.raises(ValueError):
            B.augment(q, k, v, plan, True)


class TestOverride:
    def test_the_override_applies_the_bias_through_the_host_protocol(self, host):
        q, k, v, text, refs, target = tensors()
        plan = B.plan_bias([2.0, 0.5], text, refs, target)
        override = B.attention_override(plan, q.shape[-1], True)
        options = {"optimized_attention_override": override, "cond_mark": "kept"}
        out = host.attention.attention_function(q, k, v, q.shape[1], mask=None, skip_reshape=True, transformer_options=options)
        expected = reference_attention(q, k, v, B.dense_bias(plan))
        expected = expected.transpose(1, 2).reshape(2, 24, -1)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_the_override_is_not_passed_down_so_a_fallback_cannot_apply_it_twice(self, host):
        q, k, v, text, refs, target = tensors()
        plan = B.plan_bias([2.0, 0.5], text, refs, target)
        seen = {}

        def func(q_, k_, v_, heads, mask=None, **kwargs):
            seen["options"] = kwargs.get("transformer_options")
            return conftest.attention_pytorch(q_, k_, v_, heads, mask=mask, **kwargs)

        override = B.attention_override(plan, q.shape[-1], True)
        override(func, q, k, v, q.shape[1], mask=None, skip_reshape=True, transformer_options={"optimized_attention_override": override, "other": 1})
        assert "optimized_attention_override" not in seen["options"]
        assert seen["options"]["other"] == 1

    @pytest.mark.parametrize("case", ["mask", "length", "head_dim", "not_skip_reshape"])
    def test_other_attention_calls_go_through_untouched(self, host, case):
        q, k, v, text, refs, target = tensors()
        plan = B.plan_bias([2.0, 0.5], text, refs, target)
        override = B.attention_override(plan, q.shape[-1], True)
        calls = []

        def func(*args, **kwargs):
            calls.append((args, kwargs))
            return torch.zeros(1)

        if case == "mask":
            override(func, q, k, v, 3, mask=torch.zeros(1, 1, 24, 24), skip_reshape=True)
            assert calls[0][0][0] is q and calls[0][1]["mask"] is not None
        elif case == "length":
            short = q[:, :, :10]
            override(func, short, short, short, 3, mask=None, skip_reshape=True)
            assert calls[0][0][0] is short
        elif case == "head_dim":
            wide = torch.randn(2, 3, 24, 32)
            override(func, wide, wide, wide, 3, mask=None, skip_reshape=True)
            assert calls[0][0][0] is wide
        else:
            flat = q.transpose(1, 2).reshape(2, 24, -1)
            override(func, flat, flat, flat, 3, mask=None, skip_reshape=False)
            assert calls[0][0][0] is flat
