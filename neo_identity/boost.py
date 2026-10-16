"""Reference boost without an attention mask.

Upstream's ``ref_boost`` adds ``log(boost)`` to the attention logits of every
target query against every key of a boosted reference, so the reference's keys
take a ``boost``-times larger share of the softmax before renormalisation. The
nodes do that with a dense additive mask of shape ``[1, 1, L, L]``: a 1 MP output
with two same-sized references is about 12,000 tokens, so the mask alone is
some 300 MB in bf16, and an attention kernel handed an arbitrary mask cannot use
its flash path. Under SageAttention 1 the mask is accepted and ignored, which
upstream documents as "boost has no effect".

This module gets the same logits a different way. Each attention head's queries
and keys gain eight extra dimensions. A target query carries a 1 in each of
them; a key carries ``log(boost) * sqrt(d') / 8`` in each when it belongs to a
boosted reference and 0 otherwise, where ``d'`` is the padded head size. Their
dot product, scaled by the kernel's ``1 / sqrt(d')``, is exactly ``log(boost)``
for a target query against a boosted key and exactly 0 everywhere else. The
real head dimensions are rescaled so that ``q . k / sqrt(d')`` still equals the
model's own ``q . k / sqrt(d)`` -- or, where the backend takes an explicit
``scale``, left alone with ``scale = 1 / sqrt(d)`` passed. The value vectors are
zero-padded and the padding is cut off the output. No mask is built, every
backend keeps its own kernel (SageAttention falls back to PyTorch's for a head
dimension over 128, which is still the flash kernel), and the extra work is the
eight dimensions: about 6 % of the attention's arithmetic.

The dense mask stays here as ``dense_bias`` for the tests, which hold the two
equal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

PAD = 8
"""Extra head dimensions. Eight keeps the padded head size a multiple of eight,
which flash kernels require; the bias is spread over all eight so no single
entry is large."""

HONOURS_SCALE = frozenset({"attention_pytorch", "attention_basic", "attention_sage", "attention_comfy_kitchen_int8"})
"""Forge attention functions that pass an explicit ``scale`` to their kernel.
SageAttention counts because a padded head is over 128 wide, which makes it
fall back to ``attention_pytorch`` with the keyword arguments intact. Flash and
xformers use the kernel's default scale, so the queries are rescaled instead."""

NEUTRAL = 1.0
FLOOR = 1e-4
"""A boost of 0 would be a logit of minus infinity; upstream floors it here."""


@dataclass(frozen=True)
class BiasPlan:
    """Where the bias goes on the ``[text | refs... | target]`` sequence."""

    spans: tuple[tuple[int, int, float], ...]
    """(start, stop, log_boost) for every boosted reference, in token indices."""
    rows_from: int
    """The first target token: the bias applies to queries from here on."""
    length: int
    """The whole sequence length."""

    @property
    def active(self) -> bool:
        return bool(self.spans)


def active(boosts) -> bool:
    return any(float(b) != NEUTRAL for b in boosts)


def plan_bias(boosts, text_tokens: int, reference_tokens: list[int], target_tokens: int) -> BiasPlan:
    """The plan for ``boosts`` aligned with ``reference_tokens`` (one count per
    reference, in sequence order)."""
    if len(boosts) != len(reference_tokens):
        raise ValueError(f"{len(boosts)} boosts for {len(reference_tokens)} references")
    spans = []
    offset = text_tokens
    for boost, count in zip(boosts, reference_tokens):
        boost = float(boost)
        if boost != NEUTRAL:
            spans.append((offset, offset + count, math.log(max(boost, FLOOR))))
        offset += count
    return BiasPlan(tuple(spans), offset, offset + target_tokens)


def key_bias(plan: BiasPlan, device=None, dtype=torch.float32) -> torch.Tensor:
    """``log(boost)`` per key token, 0 for keys outside every boosted reference."""
    bias = torch.zeros(plan.length, device=device, dtype=dtype)
    for start, stop, log_boost in plan.spans:
        bias[start:stop] = log_boost
    return bias


def dense_bias(plan: BiasPlan, device=None, dtype=torch.float32) -> torch.Tensor:
    """Upstream's additive mask, ``[1, 1, L, L]``: the reference the padding is
    checked against, and nothing the forward uses."""
    bias = torch.zeros(1, 1, plan.length, plan.length, device=device, dtype=dtype)
    bias[:, :, plan.rows_from :, :] = key_bias(plan, device, dtype)[None, None, None, :]
    return bias


def augment(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, plan: BiasPlan, honours_scale: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """``(q', k', v', extra_kwargs)`` whose attention equals attention under the
    dense bias.

    Shapes are ``(B, H, L, D)``. The returned ``extra_kwargs`` carries
    ``scale`` when the backend honours one, and is empty when the queries were
    rescaled instead.
    """
    if q.shape[-2] != plan.length:
        raise ValueError(f"sequence of {q.shape[-2]} tokens does not match the plan's {plan.length}")
    d = q.shape[-1]
    d_padded = d + PAD
    if honours_scale:
        q_main = q
        extra = {"scale": d**-0.5}
        bias_scale = math.sqrt(d)
    else:
        q_main = q * math.sqrt(d_padded / d)
        extra = {}
        bias_scale = math.sqrt(d_padded)

    rows = torch.zeros(plan.length, device=q.device, dtype=q.dtype)
    rows[plan.rows_from :] = 1.0
    # Worked in float64 and cast once: a float32 log(boost) would make the two
    # scalings disagree at the eighth digit, which the tests would see.
    keys = key_bias(plan, q.device, torch.float64).mul_(bias_scale / PAD).to(q.dtype)

    q_pad = rows.view(1, 1, -1, 1).expand(q.shape[0], q.shape[1], -1, PAD)
    k_pad = keys.view(1, 1, -1, 1).expand(k.shape[0], k.shape[1], -1, PAD)
    v_pad = v.new_zeros(v.shape[:-1] + (PAD,))

    q2 = torch.cat((q_main, q_pad), dim=-1)
    k2 = torch.cat((k, k_pad), dim=-1)
    v2 = torch.cat((v, v_pad), dim=-1)
    return q2, k2, v2, extra


def strip(out: torch.Tensor, heads: int, head_dim: int, skip_output_reshape: bool) -> torch.Tensor:
    """The attention output without the padded dimensions.

    Forge's attention functions return ``(B, H, L, D')`` when asked to skip the
    output reshape and ``(B, L, H * D')`` otherwise.
    """
    if skip_output_reshape:
        return out[..., :head_dim]
    b, l, _ = out.shape
    return out.reshape(b, l, heads, -1)[..., :head_dim].reshape(b, l, heads * head_dim)


def attention_override(plan: BiasPlan, head_dim: int, honours_scale: bool):
    """A callable for ``transformer_options["optimized_attention_override"]``.

    Forge's ``wrap_attn`` hands it the undecorated attention function and the
    call's arguments. Calls that are not the main stream's self-attention -- a
    different sequence length, a different head size, or a mask already present
    -- go through untouched, as does any call once the padding has been applied.
    The override is removed from the options passed down, so a backend that
    falls back to another attention function cannot re-enter it and apply the
    bias twice.
    """

    def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        if mask is not None or not skip_reshape or q.ndim != 4 or q.shape[-1] != head_dim or q.shape[-2] != plan.length:
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
        options = dict(kwargs.get("transformer_options") or {})
        options.pop("optimized_attention_override", None)
        kwargs["transformer_options"] = options
        if "scale" in kwargs:
            raise ValueError("the main stream's attention is not expected to pass its own scale")
        q2, k2, v2, extra = augment(q, k, v, plan, honours_scale)
        out = func(q2, k2, v2, heads, mask=None, attn_precision=attn_precision, skip_reshape=True, skip_output_reshape=skip_output_reshape, **kwargs, **extra)
        return strip(out, heads, head_dim, skip_output_reshape)

    return override
