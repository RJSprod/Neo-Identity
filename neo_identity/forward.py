"""The edit forward: Krea 2's transformer run on ``[text | refs... | target]``.

A port of upstream's ``krea2_edit_forward`` (lbouaraba/comfyui-krea2edit,
Apache-2.0) onto Forge Neo's ``SingleStreamDiT``. It uses the model's own
submodules -- the LoRA is already merged into them by the time this runs -- and
differs from the native forward in exactly the places the V2 recipe needs:

* the references arrive already fitted to the target grid, so they keep their
  own grid instead of being resized in latent space to the target's;
* each reference's tokens sit at the centred, fractional offset of its grid in
  the target's, and then where its placement puts them (aligned, beside, bag);
* a reference's boost rides the attention through ``boost.attention_override``.

Everything else -- the text fusion, the timestep path, the final layer, the
target-only output cropped back to the unpadded latent -- is the native forward
line for line, which the tests hold: with same-sized references, aligned and
unboosted, this forward and the native one agree to floating-point noise.
"""

from __future__ import annotations

import logging

import torch
from einops import rearrange

from . import boost as B
from . import geometry as G

logger = logging.getLogger("neo_identity")

_host = None


def host():
    """The two host functions the forward borrows, imported when first needed.

    ``pad_to_patch_size`` pads a latent to the patch grid the way the native
    forward does (circular for the target, replicate for a reference);
    ``timestep_embedding`` is Flux's sinusoidal embedding, which Krea 2 shares.
    """
    global _host
    if _host is None:
        from backend.nn.flux import timestep_embedding
        from backend.utils import pad_to_patch_size

        _host = (pad_to_patch_size, timestep_embedding)
    return _host


def edit_forward(dit, x, timesteps, context, references, placements, boosts, transformer_options=None, *, honours_scale: bool = True):
    """Run ``dit`` on the target ``x`` with ``references`` prepended.

    ``x``: the noisy target latent, ``(B, C, H, W)`` or ``(B, C, 1, H, W)`` as
    Forge's Wan-style latent arrives. ``references``: fitted latents in the
    transformer's latent space, each ``(1, C, h, w)`` or ``(B, C, h, w)``, no
    larger than the target on either side. ``placements`` and ``boosts``: one
    per reference. Returns the target's prediction in ``x``'s shape.
    """
    pad_to_patch_size, timestep_embedding = host()
    transformer_options = transformer_options or {}

    five_d = x.ndim == 5
    if five_d:
        x = x.squeeze(2)
    bs, _, H_orig, W_orig = x.shape
    patch = dit.patch

    x = pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    if len(placements) != len(references) or len(boosts) != len(references):
        raise ValueError("one placement and one boost per reference")

    refs = []
    grids = []
    for ref in references:
        ref = ref.squeeze(2) if ref.ndim == 5 else ref
        ref = ref.to(device=x.device, dtype=x.dtype)
        if ref.shape[0] != bs:
            ref = ref[:1].expand(bs, *ref.shape[1:])
        if ref.shape[-2] > H or ref.shape[-1] > W:
            raise ValueError(f"a reference of {tuple(ref.shape[-2:])} is larger than the target's {(H, W)}; it should have been fitted first")
        ref = pad_to_patch_size(ref, (patch, patch), padding_mode="replicate")
        refs.append(ref)
        grids.append((ref.shape[-2] // patch, ref.shape[-1] // patch))

    img = dit.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    ref_tokens = [dit.first(rearrange(r, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)) for r in refs]

    t = dit.tmlp(timestep_embedding(timesteps, dit.tdim).unsqueeze(1).to(img.dtype))
    tvec = dit.tproj(t)

    context = dit.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = dit.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    reflens = [r.shape[1] for r in ref_tokens]
    reflen = sum(reflens)
    combined = torch.cat((context, *ref_tokens, img), dim=1)

    device = combined.device
    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    ref_ids = [G.reference_ids(bs, i + 1, gh, gw, h_, w_, device) for i, (gh, gw) in enumerate(grids)]
    ref_ids = G.place(ref_ids, list(placements), h_, w_)
    pos = torch.cat((txtpos, *ref_ids, G.target_ids(bs, h_, w_, device)), dim=1)
    freqs = dit.pe_embedder(pos)

    block_options = transformer_options
    if B.active(boosts):
        plan = B.plan_bias(boosts, txtlen, reflens, imglen)
        block_options = dict(transformer_options)
        block_options["optimized_attention_override"] = B.attention_override(plan, dit.blocks[0].attn.headdim, honours_scale)

    for block in dit.blocks:
        combined = block(combined, tvec, freqs, None, transformer_options=block_options)

    final = dit.last(combined, t)
    out = final[:, txtlen + reflen : txtlen + reflen + imglen, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=dit.channels)
    out = out[:, :, :H_orig, :W_orig]
    return out.unsqueeze(2) if five_d else out
