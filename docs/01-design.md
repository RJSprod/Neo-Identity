# 01 — The design, as confirmed

Date: 2026-10-16. The owner's brief: a new extension in this repository that
lets the Krea 2 Identity Edit **V2** LoRA be used exactly as v1.2 is used today
in Forge Neo, compatible with Model Chain (`SD-Neo-ModelSwitchRefiner`), with
nothing else affected, reversible by disabling the extension, and fast. A
feasibility review (*Forge Neo + Krea 2 Identity Edit V2*, 16 October 2026) was
given as direction, not specification. This document is what was confirmed
against the sources and built.

## Sources read

| Source | Revision | What it settled |
| --- | --- | --- |
| Forge Neo (`Haoming02/sd-webui-forge-classic`, `neo`) | `46af880c` | the engine's capture and conditioning, the transformer's sequence, the hooks and their order, the prompt cache, the attention dispatch |
| ComfyUI-Krea2Edit (`lbouaraba/comfyui-krea2edit`) | `13ef60f2`, v2.0.0 | the V2 recipe: placement, fit geometry, boost, grounding, the negative |
| krea2edit-trainer (`lbouaraba/krea2edit-trainer`) | `56bf37b8` | the geometry contract (`docs/GEOMETRY.md`): fractional offsets, crop-to-grid, grounding cap 768 / jitter 384, LoRA key convention |
| `opparco/sd-forge-krea2-edit` | `ea7e5e97` | a v1.2 Forge port whose Neo adapter is stale (`KREA2_TAP_LAYERS`, `_unpack_context`); its module separation was useful, its adapter was not reused |
| `RedNodeAI/forge-neo-krea2-toolkit` | `3762db21` | a core patch; read for how it arms references into the native engine, not used |
| Model Chain | `57e449e` (main) | how Stage 2 routes references (`mc_references.encode` through `encode_first_stage`), that Stage 2 runs with `p2.scripts = None`, how edit mode is overridden |

The Hugging Face model card was not reachable from the session's network. The
V2 facts come from the node pack's README and CHANGELOG at v2.0.0: two LoRAs
(`krea2_identity_edit_v2.safetensors`, `krea2_general_edit_v2.safetensors`),
one per task; the new placement controls `ref_position` (last reference) and
`ref_position_a` (first), with *aligned (trained)* reproducing the previous
behaviour exactly; *beside* and the experimental *bag of features*; the
recommendation scene aligned + person beside; Turbo 8 steps CFG 1. The review's
"subject boost 2 and grounding 768" baseline is reported from the card and not
verified here; the extension defaults boost to 1 for the reason under *Speed*.

## What "how v1.2 works today" is

Forge Neo's native Krea 2 edit path. ImageStitch Integrated encodes its gallery
through `encode_first_stage`; the engine, while `[Krea2] Enable Reference` is
on, keeps each image's pixels in `ref_latents` (the img2img image in
`ini_latent`). `get_learned_conditioning` on the positive prompt reads
`[ini_latent, *ref_latents]`, scales each to a 768² area for Qwen3-VL, VAE-encodes
each at its own size, and publishes the latents through `dynamic_args.ref_latents`;
the transformer resizes them in latent space to the target's size (centre crop)
and lays `[text | refs | target]` with frames 1…N and 0. Model Chain's Stage 2
takes the same route with its own processing object and scripts off. The LoRA
is a prompt tag. There is no fit geometry, no placement, no boost, no grounded
negative.

## The decision

Wrap the three host methods **on their classes**, with dispatchers gated by one
job's state, and take the references **from the native channels**. Reasons, in
order of weight:

1. **Model Chain's Stage 2 has no script hooks.** Anything hung on the first
   stage's model or installed from `process_before_every_sampling` never sees
   the refine. A class-level dispatcher does, and the outer job's state decides
   whether it acts. The state stays armed from one generation's
   `before_process` to the next one's: Forge runs the scripts' hooks in load
   order (checked in `modules/scripts.py`: `sort_callbacks` honours only
   `metadata.ini` order instructions, and `sorting_priority` places the UI),
   so this script's `postprocess` cannot be relied on to run after Model
   Chain's, and only releases the generation's tensors. The feasibility
   review's lifecycle (install a wrapper in `process_before_every_sampling`)
   would have left Stage 2 on the native path.
2. **No new workflow.** The owner's references already arrive through
   ImageStitch, img2img and Model Chain in the order the recipe wants (scene
   first, person last; the Stage 1 scene, then the supplemental person). A
   gallery of our own would be a second place to put the same pictures and
   would not reach Stage 2.
3. **Reversible by construction.** Off, every dispatcher is a direct call to
   the original; disabled, nothing is imported; nothing is written anywhere.
4. **Nothing else touched.** Only `Krea2` and `SingleStreamDiT` are wrapped,
   and `SingleStreamDiT` is Krea 2's alone in Forge Neo.

Capture semantics are the host's: the engine stores references while its own
toggle is on, and the script asks for that toggle through `override_settings`
(`setdefault`, so Model Chain's *Disable* and the user's own override win).

## The recipe, point by point

| Recipe element | Upstream | Here |
| --- | --- | --- |
| Sequence `[text | refs… | target]`, frames 1…N / 0 | `krea2_edit_forward` | `forward.edit_forward` on the host's submodules |
| Fit geometry (crop tolerance 8 %, /16 floor capped at the target, crop-to-grid at scale) | `_fit_encode_image(fit)` | `geometry.plan_fit`, worked by hand in the tests |
| Crop geometry (v1/v1.1) | `_fit_encode_image(crop)` | `geometry.plan_crop` |
| Fractional centred offsets | `_imgids_offset` | `geometry.reference_ids` |
| Placement aligned / beside / bag | `_place_refs` | `geometry.place`, equal to a port of upstream's function in the tests |
| Boost, last ref = subject, earlier = scene | `_ref_attn_bias` dense mask | `boost.augment` padded head dimensions; equal to the dense mask in the tests |
| Grounding cap on the longest side, never enlarged | `Krea2EditGroundedEncode._prep` | `geometry.grounding_pixels` |
| The trained template, 12 tapped layers, system prefix sliced | the node's template | the host's own `Qwen3VL4BEngine`, which is the same template |
| Grounded negative at CFG > 1 | a second grounded encode | the conditioning dispatcher grounds both prompts (switch, default on) |
| Pre-encode before sampling | `target_latent` | at conditioning time, from the prompt's announced size; and in `process_before_every_sampling` for the sampled size (hires included) |
| Cache identity includes the images | not in Forge | every capture invalidates the host's prompt cache; the script clears it per job and per batch |

## Speed

The owner asked to be told before any commit if V2 is significantly slower.
It is not, in the like-for-like configuration:

- **Boost 1**: the native kernels, the same or fewer tokens, no mask. Native
  step time.
- **Boost ≠ 1**: eight extra head dimensions (~6 % of the attention's
  arithmetic, a few percent of a step). Under SageAttention the boosted
  attention runs on PyTorch's flash kernel: a share of Sage's advantage on
  those blocks for that generation. The dense-mask alternative would have cost
  ~300 MB at 1 MP with two references and forced every backend off its flash
  path; it was rejected for that reason.
- **Negative grounding** (CFG > 1): one more Qwen3-VL pass per conditioning,
  sub-second. Nothing at CFG 1.
- **VAE encodes**: one per reference per output size, before sampling, where
  the native path also encodes.

The default boost is 1 so that the default configuration is native speed; the
card's suggested 2 is one slider away and its cost is stated on the slider.

## Rejected

- **A dense attention mask for the boost** (upstream, both Forge ports): memory
  and kernel cost; SageAttention 1 ignores it silently.
- **An own reference gallery**: see decision 2.
- **Instance-level patching per generation, or a `model_function_wrapper` on a
  cloned patcher**: misses Model Chain's Stage 2; a `model_function_wrapper`
  would also have to reproduce `KModel.apply_model`'s predictor conversions.
- **Changing Model Chain**: not needed; the seam is `lifecycle.activate`.
- **Capturing references regardless of the host's toggle**: would turn Model
  Chain's *Disable* into edit mode.
- **A boost mask and a system-prompt override**: not needed for the brief.

## Risks and the first run

- The V2 LoRA's weights were not inspected (Hugging Face unreachable). Its keys
  follow the ComfyUI convention per the trainer's contract, the same as v1.2, so
  Forge's loader should take it as it takes v1.2. If the owner's original
  failure with V2 was a LoRA load error, that is in the host's loader and this
  extension cannot change it; the console line from the LoRA loader is the
  thing to read first.
- Nothing has run on a GPU. The parity tests run the host's own transformer on
  the CPU at a tiny size; the real model's behaviour under the recipe is
  reasoned, not observed. First-run console lines: `[Neo Identity] N
  reference(s), …`, `[Neo Identity] references for WxH: …`.
- SageAttention 2's `sageattn` has no `attn_mask` argument; a boosted call under
  Sage 2 with the dense mask would have raised and fallen back on every block.
  The padded approach never passes a mask, so Sage 2 falls back cleanly on the
  head size alone. Untested on a real Sage install.
