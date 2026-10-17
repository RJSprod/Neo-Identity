# Neo Identity — Krea 2 Identity Edit V2 for Forge Neo

A Forge Neo extension that runs Krea 2's edit path the way the **Krea 2 Identity
Edit V2** LoRA ([conradlocke/krea2-identity-edit](https://huggingface.co/conradlocke/krea2-identity-edit))
was trained: the reference fitted to the output grid, each reference placed
*aligned* onto the layout or *beside* it, an optional attention *boost*, and
both prompts grounded on the images. You use it exactly as you use the v1.2
LoRA today — the same ImageStitch gallery, the same img2img image, the same
Model Chain Stage 2 — with one accordion switched on and the V2 LoRA in the
prompt instead of v1.2.

This README is the behavioural specification. When behaviour changes, it changes.

---

## What you need

| Thing | Where |
| --- | --- |
| **Forge Neo** (`Haoming02/sd-webui-forge-classic`, branch `neo`) with native Krea 2 support | written against commit `46af880c`; the host's interfaces are checked at start and the extension stays inert, with a reason on the console, if they differ |
| A **Krea 2** checkpoint, Raw or Turbo, with its **Qwen3-VL 4B** text encoder and **VAE** | selected in Forge as for any Krea 2 generation |
| The **V2 LoRA**: `krea2_identity_edit_v2.safetensors` (or `krea2_general_edit_v2.safetensors` for instruction edits) | your LoRA folder; one task LoRA at a time, at strength 1 |
| Nothing else | no Python dependencies, no model downloads, no files written outside Forge's own settings |

Install: clone this repository into Forge's `extensions` folder and restart.

```
git clone https://github.com/RJSprod/Neo-Identity extensions/Neo-Identity
```

---

## Using it

The accordion **Krea 2 Identity Edit V2** sits under txt2img and img2img. Open
it, tick it, generate. The references are whatever Forge already hands Krea 2:

| Where you work | Where the references come from | Order in the sequence |
| --- | --- | --- |
| **txt2img** | the **ImageStitch Integrated** gallery | the gallery's order |
| **img2img** | the img2img image, plus ImageStitch's gallery if it is on | the img2img image first, then the gallery |
| **Model Chain Stage 2** (edit mode Enable, or Auto with the toggle on) | the Stage 1 image, plus the Stage 2 reference images (Pass Through or Decoupled) | the Stage 1 image first, then the references |

Put the LoRA in the prompt as usual — `<lora:krea2_identity_edit_v2:1>` in the
txt2img or img2img prompt, or in Model Chain's Stage 2 prompt in Append or
Replace mode. The console warns when no `<lora:` tag is in the prompt, because
the edit behaviour comes from the LoRA and the base checkpoint alone will not do
it.

Settings that the model card and the node author recommend as starting points:

- **Turbo**: 8 steps, CFG 1, Euler. Most edits.
- **Raw**: about 20 steps, CFG 3, for removals and other edits that need real guidance.
- Generate at **2 megapixels or less**; above the trained range content bleeds and subjects duplicate.
- **A person into a scene**: scene first (the gallery's first image, or Model Chain's Stage 1 image), person last; scene placement *Aligned*, subject placement *Beside*.
- **Identity**: the card suggests a subject boost of about 2. The extension's default is 1 (neutral) because that is the speed of the native path; see *Speed* below.

### The roles: scene and subject

Upstream's convention is kept: the **last** reference is the **subject** (the
only image, or the person in a scene-and-person edit) and every earlier one is
**scene**. With a single reference the subject controls apply. With three or
more, the first is scene and every other follows the subject's choice.

### The controls

| Control | Default | What it does |
| --- | --- | --- |
| **Subject placement** | Aligned | *Aligned* pins the last reference's tokens onto the output's layout, token for token, the way the models were trained. *Beside* moves them to the right of the canvas in position space: identity still flows, the layout pin is gone, the person restages freely. *Bag of features* gives every token the canvas's centre, so attention is decided by content alone — experimental upstream, experimental here. |
| **Scene placement** | Aligned | The same choice for the first reference when there are two or more. |
| **Subject boost** | 1 | How hard the output attends to the subject reference: 1 is neutral, above 1 pulls toward the reference's appearance, below 1 loosens. The card suggests about 2 for identity. |
| **Scene boost** | 1 | The same dial for the first reference when there are two or more. |
| **Grounding resolution** | 768 | The longest side of each reference as Qwen3-VL sees it. Lower favours the instruction, higher favours likeness; the released LoRAs trained at a 768 cap (jittered down to 384); 0 feeds the image at its own size. The image is never enlarged for grounding. |
| **Reference geometry** | Fit | *Fit* is the v1.2 and V2 training geometry (below). *Crop* is the v1 and v1.1 geometry — centre-crop to the output's aspect ratio, then resize — kept for those weights. |
| **Ground the negative prompt** | on | At CFG above 1 the negative prompt is encoded with the same references, which is the trained unconditional. At CFG 1 the negative is not evaluated at all, so the switch changes nothing there. |

Every control is written to the infotext under a `Neo Identity …` key and
pasted back into its control. `Neo Identity references` records how many
references the pass used; it is diagnostic and pastes into nothing, because the
images themselves are not in the infotext — as with native ImageStitch, you add
them again.

---

## What the recipe does

Krea 2's native edit path in Forge Neo already builds the token sequence
`[text | reference 1 | … | reference N | target]`, with the references on
position frames 1…N and the target on frame 0, and grounds the positive prompt
on the images through Qwen3-VL. The V2 recipe differs in five places, and the
extension changes exactly those:

| | Native Forge Neo | This extension |
| --- | --- | --- |
| **Reference latent** | the image VAE-encoded at its own size (rounded to 16 px), then resized in *latent* space, centre-cropped, to the target's size | the image fitted in *pixel* space to the target grid (Fit or Crop geometry), VAE-encoded once per output size, never resized as a latent |
| **Reference position** | the reference occupies the whole target grid | the reference keeps its own grid, at the centred offset in the target's — a half-token offset where the gap is odd — then placed Aligned, Beside or as a Bag |
| **Boost** | none | a per-reference multiplier on the output's attention to that reference |
| **Grounding size** | scaled to a 768² area, up or down | the longest side capped at the grounding resolution, never enlarged |
| **Negative prompt** | text only | grounded on the same references, when asked (default on) |

Everything else — the LoRA, the text fusion, the timestep path, the sampler and
its sigmas, the model's own loading and memory management — is the host's. With
same-sized references, Aligned and unboosted, the extension's forward and the
native forward are the same computation; a test holds that against the host's
own transformer to floating-point exactness.

### The Fit geometry

Scale the source to fit inside the output. Within 8 % of the output's aspect
ratio, centre-crop to the exact ratio and fill the grid; otherwise snap the
fitted size *down* to the 16 px token grid (never above the output's own) and
centre-crop the source so the fitted axis lands on that grid at the same scale
exactly — a resize from the scaled size down to the snapped size would squash
content by up to 15 px, and that misregistration shows as a doubled seam at the
reference's edge. The reference's tokens then sit at `(target − reference) / 2`
tokens from the top-left: fractional, because an integer floor put odd-gap
references 8 px off centre. This is the trainer's contract
(`krea2edit-trainer/docs/GEOMETRY.md`) and the node's `fit`; `tests/test_geometry.py`
works the numbers by hand.

### The boost, without a mask

Upstream applies a boost as an additive attention mask of `log(boost)` on every
target-query × reference-key logit: a dense `[1, 1, L, L]` tensor, about 300 MB
at a megapixel with two references, which keeps every attention kernel off its
flash path and which SageAttention 1 accepts and ignores. Here the same logits
come from eight extra head dimensions: a target query carries a one in each, a
boosted reference's key carries `log(boost)` spread across them, and their dot
product adds exactly `log(boost)` to the logit and exactly nothing anywhere
else; the real head dimensions are rescaled, or an explicit scale passed, so
the model's own logits are unchanged. No mask is built, every backend keeps its
kernel, and the extra work is the eight dimensions. `tests/test_boost.py` holds
the padded attention equal to the dense mask.

---

## Speed

Measured against the native path on the same card, the recipe costs:

- **Boost at 1 (the default): nothing.** The sequence is the same length or shorter (a fitted reference has as many tokens as a native one at most), the kernels are the host's, and no mask exists. Step time is the native step time.
- **Boost away from 1:** about 6 % more arithmetic in the transformer's attention (eight extra dimensions on a 128-wide head), so a few percent of the step. Under **SageAttention** the boosted attention runs on PyTorch's flash kernel instead of Sage's int8 one — still no mask — which costs part of Sage's advantage for those blocks; the console says so once per generation. Under PyTorch, Flash or xformers attention there is no change of kernel.
- **Grounding the negative at CFG above 1:** one more Qwen3-VL pass with the images when the conditioning is computed, a fraction of a second, once per generation. Nothing at CFG 1.
- **Fitting and encoding the references:** one VAE encode per reference per output size, done when the conditioning is computed — before sampling — the same point at which the native path encodes them. A hires pass encodes them again at its size, before its sampling starts. The VAE never runs inside the step loop unless a pass samples a size nobody announced, which the console then says.

Nothing has yet been timed on the user's own machine; the numbers above are the
arithmetic of the design, not measurements. If the first run is slower than the
native path with the boosts at 1, that is a defect, not a cost of the recipe.

---

## Reverting, and the rest of Forge

- **The accordion off**: the generation runs the host's native path. The extension's wrappers are on the classes but inert; every call goes straight to the host's own method.
- **The extension disabled or uninstalled**: nothing of it runs. Nothing is written to Forge's files, settings or models, so there is nothing to undo; the v1.2 workflow is exactly as it was.
- **Other models**: only Krea 2's engine class and transformer class are wrapped. SDXL, Flux, Klein, Qwen-Image, Wan and everything else never meet the extension's code.
- **The host's own toggle**: the engine only collects references while `[Krea2] Enable Reference` is on. Switching the accordion on asks for that through the generation's override settings — the same mechanism Model Chain uses for its Stage 2 — so the host applies it for the generation and restores your value afterwards. An override already present, yours or Model Chain's *Disable*, wins.
- **Model Chain**: its Stage 2 runs with scripts switched off, through its own processing object, on the model it loads itself, inside the generation's `postprocess` phase. Because the wrappers are on the classes and the recipe stays armed from one generation's start to the next one's (this script's `postprocess` only drops the generation's tensors; Forge runs the scripts' hooks in load order, so it may run before Model Chain's), the recipe applies to a Krea 2 Stage 2 with the accordion's settings. Model Chain's own controls are untouched: its *Disable* still disables (nothing is captured under it, so the pass runs natively), its reference count check still reads the engine's list, its clearing of references between images still happens — and each new reference also clears the host's cached conditioning, so a second Stage 2 image is grounded on its own Stage 1 image and not the first one's.

---

## Limits

- **Batch size** above 1 works, with the same references for every image. In img2img the host captures the first image of a batch as the reference, as it does natively.
- **Hires fix** works: the references are fitted and encoded again at the hires size before the second pass. Not validated for quality.
- **`torch.compile`** on the transformer is not driven: the generation runs the native path with a note. Turn compile off for Identity Edit.
- **A boost mask** (a region of the subject to boost) and a **grounding system prompt override** are not built.
- **The API** takes the accordion's values through the script's arguments like any always-on script; there is no separate endpoint.
- Nothing has run on the user's machine yet. On the first run, the console lines to look for are `[Neo Identity] N reference(s), …` (the references, their sizes, the roles) and `[Neo Identity] references for WxH: …` (each reference's fit), and the infotext's `Neo Identity references`.

---

## Development

```
python3 tools/run_tests.py          # the suite, on every core
python3 tools/run_tests.py --serial
```

The tests stand in for Forge with small fakes of `modules.*`, `backend.*` and
`gradio`, and load the host's **own** `backend/nn/krea.py` from a Forge Neo
checkout for the parity tests. The checkout is found through
`NEO_IDENTITY_FORGE_ROOT`, or two directories up when this repository sits in
Forge's `extensions` folder; without one the parity tests skip and the runner
says so. Every claim above about the forward — native parity, placement, boost,
batches, odd sizes — is a test against the real transformer at a tiny size on the
CPU. The CI workflow fetches Forge Neo at the pinned commit and runs them all.

Before a push, the whole suite; a new invariant's test is checked by reverting
the behaviour it holds and watching it fail.

---

## Attribution and licence

The recipe is a port of [ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit)
(Apache-2.0) by the author of the Identity Edit LoRAs, checked against the
[krea2edit-trainer](https://github.com/lbouaraba/krea2edit-trainer)'s geometry
contract. This extension is licensed under the Apache License 2.0; see `LICENSE`
and `NOTICE`. The Krea 2 weights and the LoRAs have their own licences and are
not distributed here.
