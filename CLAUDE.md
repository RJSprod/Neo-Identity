# Neo-Identity

A Forge Neo extension: **Krea 2 Identity Edit V2** on the references Forge
already collects (ImageStitch, img2img, Model Chain's Stage 2). `README.md` is
the behavioural specification — authoritative for declared behaviour, updated
when behaviour changes. `docs/01-design.md` records the design as confirmed
against the host and the upstream recipe, with what was rejected and why.

Sibling repositories of the same owner: `RJSprod/SD-Neo-ModelSwitchRefiner`
(Model Chain, the Forge Assistant, LLM Studio, Voice) and
`RJSprod/a1111-mini-paint-NEO`. This extension changes nothing in either; what
it has to know about Model Chain is in the traps below.

## Layout

| Path | Role |
| --- | --- |
| `scripts/neo_identity_edit.py` | the always-on script: the accordion, the hooks, the infotext |
| `neo_identity/settings.py` | the per-generation settings, their choices, the infotext keys |
| `neo_identity/geometry.py` | fit and crop geometry, grounding resize, position ids, placement |
| `neo_identity/boost.py` | the boost as padded head dimensions; the dense mask for the tests |
| `neo_identity/forward.py` | the edit forward on the host's `SingleStreamDiT` submodules |
| `neo_identity/references.py` | one job's references: pixels, grounding images, fitted latents |
| `neo_identity/lifecycle.py` | the class-level dispatchers, the job state, cache invalidation |
| `neo_identity/compat.py` | the host contract (pinned to Forge Neo `46af880c`), helpers |
| `tests/` | pytest; `conftest.py` is the Forge stand-in, `forge_source.py` lifts the real transformer out of a checkout |
| `tools/run_tests.py` | the suite on every core; `NEO_IDENTITY_FORGE_ROOT` names the Forge checkout |

## How it works, in one paragraph

Forge Neo's Krea 2 engine captures reference pixels in `encode_first_stage`
(ImageStitch, img2img, Model Chain all go through it) and reads them in
`get_learned_conditioning`; its transformer builds the sequence in `forward`.
`lifecycle.install()` wraps those three methods *on the classes* with
dispatchers that call the originals unless a job is active. `before_process`
activates (and adds `krea2_do_reference` to the generation's `override_settings`
so the host captures); `postprocess` releases the generation's tensors and
leaves the recipe armed until the next generation's `before_process`. While active, the conditioning
dispatcher snapshots `[ini_latent, *ref_latents]` in native order, grounds both
prompts on them at the grounding size, and encodes the fitted latents for the
announced output size; the forward dispatcher runs `forward.edit_forward` on the
transformer the job conditioned, natively on any other.

## Traps — learned here, written nowhere else

- **Model Chain's Stage 2 runs with `p2.scripts = None`**, inside the outer
  generation's `postprocess` phase. No script hook of ours fires for it; only a
  class-level dispatcher sees it, and only while the state is armed. **Forge
  runs the scripts' hooks in load order** (`alwayson_scripts` in the order the
  extension folders sort), not by `sorting_priority`, which only places the
  accordion in the tab — so this script's `postprocess` runs *before* Model
  Chain's on a folder sorted before `SD-Neo-ModelSwitchRefiner`. That is why
  `postprocess` only releases the generation's tensors (`lifecycle.release`)
  and the state stays armed until the next `before_process`. Deactivating in
  `postprocess` sends Stage 2 down the native path; a test holds it.
- **The engine keeps reference *pixels*, not latents**, in `ref_latents` and
  `ini_latent`: `(1, H, W, 3)` in [0, 1] on the CPU, one per captured image.
  Native conditioning consumes `ini_latent` (sets it to None) on the positive
  prompt; `references.snapshot` does the same, and keeps the job's set across
  the negative-first order (CFG > 1 computes the negative first), the second
  batch and the hires pass by the identity of the gallery's tensors.
- **The host caches conditioning on the text, never on the images.** A new
  reference with the same prompt would reuse the old grounding. ImageStitch
  clears the cache through its `p`; the engine has no `p`, so
  `lifecycle.invalidate_prompt_cache` empties the class-level lists in place —
  which reaches a fresh `p2` whose instance attribute is that very list — and
  the script clears the outer `p`'s own lists in `process` and `process_batch`.
  Both are needed; a test holds each.
- **The native text fusion adds into its input in place.** `txtfusion` reshapes
  the context and `add_`s into it, so a context tensor passed to the forward
  twice gives two different answers. Forge concatenates a fresh context every
  step, so the host never notices; a test that reuses a context tensor does.
  Every forward call in the tests gets `context.clone()`.
- **Capture follows the host's toggle, not ours.** `encode_first_stage` only
  stores a reference while `opts.krea2_do_reference` is on. The script asks for
  it through `p.override_settings.setdefault(...)`, so the host applies it for
  the generation and restores the user's value, and an override already there
  (the user's, or Model Chain's Stage 2 *Disable*) wins. Do not capture
  regardless of the toggle: that would turn Model Chain's *Disable* into edit
  mode.
- **The boost never builds a mask.** Eight extra head dimensions carry
  `log(boost)` into the logits (`boost.augment`). Backends that pass an explicit
  `scale` to their kernel get `scale = 1/sqrt(d)`; Flash and xformers ignore
  `scale`, so the queries are rescaled by `sqrt(d'/d)` instead
  (`compat.attention_honours_scale`, by the attention function's name).
  SageAttention falls back to PyTorch for a head over 128 wide, with the
  keyword arguments intact, so it counts as honouring. The override removes
  itself from the options it passes down: a backend that falls back to another
  attention function would otherwise apply the bias twice (b² instead of b).
  The key bias is computed in float64 and cast once; in float32 the two
  scalings disagreed at the eighth digit.
- **`SingleStreamDiT.forward` is `(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs)`**
  and `KModel.apply_model` calls it with `context=`, `control=`,
  `transformer_options=` as keywords. The dispatcher's signature mirrors it;
  `compat.check` refuses a host whose signature differs at the start.
- **References arrive at the forward already fitted.** `edit_forward` refuses a
  reference larger than the target instead of resizing a latent: a latent
  resize is the native geometry the recipe replaces. Fit at the *padded* grid:
  a latent that is not a multiple of the patch is padded by the transformer,
  and the reference is fitted to `rows * 16` px, not the requested size
  (`lifecycle.target_size`).
- **Nothing of Forge is copied into this repository.** The tests read the
  host's own files from a checkout (`tests/forge_source.py`, by `ast`); the
  stand-in attention and rope are written here. Keep it that way: Forge Neo is
  AGPL-3.0 and this repository is Apache-2.0.

## Conventions

- Develop on the designated branch; push when complete; the owner opens pull
  requests from the Claude Code UI. The default branch is `main`.
- **No model identifier** in commit messages, PR text, code comments or
  anything pushed. Commits end with `Co-Authored-By: Claude <noreply@anthropic.com>`
  and the session link.
- **Run the whole suite before every push**: `python3 tools/run_tests.py`,
  with `NEO_IDENTITY_FORGE_ROOT` pointing at a Forge Neo checkout so the parity
  tests run instead of skipping (the web session hook clones one).
- **Mutation-check new tests**: revert the behaviour a test holds and watch it
  fail; delete `__pycache__` after every restore. The first build checked 25
  such mutations, all caught.
- **A test never writes outside `tmp_path`**, and never imports the real Forge
  runtime: only `backend/nn/krea.py` and single functions lifted by name.
- **Update `README.md`** when declared behaviour changes.

## Not built, and why

- A **boost mask** (a region of the subject to boost) and a **grounding system
  prompt override**: upstream has both; neither is needed for "use V2 like
  v1.2", and each adds a control to a tab the owner wants quiet.
- **An own reference gallery**: the owner's references already come through
  ImageStitch, img2img and Model Chain; a second gallery would be a second
  workflow and would not reach Model Chain's Stage 2.
- **Changes in Model Chain**: none were needed. If Model Chain ever wants to
  choose placement per Stage 2 image, `neo_identity.lifecycle` is the seam
  (`activate(settings)` / `deactivate()`).
- **`torch.compile` support**: the compiled module stands in for the
  transformer; the dispatcher detects it and runs natively with a note.
- Nothing has run on the owner's machine yet. First-run checks are in the
  README's *Limits*.
