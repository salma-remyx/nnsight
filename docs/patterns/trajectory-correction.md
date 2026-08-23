---
title: Mid-Generation Trajectory Correction
one_liner: Score an iterated latent for drift each step and pull it back with a step-decayed, norm-capped intervention.
tags: [pattern, interpretability, diffusion, trajectory, intervention]
related: [docs/models/diffusion-model.md, docs/usage/iter-all-next.md, docs/usage/access-and-modify.md, docs/patterns/steering.md]
sources: [src/nnsight/modeling/trajectory_correction.py, src/nnsight/intervention/tracing/iterator.py]
---

# Mid-Generation Trajectory Correction

## What this is for

Iterated generators — diffusion denoising loops, autoregressive decode, any model whose key module fires once per step — can drift away from what the prompt asked for: missing objects, wrong attributes, mismatched actions. *Trajectory correction* interleaves two roles inside the sampling loop:

- a **supervisor** that reads the latent at the current step and scores it for drift, and
- a **corrector** that turns that score into a small, bounded write back to the same latent before the next step.

The contribution adapted here is the loop structure — assess on every step, correct with gain that decays toward zero as the trajectory converges, and never touch model weights. `nnsight` already gives you the two primitives this needs: `module.output` inside `for step in tracer.iter[...]` is the supervisor's read, and an in-place `[:] =` on that same value is the corrector's write. `TrajectoryCorrector` (`nnsight.modeling`) is the controller that wires them together so the loop is three lines.

Adapted from [MLLM-Guided Semantic Correction for Text-to-Video Generation](https://arxiv.org/abs/2608.16513) (Mode 2: the MLLM judge is replaced by a pluggable, parameter-free supervisor; see [What's substituted](#whats-substituted)).

## When to use

- A diffusion trajectory is losing the prompt partway through sampling and you want to steer it back *during* generation, not post-hoc.
- You want to measure *when* drift happens — which step the trajectory first diverges — as much as fix it.
- Studying whether a mid-loop latent is causally responsible for a semantic failure: correct at step N, skip step N, and compare.
- Do **not** use this for a single-step trace. With one forward pass there is no trajectory to correct — use plain [steering](steering.md) or [ablation](ablation.md).

## Canonical pattern

On a `DiffusionModel`, monitoring the denoiser output at every step:

```python
from nnsight import DiffusionModel
from nnsight.modeling import TrajectoryCorrector, cosine_drift

sd = DiffusionModel(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
    safety_checker=None,
    dispatch=True,
)

STEPS = 50

# Pass 1 — observe the uncorrected trajectory to choose an anchor.
with sd.generate("A cat", num_inference_steps=STEPS, seed=42) as tracer:
    latents = list().save()
    for step in tracer.iter[:]:
        latents.append(sd.unet.output[0].clone())

anchor = latents[0].clone()

# Pass 2 — supervise and correct.
corrector = TrajectoryCorrector(
    scorer=cosine_drift(reference=anchor),
    total_steps=STEPS,
    anchor=anchor,
)

with sd.generate("A cat", num_inference_steps=STEPS, seed=42) as tracer:
    reports = list().save()
    for step in tracer.iter[:]:
        latent = sd.unet.output[0]
        reports.append(corrector.assess(latent, step))
        sd.unet.output[0][:] = corrector.correct(latent, step)
    output = tracer.result.save()
```

`assess` never mutates anything; `correct` returns the state unchanged unless the last report's drift exceeded `threshold`. The two-pass shape is only for picking an anchor — with a real captioner supervisor you can run single-pass (see below).

## The intervention

`correct(state, step)` returns

```
state + decay(step) · gain · drift · weight · normalize(anchor − state)
```

then clamps the result so its norm never exceeds `max_norm × ‖state‖` (default `1.0` — no growth allowed). Each factor is doing a job:

| Factor | Controls |
|---|---|
| `decay(step) = max(0, 1 − step/total_steps)` | Late corrections are gentle. A correction at step 48 of 50 is scaled by `0.04`, so it cannot wash out an almost-converged sample. |
| `drift` | How far the supervisor says the state has strayed. Bigger diagnosis, bigger pull. |
| `weight` | The supervisor's per-step confidence, if it has one (`DriftReport.weight`). |
| `normalize(anchor − state)` | Direction only — magnitude comes from the factors above, not from how far apart the states happen to be. |
| `max_norm` clamp | Worst-case bound if the supervisor mis-scores a step. |

Set `direction="negative"` to apply `−normalize(state)` instead — a pure norm reduction with no anchor, useful when you have no meaningful target state.

## Supervisors

`scorer` is any callable `state, step -> DriftReport`. Two ship with the module and both are parameter-free:

### `cosine_drift(reference)` — geometric drift

`1 − cos(state, reference)`. Reads as "how orthogonal has the latent become to where the trajectory anchored". Cheap, works on any tensor, needs a reference state — the two-pass pattern above, or a prompt-conditioned target.

### `token_overlap_drift(prompt, describe=...)` — lexical drift

`1 −` (fraction of the prompt's content words present in a textual description of the state). This is the stand-in for the paper's MLLM semantic-alignment read. The default `describe` is `tensor_token_description`, a deterministic hash of the state's value distribution onto a fixed vocabulary — enough to exercise the plumbing offline, **not** a semantic signal. For real use, pass a decoder:

```python
from nnsight.modeling import token_overlap_drift
from nnsight import VisionLanguageModel

vlm = VisionLanguageModel("llava-hf/llava-interleave-qwen-0.5b-hf", device_map="auto")

def caption_preview(latent):
    # Decode the latent to a preview frame with the pipeline's own VAE,
    # then ask a VLM what it shows — the paper's preview-frame supervisor.
    preview = sd._model.pipeline.vae.decode(latent / sd._model.pipeline.vae.config.scaling_factor)
    frame = preview.sample.clamp(-1, 1).add(1).div(2)
    with vlm.trace(image=PIL.Image.fromarray(frame.squeeze(0).permute(1, 2, 0).cpu().numpy())):
        return vlm.output.save()

corrector = TrajectoryCorrector(
    scorer=token_overlap_drift("A red panda eating bamboo", describe=caption_preview),
    total_steps=STEPS,
)
```

Anything that returns a `DriftReport` works — a CLIP similarity scorer, a learned probe, a hand-written heuristic.

## Reading the history

`corrector.history` holds one `DriftReport` per assessed step. After the trace exits, `correction_summary` turns it into plain numbers that survive outside the trace:

```python
from nnsight.modeling import correction_summary

summary = correction_summary(corrector)
# {'steps': 50, 'corrected_steps': 37, 'mean_drift': 0.43,
#  'max_drift': 0.91, 'total_correction_strength': 8.2}
```

The drift curve over steps is often the more interesting research output than the corrected image — it localizes *when* the trajectory diverges.

## Variations

### Correct only a window of steps

`assess`/`correct` are plain calls, so gate them in Python:

```python
with sd.generate(prompt, num_inference_steps=50) as tracer:
    for step in tracer.iter[:]:
        latent = sd.unet.output[0]
        if 10 <= step < 30:
            corrector.assess(latent, step)
            sd.unet.output[0][:] = corrector.correct(latent, step)
```

### Treat it as a diagnostic only

Never call `correct` — you get a full per-step drift trace with no intervention at all. Useful for comparing drift curves across prompts, seeds, or guidance scales.

### Anchor to a target run

`cosine_drift`'s reference doesn't have to be the same trajectory's step 0. Record a latent from a *good* run and correct a failing run toward it — the trajectory-level analog of [activation patching](activation-patching.md).

## Interpretation tips

- **A drift score is only as meaningful as its supervisor.** `cosine_drift` measures geometry, not semantics. Confirm the corrected images actually changed in the intended direction before trusting any conclusion drawn from the scores.
- **Sweep `gain` before concluding "no effect".** Too small and the correction is invisible; too large and the norm cap silently eats it. Check `total_correction_strength` in the summary to tell an ineffective gain from a clamped one.
- **Watch for the cap.** At `max_norm=1.0` an aggressive correction saturates at a reflection of the input state. Lower `gain`, or raise `max_norm` deliberately.
- **Corrected ≠ better.** The paper reports gains from a *semantic* supervisor; a geometric proxy pulls toward the anchor, which can suppress legitimate progression toward the final image. Use the two-pass comparison, not a single run.
- **Compare against an always-on, non-decayed baseline** (`decay` disabled via `total_steps` much larger than the real step count) to see what the decay is buying you.

## What's substituted

This page adapts the paper's loop structure with these substitutions (Mode 2 — auxiliary components swapped for target-native equivalents):

| Paper | Here |
|---|---|
| Semantic Assessment Supervisor (MLLM judging preview frames) | Pluggable `scorer` callable; two parameter-free proxies ship (`cosine_drift`, `token_overlap_drift`) |
| MLLM captioner inside the loop | Optional — bring your own VLM via `describe=`; not bundled, to keep the capability dependency-free |
| Semantic Modification Assistant (learned latent trajectory intervention) | Fixed direction toward an anchor, magnitude from `drift` × step-decayed gain, hard norm cap |
| Video benchmarks / evaluation suite | Out of scope — `correction_summary` for logging only |

Kept at full fidelity: per-step assess-then-correct inside the sampling loop, step-decayed intervention strength, bounded parameter-free corrections, no weight updates.

## Gotchas

- **One instance per trajectory.** `anchor` and `history` live on the corrector. Call `reset()` between runs or construct a fresh one.
- **`assess` before `correct` at each step.** `correct` uses the *most recent* report; calling it before any `assess` raises `RuntimeError`.
- **`correct` returns the state object unchanged when no correction is due** — assigning it back is free, but don't mutate the returned tensor expecting a copy.
- **Flatten-shape agreement.** The anchor and the state must have the same element count. A UNet tuple output and its `output[0]` differ; pick one and be consistent.
- **CFG doubles the batch.** Under `guidance_scale > 1` the denoiser's batch has both conditional and unconditional rows; the anchor must come from the same configuration or the shapes won't line up. See [diffusion-model](../models/diffusion-model.md).
- **Iter loop + trailing module access.** Code after `for step in tracer.iter[:]:` runs, but a regular module's `.output` access there raises `OutOfOrderError`. Save what you need inside the loop, or use a bounded `iter[:N]`. See [iteration gotchas](../gotchas/iteration.md).

## Related

- [steering](steering.md) — one-shot direction addition; the single-step cousin of this pattern.
- [activation-patching](activation-patching.md) — cross-run state replacement rather than in-loop correction.
- [diffusion-model](../models/diffusion-model.md) — the model class this loop usually runs on.
- `docs/usage/iter-all-next.md` — the iteration API the loop is built on.
- MLLM-Guided Semantic Correction for Text-to-Video Generation ([arXiv:2608.16513](https://arxiv.org/abs/2608.16513)).
