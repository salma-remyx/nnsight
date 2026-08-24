---
title: Timestep Feature Reuse
one_liner: Cache a block's output every Nth iteration and predict it in between, bypassing the block's forward entirely.
tags: [pattern, efficiency, diffusion, caching, intervention]
related: [docs/usage/iter-all-next.md, docs/usage/cache.md, docs/usage/generate.md, docs/patterns/ablation.md]
sources: [src/nnsight/modeling/feature_reuse.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracing/tracer.py]
---

# Timestep Feature Reuse

## What this is for

Iterative models — diffusion denoisers above all — run the same blocks once per timestep.
Not every block's output changes much between neighboring timesteps, which is the opening
feature-caching methods exploit: compute a block on some timesteps, predict it on the rest,
and skip the forward entirely on the predicted ones.

nnsight exposes every primitive this needs. `tracer.iter[:]` steps through iterations,
`module.output` reads a computed feature, and `module.skip(replacement)` bypasses the
module's forward so a predicted feature flows onward in place of a computed one.
[`apply_feature_reuse`](#canonical-pattern) drives that loop for you.

Adapted from [LinCa: Accelerating Diffusion Models via Learnable Decomposed Feature
Caching](https://arxiv.org/abs/2608.17973). LinCa's contribution is a *learned invertible
decomposition* — cached features are split into sub-components with distinct continuity
properties, each predicted at a matched order, then reconstructed losslessly. The learned
part ships as trained weights for specific checkpoints (FLUX, Qwen-Image, HunyuanVideo) and
is not reproduced here. What this pattern keeps is the loop and the per-component ordering:
a parameter-free Lagrange predictor over the timestep axis, with a per-module interpolation
order standing in for LinCa's differentiated prediction orders.

## When to use

- Speeding up traced diffusion generation when you control the quality tradeoff.
- Measuring how *temporally continuous* a block's features are — a block that interpolates
  well at high stride is one whose features change slowly across timesteps, which is a
  finding in itself.
- Building an ablation-style baseline: compare output at a given stride against
  full computation to bound how much a block's per-timestep work matters.

Not for: models whose modules fire once per forward pass rather than once per iteration.
The reuse grid is defined over `tracer.iter` steps, so there is nothing to reuse without a
multi-step loop.

## Canonical pattern

```python
from nnsight import DiffusionModel
from nnsight.modeling import apply_feature_reuse

sd = DiffusionModel("segmind/tiny-sd", torch_dtype=torch.float16)

report = {}
with sd.generate("A photo of a cat", num_inference_steps=20) as tracer:
    report["plans"] = apply_feature_reuse(
        tracer,
        *sd.feature_reuse({"unet.up_blocks.2": (4, 3)}),
    )
    for _ in tracer.iter[:]:
        pass
    output = tracer.output.save()

print(report["plans"]["unet.up_blocks.2"].report("unet.up_blocks.2"))
# unet.up_blocks.2: computed 5 timesteps [0, 4, 12, 16], predicted 15 (75% reused, order=3)
```

`sd.feature_reuse(spec)` resolves the spec's dotted paths onto envoys and returns the
`(modules, plan)` pair; `apply_feature_reuse` consumes it. Spec values are:

| Value | Meaning |
|---|---|
| `4` | stride 4, default order 2 |
| `(4, 3)` | stride 4, order 3 |
| `FeatureReusePlan(stride=4, order=3, start=1, stop=15)` | full control over the grid |

**Stride** is how many timesteps pass between computations — stride 4 computes on
timesteps 0, 4, 8, ... and predicts the three in between, so the block runs a quarter as
often. **Order** is how many cached samples the predictor interpolates through: order 1
holds the last value, order 2 is linear, order 3 is quadratic. Higher order tracks smooth
feature drift more closely and amplifies noise more.

Modules not named in the spec are left on the compute path. That is the intended shape —
reuse is a per-block decision, and the gains come from caching a subset rather than
everything.

## Variations

### Different blocks at different strides

The point of a per-module order is that blocks differ. Deep blocks with smooth features
interpolate well at high stride and order; early blocks and attention-heavy blocks often
need to run more often:

```python
spec = {
    "unet.down_blocks.0": 1,                # run every step
    "unet.down_blocks.2": (2, 2),
    "unet.up_blocks.1": (3, 2),
    "unet.up_blocks.3": (4, 3),             # smooth, cache hard
    "unet.mid_block": (2, 3),
}
with sd.generate(prompt, num_inference_steps=30) as tracer:
    plans = apply_feature_reuse(tracer, *sd.feature_reuse(spec))
    for _ in tracer.iter[:]:
        pass
```

### Measuring temporal continuity instead of speeding things up

Run the same block at increasing stride and watch how far the output drifts. A block whose
output survives stride 8 has temporally continuous features; one that degrades by stride 2
does not. This is the diagnostic version of the pattern and needs no speedup claim:

```python
for stride in (1, 2, 4, 8):
    with sd.generate(prompt, num_inference_steps=16, seed=0) as tracer:
        plans = apply_feature_reuse(
            tracer, *sd.feature_reuse({"unet.mid_block": stride}), num_steps=16
        )
        reused = tracer.output.save()
    with sd.generate(prompt, num_inference_steps=16, seed=0) as tracer:
        exact = tracer.output.save()
    drift = (reused - exact).abs().mean()
    print(f"stride {stride}: mean drift {drift:.4f}")
```

### Reusing within a plain trace, not `generate`

`generate` bounds `tracer.iter[:]` from `num_inference_steps`. Inside a bare `trace` there
is no such bound, so pass `num_steps` explicitly or the loop will not terminate:

```python
with sd.trace(prompt, num_inference_steps=8) as tracer:
    plans = apply_feature_reuse(
        tracer, *sd.feature_reuse({"unet.mid_block": 2}), num_steps=8
    )
    for _ in tracer.iter[:8]:
        pass
```

### Reading the report

`apply_feature_reuse` returns one plan per module with the executed and predicted timestep
lists filled in. The return value only exists inside the trace body, so stash it in a
container declared outside if you want it afterward:

```python
report = {}
with sd.generate(prompt, num_inference_steps=20) as tracer:
    report["plans"] = apply_feature_reuse(tracer, *sd.feature_reuse(spec))
    for _ in tracer.iter[:]:
        pass

for path, plan in report["plans"].items():
    print(plan.report(path))       # "unet.mid_block: computed 10 timesteps ..., 50% reused"
    print(plan.reuse_ratio)        # 0.5
    print(plan.executed)           # [0, 2, 4, ...]
```

## Interpretation tips

- **`reuse_ratio` is an execution fraction, not a wall-clock speedup.** It counts
  timesteps, not FLOPs. Translate it to a real speedup by weighting each module's ratio by
  that module's share of the denoiser's compute.
- **A block that interpolates well is a finding.** High stride at low drift means the
  block's features are temporally continuous — its per-timestep work is close to redundant
  across neighboring steps. That is the observation feature-caching methods are built on,
  visible here without training anything.
- **Order and stride trade against each other.** Raising the order often lets you raise
  the stride; raising both at once on a noisy block is how you get artifacts.
- **Validate against a fixed seed.** Reuse changes the trajectory, so compare against an
  identically-seeded exact run — never two unseeded runs.

## Gotchas

- **The first timestep always computes.** There is nothing to predict from, so a plan that
  would skip step 0 computes instead.
- **Stride 1 disables reuse** for that module and reproduces an unmodified run exactly —
  useful as a sanity check that your wiring is correct.
- **Errors compound downstream.** A predicted feature feeds the next timestep's real
  computation. On residual-heavy stacks the drift grows with the number of reused steps in
  a row, which is what the stride controls.
- **Tuple-returning blocks are handled, exotic ones are not.** Outputs like
  `(hidden_states, None)` predict back as tuples; an output containing no tensors at all
  is left on the compute path silently rather than raising.
- **Duplicate or non-monotonic timesteps are tolerated.** If a scheduler attribute is
  misread and two samples land on the same timestep, the duplicate nodes are dropped
  before fitting instead of dividing by zero.

## Related

- [docs/usage/iter-all-next.md](../usage/iter-all-next.md) — the iteration API this is
  built on, and the pitfalls of unbounded `iter[:]`.
- [docs/usage/cache.md](../usage/cache.md) — `tracer.cache()` records activations without
  reusing them; the two compose.
- [docs/usage/generate.md](../usage/generate.md) — multi-step generation.
- [docs/patterns/ablation.md](ablation.md) — the same skip machinery used to measure
  contribution rather than save compute.
