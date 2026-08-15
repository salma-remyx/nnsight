---
title: Directional Erasure of Merge Interference
one_liner: Erase a causally load-bearing direction from the residual stream and check whether norm-matched wrong-direction controls fail.
tags: [pattern, interpretability, model-merging, residual-stream, causal-intervention]
related: [docs/patterns/steering.md, docs/patterns/ablation.md, docs/patterns/multi-prompt-comparison.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/modeling/merge_interference.py]
---

# Directional Erasure of Merge Interference

## What this is for

When a model is merged with task vectors (task arithmetic) and behavior
degrades, the field usually blames *magnitude*: layerwise representation
bias, cross-task non-linearity, parameter overlap. The causal-structure
view (adapted from arXiv:2608.11797, "Orientation, not magnitude") says
the interference is carried by a *direction* in the residual stream —
erasing that direction removes expressed interference dose-dependently
and saturates at exact erasure, while erasing a **norm-matched
wrong direction** fails or backfires.

The test is a two-arm intervention:

- **Causal arm** — project the residual stream off the carried direction.
- **Control arm** — apply a norm-matched projection along an unrelated
  direction (here: a per-position permutation of the hidden dims).

If only the causal arm moves behavior, the direction — not the
magnitude of the edit — is load-bearing.

## When to use

- Diagnosing why a merged model interferes on a task.
- Checking whether a residual-stream displacement is causally carried
  or merely correlated.
- Building dose-response curves for an intervention (partial
  coefficients in `[0, 1]`, over-projection `> 1`).

## Canonical pattern

```python
import torch
from nnsight import LanguageModel
from nnsight.modeling import (
    directions_from_contrast,
    erase_from_resid,
    interference_dose,
    shuffle_columns,
)

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

LAYER = 6
prompt = "The capital of France is"

with model.trace(prompt):
    base_l = model.transformer.h[LAYER].output[0].save()
    base_out = model.lm_head.output[:, -1, :].save()

# The "merged" state proxy: a second prompt whose trajectory you want to
# erase back toward the baseline. In a real merge study, load the merged
# weights and use the SAME prompt.
with model.trace(prompt + " (task)"):
    merged_l = model.transformer.h[LAYER].output[0].save()
    merged_out = model.lm_head.output[:, -1, :].save()

directions = directions_from_contrast(base_l, merged_l)

with model.trace(prompt + " (task)"):
    erase_from_resid(model.transformer.h[LAYER], directions)
    erased_out = model.lm_head.output[:, -1, :].save()

with model.trace(prompt + " (task)"):
    erase_from_resid(model.transformer.h[LAYER], shuffle_columns(directions))
    control_out = model.lm_head.output[:, -1, :].save()

dose = interference_dose(base_out, erased_out).max()
control = interference_dose(base_out, control_out).max()
print(f"erasure dose: {dose:.3f}  matched-control dose: {control:.3f}")
```

`erase_from_resid` handles the tuple-vs-tensor shape of block outputs;
`directions_from_contrast` unit-normalizes per position.

## Variations

**Dose-response.** Call `erase_from_resid(..., coefficient=c)` for
`c` in `[0.25, 0.5, 0.75, 1.0, 1.25]`. The paper's signature result is
dose-dependent removal that *saturates* at exact erasure (`c = 1`) —
beyond it you are over-projecting, which is where norm-matched controls
backfire.

**Layer sweep.** The paper finds early-layer erasure is undone by
propagation (rebuilt to ~99% of norm downstream) while late-layer
erasure sticks. Loop over layers and compare the dose measured at
`lm_head` — the recovery curve is the finding, not a single number.

```python
for layer in range(model.config.n_layer):
    d = directions[layer]  # per-layer directions from a contrast pass
    with model.trace(prompt):
        erase_from_resid(model.transformer.h[layer], d)
        out = model.lm_head.output[:, -1, :].save()
    print(layer, interference_dose(base_out, out).max().item())
```

## Interpretation tips

- **Orientation is the claim.** If the matched control removes as much
  interference as the true direction, you have found a magnitude effect,
  not a causal direction — report that honestly.
- **The contrast proxy vs. the paper's ledger.** The paper identifies the
  carried direction from an exact factorial cross-term ledger. Here
  `directions_from_contrast` substitutes the parameter-free mean
  displacement between two model states. For a real merged model, use
  base-vs-merged activations of the *same* prompt set.
- **Instruction wrappers gate the effect.** The paper reports the same
  erasure finds far less removable interference under instruction
  wrappers that pin a template main effect. Run both a raw and a
  wrapped prompt before concluding an erasure "doesn't work".

## Gotchas

- Access blocks in forward-pass order within an invoke; out-of-order
  access across invokes on the same module needs a `tracer.barrier(n)`
  (see [multi-prompt-comparison](multi-prompt-comparison.md)).
- `erase_from_resid` edits in place (`hidden[:] = ...`). If you also
  saved the pre-edit activation in the same invoke, `.clone()` it first —
  a saved variable is a reference (see
  [docs/gotchas/modification.md](../gotchas/modification.md)).
- Directions must be unit-norm. `directions_from_contrast` normalizes;
  if you build directions yourself, normalize along the last dim.
