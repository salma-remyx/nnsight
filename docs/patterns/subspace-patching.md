---
title: Subspace Activation Patching
one_liner: Patch only the component of an activation inside a hypothesized low-dimensional subspace - and run the null control that tells you whether the result actually means anything.
tags: [pattern, interpretability, causal-mediation, patching, subspace]
related: [docs/patterns/activation-patching.md, docs/patterns/steering.md, docs/usage/barrier.md, docs/patterns/attribution-patching.md]
sources: [src/nnsight/modeling/subspace.py, docs/usage/barrier.md]
---

# Subspace Activation Patching

## What this is for

Standard [activation patching](activation-patching.md) replaces an entire activation vector. **Subspace** activation patching replaces only the component that lies inside a hypothesized low-dimensional subspace - a direction from a probe, a steering direction, an SAE feature's decoder row, a rank-1 edit direction - and leaves the orthogonal complement of the running activation alone.

The payoff is supposed to be attribution: if patching *just* subspace `U` flips behavior, the story goes, the feature lives in `U`. Makelov, Lange & Nanda (2023) show this inference is unsafe. A subspace patch can flip behavior through a **dormant parallel pathway** - the model reads the feature somewhere else entirely - so a successful patch does not certify that `U` is causally load-bearing. The paper ties this to why rank-1 fact editing can succeed behaviorally while the edited direction is not where the fact is stored.

The fix is a **null control**: also patch a random subspace of the *same rank*. A candidate subspace is only interesting if it beats the control, not merely if the patch works. nnsight ships the projection helpers in `nnsight.modeling.subspace`.

## When to use

- You have a hypothesized feature subspace (probe direction, contrast-set mean difference, SAE feature, rank-1 edit) and want to know if it carries a behavior.
- You want a *targeted* version of activation patching that does not overwrite everything upstream of a layer.
- You are about to claim "this direction is where the model stores X" - run the control first.
- Evaluating rank-1 / rank-k edits and whether the edited directions are the storage location or just one pathway.

## Canonical pattern

Patch the clean run's rank-1 subspace component into the corrupt run, with the control in the same trace. Patching *only* the subspace means:

```
patched = corrupt + P_U(clean) - P_U(corrupt)
```

```python
import torch
from nnsight import LanguageModel
from nnsight.modeling import orthonormalize, random_basis, subspace_patch

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

clean   = "The Eiffel Tower is in the city of"   # next token: " Paris"
corrupt = "The Colosseum is in the city of"      # next token: " Rome"
LAYER = 6

paris = model.tokenizer.encode(" Paris")[0]

# --- Step 1: build the candidate basis from a clean/corrupt activation pair.
with model.trace() as tracer:
    with tracer.invoke(clean):
        clean_hs = model.transformer.h[LAYER].output[:, -1, :].save()
    with tracer.invoke(corrupt):
        corrupt_hs = model.transformer.h[LAYER].output[:, -1, :].save()

direction = clean_hs[0] - corrupt_hs[0]
candidate = orthonormalize(direction.unsqueeze(1))          # [768, 1]
control   = random_basis(768, rank=1, generator=torch.Generator().manual_seed(0))

# --- Step 2: patch each basis into a fresh corrupt run.
def patch_with(basis):
    with model.trace() as tracer:
        barrier = tracer.barrier(2)                          # 2 participating invokes
        with tracer.invoke(clean):
            source = model.transformer.h[LAYER].output[:, -1, :]
            barrier()                                        # source is ready
        with tracer.invoke(corrupt):
            barrier()                                        # wait for source
            running = model.transformer.h[LAYER].output[:, -1, :]
            running += subspace_patch(running, source, basis)
            patched = model.lm_head.output[:, -1, :].save()
        with tracer.invoke(corrupt):                         # no barrier: independent
            baseline = model.lm_head.output[:, -1, :].save()
    return patched.softmax(-1)[0, paris].item(), baseline.softmax(-1)[0, paris].item()

cand_p, base_p = patch_with(candidate)
ctrl_p, _      = patch_with(control)

print(f"baseline P(Paris) = {base_p:.4f}")
print(f"candidate P(Paris) = {cand_p:.4f}   (rank-1 subspace patch)")
print(f"control   P(Paris) = {ctrl_p:.4f}   (random same-rank subspace)")
```

On GPT-2 at layer 6 this prints roughly `baseline 0.0033 / candidate 0.0056 / control 0.0033`. The candidate raises P(Paris) ~70% relative; the random control leaves it untouched. That contrast is the signal.

**Read it as:** the aligned direction carries *some* of the city information. It is not yet evidence that layer 6 is where the model reads it from - see "The illusion" below.

## The illusion: why the control is not enough

The control rules out "any same-rank perturbation does this". It does **not** rule out the paper's failure mode: the behavioral effect flowing through a **dormant parallel pathway** rather than through `U` itself. The model can store the feature in subspace `V` (causally load-bearing, never patched) while your patch into `U` happens to recruit a pathway that drives the same output.

Evidence that actually supports "the feature lives in `U`" goes beyond patching:

- **Block the rest.** Patch `U` *and* freeze / zero the rest of the activation; if behavior still flips, the pathway really is through `U`.
- **Patch the complement.** Patch everything *except* `U`. If that also flips behavior, `U` was not necessary.
- **Independent circuit evidence.** Manual circuit analysis (attention patterns, path patching) that separately localizes the feature - the paper's IOI success case does exactly this.
- **Consistency with editing.** If rank-1 editing in `U` changes behavior but localization keeps pointing elsewhere, suspect the illusion - the paper shows this exact dissociation in factual recall.

Treat subspace patching as a *screening* tool: cheap, targeted, and only meaningful with the same-rank control attached.

## Variations

### Higher-rank subspaces

Any `[d, k]` basis works; the helpers orthonormalize internally.

```python
# Top-k left singular vectors of a contrast-set activation matrix.
basis = torch.linalg.svd(activations_matrix).U[:, :4]       # [768, 4]
control = random_basis(768, rank=4, generator=torch.Generator().manual_seed(0))
```

Use `random_basis(dim, rank=k)` with the same `k` for the control - the control must match the candidate's rank to be a fair null.

### Projection without patching

To inspect how much of an activation lies in a subspace (e.g. measuring feature strength per token):

```python
from nnsight.modeling import project_component

with model.trace(prompt):
    hs = model.transformer.h[LAYER].output.save()
    comp = project_component(hs, basis).save()

strength = comp.norm(dim=-1)   # [batch, seq] - how "in-subspace" each token is
```

### Subspace steering

Adding (rather than transferring) a subspace component is a targeted form of [steering](steering.md):

```python
with model.trace(prompt):
    running = model.transformer.h[LAYER].output[:, -1, :]
    running += project_component(coef * direction_vector, basis)
    out = model.lm_head.output[:, -1, :].save()
```

### Whole space reduces to plain patching

With `basis = torch.eye(d)`, `subspace_patch` returns `source - activation` and the pattern collapses to standard activation patching - a useful sanity check that your basis handling is right.

## Interpretation tips

- **Always report the control number.** "Subspace patch flips the answer" is uninterpretable on its own; "flips it and the random same-rank control does not" is the minimum publishable unit here.
- **Compare against full-vector patching** at the same layer. If the full patch flips behavior and the subspace patch does not, the subspace is missing load-bearing components.
- **Sweep rank.** If behavior only appears at high rank, your "low-dimensional feature" may be an artifact of the rank you chose.
- **Relative and absolute changes both matter.** A rank-1 patch moving a probability from 0.003 to 0.006 is a 2x relative change on a small base - check the top-k tokens, not just one logit.
- **Layer and position still apply** exactly as in [activation patching](activation-patching.md): last position tests sensitivity, subject tokens test information flow.

## Gotchas

- **The barrier count is 2, not 3.** Only the two invokes that exchange `source` call `barrier()`. The baseline invoke is independent and must not join the barrier, or the trace deadlocks.
- Inside one invoke, access modules in forward-pass order; capture `source` before you read the patched module's output in the corrupt invoke (that is what the barrier orders).
- `running += subspace_patch(...)` mutates in place. `.clone()` the running activation first if you need the pre-patch value.
- `orthonormalize` drops near-zero columns from rank-deficient input, so a candidate built from collinear vectors may come back with smaller rank than you asked - check `.shape`.
- `subspace_patch(activation, source, basis)` is ordered (destination, source); swapping the arguments silently patches in the wrong direction.
- For Llama-family models the residual lives at `model.model.layers[L].output`, not `model.transformer.h[L].output`. `print(model)` to confirm.
- `block.output[0]` is the residual in `transformers<5.0`; in `>=5.0` `block.output` is already the tensor.

## Related

- [activation-patching](activation-patching.md) - the full-vector version this generalizes.
- [steering](steering.md) - adding directions rather than transferring components; the refusal-direction variant already uses projection.
- [attribution-patching](attribution-patching.md) - linear approximation for sweeping many sites cheaply.
- [multi-prompt-comparison](multi-prompt-comparison.md) and `docs/usage/barrier.md` - the invoke/barrier mechanics used above.
- Makelov, Lange & Nanda (2023), "Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching" ([arXiv:2311.17030](https://arxiv.org/abs/2311.17030)).
