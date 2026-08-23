---
title: Test-Time Latent Optimization
one_liner: Gradient-descend a mid-layer latent on continuation log-probs with weights frozen, then re-inject it during generation.
tags: [pattern, interpretability, gradients, test-time, optimization]
related: [docs/usage/backward-and-grad.md, docs/patterns/gradient-based-attribution.md, docs/patterns/steering.md, docs/usage/generate.md]
sources: [src/nnsight/modeling/latent_optimization.py]
---

# Test-Time Latent Optimization

## What this is for

Optimization-based latent reasoning adapts a model to a *single input* at test time: freeze the weights, insert an optimizable latent state into one block's residual stream, and descend the log-probability of a target continuation. Because causal self-attention gives every continuation-token log-prob a differentiable path back through the remaining blocks, gradients from the whole continuation are credited directly to the latent — no decoded-token feedback loop.

This is the test-time counterpart to [steering](steering.md): instead of adding a *precomputed* direction, you *optimize* one per instance. Use it to study how far a single low-rank additive nudge at one layer can move a model's output, which layers make the best optimization space, and which continuation tokens the latent actually influences.

`nnsight.modeling.latent_optimization.LatentOptimizer` packages the loop.

## When to use

- Test-time adaptation of a frozen model toward a target continuation.
- Measuring which layers are the most effective optimization space (sweep `layer=`).
- Token-level credit assignment: which continuation tokens does the latent influence?
- Building a stronger steering vector than a mean-difference contrast set.

## Canonical pattern

```python
import torch
from nnsight import LanguageModel
from nnsight.modeling.latent_optimization import LatentOptimizer

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

opt = LatentOptimizer(
    model,
    "The Eiffel Tower is in the city of",
    " Paris, the capital of France.",
    layer=6,          # default: middle block
    steps=10,         # paper uses <= 10
    lr=1e-3,          # paper uses 1e-3
)

with model.session():
    for _ in range(opt.steps):
        with model.trace(opt.text):
            opt.step(opt.score())

result = opt.result()
print(result.losses)      # descending; weights stayed frozen
print(result.grad_norms)

# Re-inject the optimized latent during generation.
with model.generate(opt.prompt, max_new_tokens=8, do_sample=False) as tracer:
    opt.apply()
    tokens = tracer.result.save()
print(model.tokenizer.decode(tokens[0][opt.n_prompt:]))
```

Each pass: `score()` adds `delta` over the prompt span at `blocks[layer]` and returns the reward-weighted continuation NLL; `step()` backprops it and applies a plain-Python Adam update to the latent alone.

## Variations

### Which tokens does the latent influence?

```python
out = []
with model.trace(opt.text):
    out[:] = opt.attribute()

tokens, scores = out
for tok, s in zip(tokens, scores.tolist()):
    print(f"{tok!r:>12}  {s:.4f}")
```

A token's score is the L2 norm of the gradient of its log-probability with respect to the latent. Influence tends to concentrate on the content words of the continuation rather than punctuation or connectives.

### Sweep the layer

```python
for layer in range(len(opt.blocks)):
    opt = LatentOptimizer(model, prompt, continuation, layer=layer, steps=10)
    with model.session():
        for _ in range(opt.steps):
            with model.trace(opt.text):
                opt.step(opt.score())
    print(layer, opt.losses[0] - opt.losses[-1])   # improvement per layer
```

Early-to-middle layers are typically the most effective optimization space; late layers have too few remaining blocks to spread the credit.

### Sweep the strength at application time

```python
with model.generate(opt.prompt, max_new_tokens=8, do_sample=False) as tracer:
    opt.apply(coefficient=0.5)
    tokens = tracer.result.save()
```

### Reward weighting

Pass a callable mapping the continuation's per-token log-probabilities `[batch, n_cont]` to non-negative weights. Down-weight low-confidence tokens, up-weight the final answer span, or plug in an external verifier's per-token scores:

```python
def confidence_weighted(logprobs):
    return torch.sigmoid(logprobs)   # soft: near-1 for confident tokens

opt = LatentOptimizer(model, prompt, continuation, reward=confidence_weighted)
```

## Interpretation tips

- **Check `losses` descends.** If it doesn't, the latent is too small relative to the residual stream or `layer` is too late. Watch `grad_norms` — it should be nonzero from step 0.
- **Compare against a zero latent.** `apply()` before any `step()` reproduces unmodified generation, giving you a free baseline in the same session.
- **`delta` norms are small.** The residual stream at a mid layer has a large norm; a latent that meaningfully steers output may still look tiny next to it. Judge by output change, not raw norm.
- **Token attribution is the interpretability payoff.** It tells you *where* in the continuation the latent spends its influence — the same diagnostic the paper uses to show influence concentrating on reasoning connectors.

## Gotchas

- **The `with` statements must be written in your code, not inside a helper.** nnsight captures trace bodies from the caller's frame; a library function that opens `model.trace()` itself falls back to running the model untraced. `LatentOptimizer`'s methods are designed to be called *inside* your `with` blocks for exactly this reason.
- **Gradient reads use plain `loss.backward()`.** `delta` is a leaf tensor, so `delta.grad` comes straight off autograd. A `with loss.backward():` context can only be captured from a `with` block in the user's frame.
- **Tuple assignment doesn't leave a trace body.** `tokens, scores = opt.attribute()` inside a `with` won't propagate out — capture into a pre-existing container (`out[:] = opt.attribute()`).
- **`model.generate()` runs under `no_grad`.** Optimize with `model.trace()` (teacher-forced) and only *apply* the latent in `generate()`.
- **Layer access is forward-order within one invoke.** `score()` and `apply()` each touch exactly one block plus `lm_head`, in execution order — don't interleave other block accesses between them inside the same invoke.

## Related

- [gradient-based-attribution](gradient-based-attribution.md) — the same `.grad` machinery for explaining an existing prediction rather than optimizing an intervention.
- [steering](steering.md) — additive interventions with a *precomputed* direction; a good baseline to beat.
- [docs/usage/backward-and-grad.md](../usage/backward-and-grad.md) — the full backward-context reference.
