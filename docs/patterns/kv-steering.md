---
title: KV-Cache Steering
one_liner: Add a direction to attention k/v projections once at prefill and let every later decode step attend over the steered cache.
tags: [pattern, interpretability, steering, kv-cache, generation]
related: [docs/patterns/steering.md, docs/usage/generate.md, docs/usage/iter-all-next.md, docs/patterns/per-head-attention.md]
sources: [src/nnsight/modeling/kv_steering.py, src/nnsight/modeling/language.py:181, src/nnsight/intervention/tracing/iterator.py]
---

# KV-Cache Steering

## What this is for

Residual-stream steering (see [steering](steering.md)) re-applies its addition on *every* generation step, because the residual stream is recomputed from scratch each step. The KV cache is different: keys and values computed during prefill are **reused verbatim by every subsequent decode step**. Add a direction to a layer's `k_proj`/`v_proj` output once, during prefill, and the intervention persists through the whole generation with zero per-step cost.

This is the mechanism of *cache steering* (arXiv 2507.08799, "KV Cache Steering for Inducing Reasoning in Small Language Models"), where a direction built from contrasting reasoning traces vs. direct answers is applied to the cache to push small models toward explicit chain-of-thought — no fine-tuning, no prompt changes.

nnsight ships this as `KVSteering` (`src/nnsight/modeling/kv_steering.py`), which resolves the right projection module for your architecture and scopes the write to generation step 0.

## When to use

- Steering long generations without paying an intervention per step.
- Comparing cache steering head-to-head with residual-stream steering at the same layer.
- Building reasoning-inducing / refusal / persona interventions on the attention cache.
- Testing whether a behavior is linearly represented in the key or value space of a layer.

## Canonical pattern

```python
from nnsight import LanguageModel
from nnsight.modeling.kv_steering import KVSteering

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
steering = KVSteering(model, layer=6)

# Contrast sets: prompts that exhibit the target behavior vs. ones that don't.
# (The paper uses GPT-4o-generated step-by-step traces vs. direct answers.)
direction = steering.direction(
    positive=["Let me think step by step. 2 plus 2 is 4.",
              "To solve this, I will break it down step by step."],
    negative=["The answer is 4.",
              "The answer is obvious."],
)

prompt = "Q: What is 3 + 5? A:"
with model.generate(prompt, max_new_tokens=20) as tracer:
    steering.apply(tracer, direction, coef=1.5)
    output = tracer.result.save()

print(model.tokenizer.decode(output[0]))
```

`direction(...)` runs both prompt sets through a single trace and returns the unit-norm difference of mean projection outputs. `apply(...)` enters `tracer.iter[0]` — prefill — and adds `direction * coef` to the last position's key/value projection output; everything the model generates afterward attends over the steered cache.

## How it works

- **Projection resolution.** Llama-style attention exposes `k_proj` / `v_proj` submodules and is steered directly. GPT-2-style attention uses a fused `c_attn` whose output concatenates `[q | k | v]`; `KVSteering` slices out the k or v chunk and steers in place, so the write propagates to the model's tensor.
- **One-shot semantics.** The loop `for _ in tracer.iter[0]:` scopes the body to generation step 0 (prefill). During decode steps, the cache entries written at prefill are reused — the addition never fires again.
- **Position choice.** `apply` steers the **last prefill position** — the KV entry that generated tokens attend to most heavily. To steer every prefilled position instead, write `out[:, :, :]` (see [Variations](#variations)).

## Variations

### Steering keys vs values

```python
k_direction = steering.direction(POSITIVE, NEGATIVE, proj="k_proj")
v_direction = steering.direction(POSITIVE, NEGATIVE, proj="v_proj")
```

Keys control *what* is attended to; values control *what is read out*. The paper finds value-side steering generally more effective for reasoning induction — but check both on your model.

### Steering all prefill positions

```python
with model.generate(prompt, max_new_tokens=20) as tracer:
    module, idx = steering._projection("v_proj")
    out = steering._proj_view(module.output, idx)
    with torch.no_grad():
        for _ in tracer.iter[0]:
            out[:] += v_direction * 1.5
    output = tracer.result.save()
```

Stronger and more disruptive than last-position steering, the same trade-off as in residual-stream steering.

### Persisting the direction computation

`steering.direction(..., cache=True)` computes the direction inside a `model.edit()` context, so the collection edits persist as model defaults instead of running once.

## Interpretation tips

- **Coefficient scales differ from residual steering.** Key/value projections have their own (typically larger) norms than the residual stream — a useful `coef` here can be orders of magnitude larger than the 1–10 typical for residual additions. Sweep broadly.
- **Greedy decoding can mask the effect.** On confident predictions a moderate steer changes logits without flipping the argmax. Inspect logit differences or sample, rather than only comparing token IDs.
- **Sweep layer and coef together.** As with residual steering, there is a band where behavior shifts and fluency holds; outside it, output degrades (repeated tokens, gibberish).
- **Compare against a random direction** of the same norm at the same layer — a real concept direction should beat it.

## Gotchas

- `apply` must be called **inside** the `with model.generate(...)` body; outside a generation context there is no prefill step to scope to.
- `proj` passed to `apply` must match the projection the direction was computed against — k-directions and v-directions live in different spaces.
- GPT-2's fused `c_attn` output is 3× the hidden size; the helper slices it for you, but if you write raw code against it, remember which third is which.
- In-place writes propagate because the sliced view aliases the model's tensor — don't `.clone()` or `.contiguous()` the view before writing.
- Pass `max_new_tokens`; without it the generation has no bounded step count (see [docs/usage/iter-all-next.md](../usage/iter-all-next.md)).

## Related

- [steering](steering.md) — the residual-stream counterpart; same direction computation, different target and per-step cost.
- [per-head-attention](per-head-attention.md) — slicing attention outputs per head.
- [docs/usage/generate.md](../usage/generate.md) — generation under a tracing context.
- Wu et al. (2025), "KV Cache Steering for Inducing Reasoning in Small Language Models" (arXiv 2507.08799).
