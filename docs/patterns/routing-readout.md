---
title: MoE Routing Readout
one_liner: Turn MoE router logits into a per-token interpretability signal — gate entropy, top-k margin, mass, and expert spread — without leaving the trace.
tags: [pattern, interpretability, moe, routing, readout]
related: [docs/usage/trace.md, docs/usage/access-and-modify.md, docs/usage/save.md, docs/patterns/steering.md]
sources: [src/nnsight/modeling/moe/routing_readout.py, docs/models/vllm.md]
---

# MoE Routing Readout

## What this is for

A mixture-of-experts model computes a routing decision at every MoE layer, for every token: a softmax over experts that picks which `top_k` of them will process that token. Those router logits are free — the model produces them whether you look or not — and they carry real signal about what the model is doing. "Beyond the Trace" (arXiv:2608.17638) shows that a ridge regression on exactly these native statistics reconstructs most of a learned 64-axis reasoning-state readout (their R64), well enough to drive test-time decisions like candidate selection and stop-and-resample.

This pattern reads the router logits through nnsight and reduces them to four scalars per layer per token. The scalars are parameter-free, so there is nothing to train before you can look at them; if you later want a learned readout on top, these are the features you regress against.

Adapted from that paper: the statistics layer is ported faithfully, while the learned J-space and its ridge weights are model-specific artifacts and out of scope here.

## When to use / when not to use

- Use when you want a cheap per-token signal out of an MoE model — logging "is the router committing or hedging?" over a generation.
- Use as feature columns for any downstream readout you fit yourself (effort vs. strain classifiers, stall detectors, candidate scoring).
- Use with either execution path: the vLLM router (`model.model.layers[i].mlp.gate`) or a HuggingFace MoE model's `mlp.gate`.
- Not useful on dense models — there is no router to read. Use the residual stream or attention patterns instead.

## Canonical pattern

```python
from nnsight.modeling.vllm import VLLM
from nnsight.modeling import RoutingReadout

model = VLLM("openai/gpt-oss-20b", dispatch=True)
LAYERS = (8, 16, 24, 32)

readout = RoutingReadout(top_k=8)   # gpt-oss routes to 8 experts

with model.trace("Solve: 17 * 24 =", temperature=0.0, max_tokens=64) as tracer:
    blocks = readout.attach(
        [model.model.layers[i].mlp.gate.output for i in LAYERS]
    ).save()
    steps = list().save()
    for _ in tracer.iter[:]:
        steps.append([b[-1] for b in blocks])   # last token per step

# steps[t][l] -> [entropy, margin, topk_mass, spread] for layer l at step t
```

Each layer contributes a `[tokens, 4]` block; `readout.vector` flattens the token-means into one `[n_layers * 4]` feature vector.

## The four statistics

| Statistic | What it reads |
|---|---|
| `entropy` | Entropy of the full routing distribution. Low = the router is committing to a narrow expert set. |
| `margin` | Gap between the strongest and the k-th strongest logit. Wide = confident routing with no near-ties. |
| `topk_mass` | Probability mass on the `top_k` experts the router will actually use. |
| `spread` | How many active experts share that mass meaningfully. `1.0` = one expert dominates; `k` = even split. |

The paper separates two things the emitted trace does not: inference effort from problem-induced strain. In practice the discriminator is that strain tends to show up as routing that is simultaneously *confident and unstable* — low entropy with a collapsing spread — across many layers at once, where plain effort looks like high margin with a healthy spread. Which combination matters is model- and task-specific; log them and look.

## Variations

### HF MoE models

The same module works on the `LanguageModel` path — `mlp.gate` holds the router logits there too:

```python
from nnsight import LanguageModel
from nnsight.modeling import RoutingReadout

model = LanguageModel("Qwen/Qwen1.5-MoE-A2.7B", device_map="auto", dispatch=True)
readout = RoutingReadout(top_k=4)   # check config.num_experts_per_tok

with model.trace(prompt):
    blocks = readout.attach(
        [model.model.layers[i].mlp.gate.output for i in (8, 16)]
    ).save()
```

### Computing your own statistics

`routing_features(logits, top_k)` is exported on its own if you want a different reduction — it returns the named statistics plus the full `probs` tensor:

```python
from nnsight.modeling import routing_features

with model.trace(prompt):
    feats = routing_features(model.model.layers[16].mlp.gate.output, top_k=8)
    top_expert = feats["probs"].argmax(dim=-1).save()
```

### Correlating the readout with behavior

The paper's headline use: score completed candidates and pick with the readout rather than by majority vote. The shape is two invokes per candidate, then argmax over a score you define on the vectors:

```python
with model.trace() as tracer:
    vectors = list().save()
    for prompt in candidates:
        with tracer.invoke(prompt):
            readout.attach([model.model.layers[i].mlp.gate.output for i in LAYERS])
            vectors.append(readout.vector)

# fit a direction on labeled data, then score: vectors[i] @ direction
```

### Router edits

Because the router is a plain `ReplicatedLinear` (full on every rank, no gathering needed — see [docs/models/vllm.md](../models/vllm.md)), the readout and the intervention sit on the same surface. Masking an expert's logit to `-inf` ablates it:

```python
with model.trace(prompt):
    gate = model.model.layers[16].mlp.gate
    mask = torch.full_like(gate.output, float("-inf"))
    mask[..., 3] = 0.0          # keep every expert except #3
    gate.output = gate.output + mask
```

The paper's closing result is exactly this loop: read the state, name a mechanism, edit the router, watch the predicted behavior appear.

## Interpretation tips

- **Pass layers in forward-pass order.** Within one invoke, module access order is the constraint it always is in nnsight — access the gates in the order the layers fire. To read a layer again after a later one, use a second invoke.
- **Pick `top_k` from the config.** `config.num_experts_per_tok` (HF) or the model card (vLLM). It is clamped to `n_experts` internally, so an over-large value degrades gracefully rather than raising.
- **`spread` needs context.** On its own it says little; it is most informative next to `entropy` and `margin` for the same token.
- **Pool over a window before deciding anything.** The paper aggregates the readout over a window, not a single token — per-token routing is noisy. `readout.means()` / `.vector` do the token-mean for you.
- **Compare against a baseline run.** The useful claim is almost always relative: this prompt vs. that one, this candidate vs. its siblings, before an edit vs. after.

## Gotchas

- `top_k` is a property of the *model*, not a free parameter — setting it to something the router doesn't do makes `topk_mass` and `spread` meaningless (entropy and margin stay fine).
- The readout computes on the fly inside the trace; `attach` does not copy. Clone a block if you need it unchanged after mutating the same gate's output.
- On vLLM with `tensor_parallel_size > 1`, `mlp.gate.output` is replicated and identical on every rank — no barrier or gathering needed for the readout itself.
- Under EP, individual experts are not addressable as submodules; if you want to know *which* experts fired, read `probs` from `routing_features` — the router distribution is the full picture.

## Related

- [docs/models/vllm.md](../models/vllm.md) — the MoE / expert-parallelism section, and why the router needs no gathering.
- [docs/usage/iter-all-next.md](../usage/iter-all-next.md) — `tracer.iter` for per-step readouts during generation.
- [docs/patterns/steering.md](steering.md) — the same write surface used for intervention rather than readout.
- "Beyond the Trace: Coupling an Interpretable Reasoning-State Readout to Native MoE Routing" (arXiv:2608.17638).
