---
title: MoE Routing Signals
one_liner: Read per-token router statistics - entropy, top-k margin, expert load - from Mixture-of-Experts routers as hallucination and routing-health signals.
tags: [pattern, interpretability, moe, routing, hallucination]
related: [docs/usage/access-and-modify.md, docs/usage/save.md, docs/usage/iter-all-next.md, docs/models/vllm.md]
sources: [src/nnsight/modeling/moe_routing.py]
---

# MoE Routing Signals

## What this is for

In a Mixture-of-Experts model every token passes through a router that scores all experts and picks a few. That routing decision is a per-token signal dense transformers simply do not have: it says how committed the model was at this position, and to which experts.

Reading it turns out to be informative about more than load balancing. ["Mixture-of-Expert Blocks Contain Strong Hallucination Detection Signals"](https://arxiv.org/abs/2608.17687) (InnerExpert) shows routing-level statistics — router entropy, top-k disagreement, expert usage — separate supported content from unsupported content at the *token* level, and that they beat dense-only internal signals for that purpose while needing a single forward pass.

This recipe extracts those signals with `nnsight.modeling.moe_routing`. It gives you the features; what you correlate them against (hallucination spans, dataset difficulty, routing collapse) is your experiment.

## When to use

- Localizing which tokens a MoE model was "unsure" at the routing level.
- Comparing routing behavior across prompts, layers, or expert-parallel layouts.
- Detecting router collapse (one expert absorbing most tokens).
- Building per-token features for a downstream probe or detector.

## Canonical pattern

```python
from nnsight import LanguageModel
from nnsight.modeling import find_routers, routing_features

model = LanguageModel("ibm/PowerMoE-3b", device_map="auto", dispatch=True)
routers = find_routers(model)          # one Envoy per MoE layer, in order

prompt = "The Eiffel Tower is located in the city of"

entropy = None
with model.trace(prompt):
    feats = routing_features(routers[0].output, layer=0)
    entropy = feats["router_entropy@0"].save()

print(entropy.shape)                   # one value per token
```

`find_routers` walks the envoy tree and matches on module *shape* — a `gate`/`router` module with a `[num_experts, hidden]` weight sitting next to an experts container — so the same call works whether the architecture calls it `mlp.gate` (Qwen3-MoE, Mixtral, vLLM) or `block_sparse_moe.router` (GraniteMoe).

## The signals

| Feature | Meaning | What to look for |
|---|---|---|
| `router_entropy` | Shannon entropy of the full softmax over experts | Low = router committed hard; the paper's headline routing signal |
| `topk_entropy` | Entropy of the normalized top-k routing weights | Near 0 = one expert dominates the selected set |
| `topk_margin` | Gap between the two strongest routing weights | Small = router could not decide ("expert disagreement") |
| `topk_weight_mass` | Probability mass the selected experts carry | 1.0 under `norm_topk_prob`, lower otherwise |
| `expert_load` | Tokens this step routed to this token's top expert | High = this token went to a crowded expert |
| `top1_expert` | Winning expert id per token | Grouping key when aggregating a corpus |

Collect several layers with `layer=` so the keys don't collide:

```python
collected = {}
with model.trace(prompt):
    for i, router in enumerate(routers):
        feats = routing_features(router.output, layer=i)
        collected[f"router_entropy@{i}"] = feats[f"router_entropy@{i}"].save()
```

## Per-step, during generation

For token-level analysis of a generation, iterate steps and collect one row each:

```python
rows = []
with model.generate(prompt, max_new_tokens=20) as tracer:
    for _ in tracer.iter[:]:
        feats = routing_features(routers[-1].output)
        rows.append(feats["router_entropy"].save())
    text = tracer.result.save()

# rows[0] covers the prompt; each later row is one generated token.
```

The first row spans the whole prompt; subsequent rows are shape `(1,)` — the single token being decoded. Aligning row `n` with `text[n]` gives you per-generated-token signals.

## Expert usage and routing health

`expert_usage` rebuilds a layer's per-expert token histogram from a features dict — a quick routing-collapse check:

```python
from nnsight.modeling import expert_usage

with model.trace(prompt):
    feats = routing_features(routers[0].output)
    top1 = feats["top1_expert"].save()

usage = expert_usage({"top1_expert": top1})
print(usage.argmax(), "of", len(usage), "experts took the most tokens")
```

A healthy router spreads load; a collapsed one funnels most tokens to a couple of experts (which also makes `expert_load` uniformly large and uninformative — worth checking before reading anything into it).

## Layouts, and why there's a normalizer

There is no single "router output" shape. `routing_features` accepts whatever the router returned and recovers the logits by shape, so you never have to remember which architecture returns what:

| Router | Returns |
|---|---|
| Qwen3-MoE / Mixtral `*TopKRouter` | `(logits, scores, indices)` |
| GraniteMoe `GraniteMoeTopKRouter` | `(indices, weights, logits)` |
| vLLM `mlp.gate` (ReplicatedLinear) | logits first, or bare |

Use `normalize_router_output(output)` directly when you want the raw `(logits, topk_weights, topk_indices)` triple — e.g. to mask a router logit for an expert ablation (see [ablation](ablation.md)).

## On vLLM

The same helpers work against `nnsight.modeling.vllm.VLLM`, where the router is `mlp.gate` and — being `ReplicatedLinear` — identical on every rank, so reading it under tensor or expert parallelism needs no gathering at all. See [docs/models/vllm.md](../models/vllm.md) for the MoE batching details.

```python
from nnsight.modeling.vllm import VLLM
from nnsight.modeling import find_routers, routing_features

model = VLLM("Qwen/Qwen1.5-MoE-A2.7B", tensor_parallel_size=2, dispatch=True)
routers = find_routers(model)
```

## Gotchas

- Save via plain assignment, not a comprehension — comprehension bodies run in their own frame, so their locals don't survive the trace. See [docs/gotchas/save.md](../gotchas/save.md).
- Access modules in forward order within an invoke; the `find_routers` order is module order, which is what you want. See [docs/gotchas/order-and-deadlocks.md](../gotchas/order-and-deadlocks.md).
- Routers are only present in MoE layers. Dense layers (common as the first and last blocks in otherwise-MoE models) simply don't appear in `find_routers` output.
- `top1_expert` is the one integer feature; cast before arithmetic if you mix it with the float columns.
- Features are computed on the router's own token axis, which is flattened batch x sequence. Index accordingly when batching.

## Related

- [ablation](ablation.md) - masking a router logit to ablate an expert
- [logit-lens](logit-lens.md) - the dense-model counterpart for per-layer decoding signals
- [docs/usage/iter-all-next.md](../usage/iter-all-next.md) - `tracer.iter` for per-step collection
- InnerExpert, ["Mixture-of-Expert Blocks Contain Strong Hallucination Detection Signals"](https://arxiv.org/abs/2608.17687) (arXiv:2608.17687)
