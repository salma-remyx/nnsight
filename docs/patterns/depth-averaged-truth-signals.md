---
title: Depth-Averaged Truth Signals
one_liner: Fit a linear truth probe at every layer from one activation cache, then average the scores across depth to detect hallucinations.
tags: [pattern, interpretability, hallucination, safety, linear-probe]
related: [docs/usage/cache.md, docs/patterns/logit-lens.md, docs/patterns/steering.md]
sources: [src/nnsight/modeling/truthfulness.py, src/nnsight/intervention/tracing/tracer.py:545]
---

# Depth-Averaged Truth Signals

## What this is for

Adapted from *HalluTracer: Hallucination Detection via Depth-Averaging Truth Signals* (arXiv:2608.16353). The paper's finding is that LLM hidden states carry a *linearly separable* truthfulness signal, and — the important part — that this signal is **not concentrated at one depth**. Per-layer probes are only weakly correlated, so averaging their scores across the full forward pass suppresses layer-specific noise and captures nearly all of the linearly accessible evidence. Hallucination detection is a depth-*aggregation* problem, not a layer-*selection* problem.

The practical consequence for interpretability work: stop hunting for the magic layer, and stop reading a single probe. Read them all and average.

nnsight's activation cache ([docs/usage/cache.md](../usage/cache.md)) is the natural substrate — `tracer.cache()` with no module filter records every layer's output in one call, which is exactly the depth-wide tensor this needs.

## When to use

- Scoring generations for hallucination risk *before* the answer token is emitted (white-box detection).
- Deciding whether a single-layer probe you trained is leaving signal on the table.
- Measuring how redundant a model's layers are for a given linear property.

## Canonical pattern

```python
from nnsight import LanguageModel
from nnsight.modeling import fit_truth_detector, truth_score
from nnsight.modeling.truthfulness import layer_states

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

# Prompts whose truthfulness you know, from any labeled set you have.
true_prompts = ["The Eiffel Tower is in the city of Paris."]
false_prompts = ["The Eiffel Tower is in the city of London."]

def collect(prompts):
    with model.trace(prompts) as tracer:
        cache = tracer.cache()
    # "transformer.h" selects the residual stream at every block, in depth
    # order. position="last" reads the next-token site the paper probes.
    return layer_states(cache, pattern="transformer.h", position="last")

truthful = collect(true_prompts)    # [depth, batch, hidden]
fabricated = collect(false_prompts) # [depth, batch, hidden]

states = torch.cat([truthful, fabricated], dim=1)
labels = torch.cat([torch.ones(len(true_prompts)),
                    torch.zeros(len(false_prompts))])

detector = fit_truth_detector(states, labels)

# Score new prompts: higher = more truthful, compare to detector.threshold.
scores, curve = truth_score(detector, collect(["Madison Square Garden is in the city of"]))
print("truthful" if scores[0] > detector.threshold else "hallucinated")
```

## Variations

### Inspect what averaging actually bought you

`detector.metrics` records the per-layer accuracies next to the depth-averaged one, so you can see whether the model's layers agree or each contribute something:

```python
print(f"averaged:   {detector.metrics['averaged_accuracy']:.3f}")
print(f"best layer: {detector.metrics['best_layer_accuracy']:.3f}")
print(f"mean layer: {detector.metrics['per_layer_accuracy']:.3f}")
```

If `best_layer_accuracy` is close to `averaged_accuracy`, one layer already carries the signal and depth is doing little. If the averaged score leads, the paper's aggregation effect is present in your model.

### Measure layer redundancy directly

`layer_agreement` returns the fraction of layer pairs whose score signs agree, per prompt. This is the observable behind the paper's geometric claim — near `1.0` means redundant layers (averaging adds little), near `0.5` means independent voters (averaging cancels real noise):

```python
from nnsight.modeling.truthfulness import layer_agreement

agreement = layer_agreement(curve)   # [batch]
```

### Use it on a VLM

The same recipe works on `VisionLanguageModel` — only the layer pattern changes. For LLaVA-style models the language stack is under `language_model.model.layers`:

```python
from nnsight import VisionLanguageModel

vlm = VisionLanguageModel("llava-hf/llava-interleave-qwen-0.5b-hf", device_map="auto")

with vlm.trace(prompt, image) as tracer:
    cache = tracer.cache()

states = layer_states(cache, pattern="language_model.model.layers")
```

### Score a generation step

When the cache accumulated entries across generation steps, pass `step` to read a specific one (see the [cache docs](../usage/cache.md) for the `Entry` vs `list[Entry]` distinction):

```python
states = layer_states(cache, pattern="transformer.h", step=4)
```

## Interpretation tips

- **The probe is only as good as the labels.** A linear probe learns whatever separates your two prompt sets. If the split is something simpler than truthfulness (length, topic, entity frequency), that is what it will read. Control for it in your prompt design.
- **Score before the answer token.** The paper's point is that the signal is present *pre-emission*. Probing at a position that already contains the generated answer measures something else.
- **`position="mean"`** is a useful robustness check — if the detection only works at one position, be suspicious of the probe.
- **The threshold is fit on your data.** `detector.threshold` defaults to the midpoint of the averaged training scores. Recalibrate it on held-out data before treating it as an operating point.

## Gotchas

- **All selected layers must share a hidden size.** `layer_states` raises if they don't — a pattern that sweeps in both residual blocks and a narrower submodule (attention outputs, an `lm_head`) will trip this. Keep the pattern on one module family.
- **Layer order comes from numeric index**, so `h.10` correctly sorts after `h.9`. If your model's layer names carry no index, pass an explicit depth-ordered `paths=[...]` instead of a pattern.
- **2-D vs 3-D outputs.** A plain `nn.Linear` stack caches `[batch, hidden]` with no sequence axis; transformer blocks cache `[batch, sequence, hidden]`. `layer_states` handles both, but `position` only means something in the second case.
- **`tracer.cache()` must be called inside the trace**, and before any interventions you want it to see past. See [docs/usage/cache.md](../usage/cache.md).

## Related

- [activation-cache](../usage/cache.md) — the underlying cache API.
- [logit-lens](logit-lens.md) — another per-layer read of the residual stream, decoding tokens instead of a trained property.
- [steering](steering.md) — the write-side counterpart: modify a direction rather than read one.
