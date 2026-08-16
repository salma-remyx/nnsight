---
title: Model Diffing with SAE Features
one_liner: Diff SAE feature activations between a base model and a fine-tuned (e.g. multimodal) counterpart to isolate training-induced features, then remove or steer them.
tags: [pattern, interpretability, sae, model-diffing, vlm, steering]
related: [docs/patterns/sae-and-auxiliary-modules.md, docs/patterns/steering.md, docs/patterns/ablation.md]
sources: [src/nnsight/intervention/feature_diff.py, src/nnsight/modeling/vlm.py]
---

# Model Diffing with SAE Features

Adapted from "Multimodal Model Diffing for Feature Discovery and Control" (MMDiff, arXiv:2608.09928).

## What this is for

A base language model and its fine-tuned counterpart (e.g. a multimodal-adapted VLM) share most of their weights, but the fine-tuning *changes* which features fire. Encoding identical text through both models and diffing SAE feature activations isolates exactly those changed features — which are candidate causal levers for the new behavior. The same machinery supports per-task contrastive analysis (positive vs. negative example sets) and feature-level control (removing or steering the discovered directions).

Three building blocks live in `nnsight.intervention.feature_diff`:

- `feature_activation_diff(source_features, target_features)` — per-feature mean diff + firing rates, with `.topk(k)` for ranking.
- `contrastive_feature_scores(positive_features, negative_features)` — the same diff framed as task-specific feature detection.
- `steer_features(hidden, decoder_directions, indices, alpha, features, mode)` — `"remove"` ablates a feature's own contribution; `"steer"` adds its decoder direction scaled by `alpha`.

## Pattern: diff a VLM against its base LM

`VisionLanguageModel.diff_features` runs the whole collection for you: it traces identical text through both models, encodes the hidden states at the given layers, and returns a `FeatureDiff`. Both models must be loaded (`dispatch=True`); pass text-only inputs.

```python
import torch
from nnsight import LanguageModel, VisionLanguageModel

vlm = VisionLanguageModel(
    "llava-hf/llava-interleave-qwen-0.5b-hf", device_map="auto", dispatch=True
)
base = LanguageModel("Qwen/Qwen2-0.5B", device_map="auto", dispatch=True)

sae = ...  # any SAE with .encode / a decoder whose rows are feature directions

diff = vlm.diff_features(
    base,
    "A photo of a dog sitting on the left of a cat",
    encoder=sae.encode,
    layer="model.language_model.layers.12",
    base_layer="model.layers.12",
)

top = diff.topk(10)  # features most up-regulated by multimodal training
```

`layer` / `base_layer` are dotted module paths so the same call works across architectures (tuple-returning blocks are unwrapped automatically).

## Pattern: contrastive task-specific detection

Collect SAE features on a positive example set (the behavior is present) and a matched negative set, then diff:

```python
from nnsight.intervention.feature_diff import contrastive_feature_scores

scores = contrastive_feature_scores(positive_features, negative_features)
causal_candidates = scores.topk(20).indices
```

## Pattern: feature-level control inside a trace

Remove or steer the discovered features with `steer_features`. `mode="remove"` subtracts each feature's own contribution (its activation times its decoder row); `mode="steer"` adds `alpha * direction` at every position.

```python
from nnsight.intervention.feature_diff import steer_features

with vlm.trace("Describe the spatial layout of this image", images=[img]):
    layer = vlm.model.language_model.layers[12]
    feats = sae.encode(layer.output)
    layer.output = steer_features(
        layer.output, sae.decoder.weight, causal_candidates,
        features=feats, mode="remove",
    )
    logits = vlm.lm_head.output[:, -1, :].save()
```

## Gotchas

- `.diff_features` opens its own traces internally; call it outside any `with model.trace(...)` block, and `.save()` any tensors you collect yourself.
- The encoder is applied to saved hidden states after each trace exits, so it runs as ordinary PyTorch — no tracing constraints on the SAE itself.
- Diff rankings reflect correlation, not causation: validate top features with `steer_features(..., mode="remove")` before drawing conclusions, as in the paper.
