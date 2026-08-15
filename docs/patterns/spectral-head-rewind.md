---
title: Spectral Head Rewind
one_liner: Score attention heads by how much continual pre-training moved them, and rewind the low-importance ones to the pre-trained weights with model.edit().
tags: [pattern, interpretability, attention, heads, edit, domain-adaptation]
related: [docs/patterns/per-head-attention.md, docs/usage/edit.md, docs/patterns/ablation.md]
sources: [src/nnsight/modeling/spectral_rewind.py, src/nnsight/intervention/tracing/editing.py, docs/usage/edit.md]
---

# Spectral Head Rewind

## What this is for

You have two checkpoints of the same architecture — a general pre-trained
model and a continually pre-trained (CPT) domain adaptation of it — and want
to answer: **which attention heads actually carried the domain adaptation,
and what happens if the rest go back to their pre-trained weights?**

`nnsight.modeling.head_importance` scores each head by the relative magnitude
of the update its rows of the attention output projection received during
CPT, and `nnsight.modeling.rewind_heads` restores the low-importance heads to
the pre-trained weights via `model.edit()` — so the rewound model is an
ordinary edited nnsight model you can trace, generate from, and A/B against
both checkpoints.

Adapted from *Diffract: Spectral View of LLM Domain Adaptation*
([arXiv:2608.10850](https://arxiv.org/abs/2608.10850v1)), which defines a
head-importance criterion from SVD of attention-head projection matrices,
shows that up to 60% of head updates can be removed without measurable
quality loss, and that selective rewinding can improve benchmark accuracy by
up to 4% over the fully-trained baseline. Here the paper's full SVD
subspace-distance criterion is replaced by a parameter-free relative-update
proxy (per-head Frobenius norm of the weight delta, normalized by the
pre-trained rows), and the paper's benchmark suite is out of scope — bring
your own eval.

## Canonical pattern

```python
from nnsight import LanguageModel
from nnsight.modeling import head_importance, rewind_heads

base = LanguageModel("your-org/base-model", dispatch=True)
cpt  = LanguageModel("your-org/domain-adapted", dispatch=True)

# 1. Score every attention head by how far CPT moved it.
scores = head_importance(cpt, base)

for path, per_head in scores.items():
    print(path, per_head.shape)   # e.g. model.layers.11.self_attn.o_proj  [32]

# 2. Rewind the lowest 60% (the paper's headroom) to pre-trained weights.
rewound = rewind_heads(cpt, base, scores, fraction=0.6)

# 3. Evaluate as usual — `rewound` is an edited model sharing cpt's module.
with rewound.trace(prompt):
    logits = rewound.lm_head.output.save()
```

## How the score works

An attention output projection `W` maps the concatenated head outputs back
to the residual stream, with shape `[hidden, n_heads * head_dim]`. The row
block `W[i * head_dim : (i + 1) * head_dim]` produces head `i`'s
contribution, so the paper's question "did CPT move this head?" becomes a
per-block distance:

```
score[i] = ||W_cpt[i] - W_base[i]||_F  /  ||W_base[i]||_F
```

This is the dominant (mode-2) component of the full SVD distance the paper
computes, and it needs no eigen-decomposition — a useful stand-in when you
are ranking heads rather than studying subspace geometry. The paper's
stronger finding — that CPT leaves *singular value spectra* largely invariant
and adaptation is driven by *singular vector* rotation — is precisely why
magnitude-of-update is a defensible proxy here: the heads that matter are the
ones whose subspaces actually turned, not the ones that merely scaled.

`head_importance` discovers projections by name (`c_proj`, `o_proj`,
`out_proj`, `wo`) anywhere in the module tree, so it works on GPT-2,
Llama-family, Qwen, and most HF attention stacks without per-architecture
plumbing. Head counts are read off the attention module (`num_heads`,
`n_heads`, ...) with a fallback to `model.config.num_attention_heads`; pass
`n_heads=` to override both.

## Choosing the fraction

```python
from nnsight.modeling import low_importance_heads

# Inspect before rewinding.
selection = low_importance_heads(scores, fraction=0.6)
for path, heads in selection.items():
    print(path, heads)

# Or rewind an explicit selection.
rewound = rewind_heads(cpt, base, scores, fraction=selection)
```

The paper reports headroom up to 60% of head updates removed without
measurable quality loss, which is why `0.6` is the default. In practice you
sweep: `fraction=0.0` is the CPT baseline, `fraction=1.0` is full rewind to
the pre-trained model, and the interesting region is between — the paper
found a sweet spot *above zero*, i.e. some CPT head updates actively hurt
general benchmarks.

## What you get back

`rewind_heads` runs the row-block copies inside `with model.edit():` and
returns the edited handle, so:

- `rewound.trace(...)` / `rewound.generate(...)` run the rewound model.
- The copy is applied in place on the shared underlying module — the handle
  and `cpt` are the same `torch.nn.Module` (see [edit](../usage/edit.md) for
  the shallow-copy semantics). Rewind a fresh copy if you need both variants
  side by side.
- The head *activations* are untouched; only the output-projection weights
  move. To also ablate a head's activation at runtime, combine with
  [per-head-attention](per-head-attention.md).

## Variations

### Rank heads globally across layers

```python
flat = sorted(
    (float(s), path, i)
    for path, per_head in scores.items()
    for i, s in enumerate(per_head)
)
```

The paper observes strong *domain-dependent* heterogeneity — code adaptation
moves a different head population than math does. Comparing the ranked head
lists across two CPT checkpoints of the same base is often the interesting
plot.

### Zero-ablate the low-importance heads instead

The flip experiment from the paper: if the score is right, *deleting* the
low-importance heads' CPT updates and *rewinding* them should behave
similarly, and zeroing a *high*-importance head should hurt.

```python
with cpt.edit() as ablated:
    # zero head 3's rows in layer 11's o_proj (see per-head-attention.md for
    # the activation-space version)
    W = ablated.model.layers[11].self_attn.o_proj.weight
    W[3 * head_dim : 4 * head_dim] = 0
```

## Gotchas

- Both models must share an architecture — paths resolve on `base_model` by
  the same dotted name. Fine-tunes, CPT checkpoints, and quantization-free
  merges of the same base qualify; two different model families do not.
- `head_importance` raises if a projection's output width is not divisible by
  the discovered head count (MQA/GQA stacks where `o_proj` is shared but the
  module lacks a head-count attribute). Pass `n_heads=` explicitly.
- The score is weight-space, not activation-space: a head whose rows barely
  moved can still be *functionally* important if CPT changed everything
  upstream of it. Cross-check with attribution patching
  ([attribution-patching](attribution-patching.md)) before drawing causal
  conclusions.

## Related

- [per-head-attention](per-head-attention.md) - Activation-space access to a
  single head, the natural companion to weight-space scoring.
- [edit](../usage/edit.md) - The persistent-edit mechanism `rewind_heads`
  builds on.
- `src/nnsight/modeling/spectral_rewind.py` - Implementation.
- [arXiv:2608.10850](https://arxiv.org/abs/2608.10850v1) - Diffract: Spectral
  View of LLM Domain Adaptation.
