"""Spectral head-importance scoring and selective rewinding of attention heads.

Given two checkpoints of the same architecture — a base (pre-trained) model
and a continually pre-trained (CPT) adaptation of it — this module:

1. Locates the attention output-projection matrices (``c_proj`` /
   ``o_proj`` / ``out_proj``) in the wrapped model.
2. Scores each attention head by the size of the update its rows of the
   projection received during CPT, relative to the pre-trained rows.
3. Rewinds the lowest-importance heads to their pre-trained weights via
   ``model.edit()``, leaving high-importance heads on the adapted weights.

Adapted from "Diffract: Spectral View of LLM Domain Adaptation"
(arXiv:2608.10850), which defines a head-importance criterion from SVD of
attention-head projection matrices and shows that rewinding low-importance
heads to the pre-trained state can improve over the fully-trained baseline.
Here the paper's full SVD/subspace distance is replaced by a parameter-free
relative-update-magnitude proxy (the mode-2 component of the SVD distance:
the Frobenius norm of each head's row-block update, normalized by the
pre-trained block), and the paper's benchmark suite is cut — the deliverable
is the nnsight-native scoring + rewinding capability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import torch

if TYPE_CHECKING:
    from .base import NNsight
else:
    NNsight = object

#: Attribute names of attention output projections across common architectures.
OUTPUT_PROJECTION_NAMES = ("c_proj", "o_proj", "out_proj", "wo")

#: Attribute names that may carry the head count on the attention module.
HEAD_COUNT_ATTRS = ("num_heads", "n_heads", "num_attention_heads")


def _head_count(module: torch.nn.Module, config) -> int:
    """Best-effort head count for the module owning an output projection."""
    for attr in HEAD_COUNT_ATTRS:
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    for attr in HEAD_COUNT_ATTRS:
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        "Could not determine attention head count for "
        f"`{type(module).__name__}`. Pass `n_heads=` explicitly."
    )


def _iter_output_projections(model: NNsight) -> List[Tuple[str, torch.nn.Linear]]:
    """Yield ``(path, module)`` for every attention output projection found.

    A module qualifies if its own name is one of ``OUTPUT_PROJECTION_NAMES``
    and it exposes a 2-D ``weight``.
    """
    found = []
    for path, module in model._module.named_modules():
        if path.split(".")[-1] not in OUTPUT_PROJECTION_NAMES:
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            found.append((path, module))
    return found


def _resolve(model: NNsight, path: str) -> torch.nn.Module:
    """Resolve a dotted module path on the underlying module."""
    return model._module.get_submodule(path)


def _parent(path: str) -> str:
    """Parent path of a dotted module path (``''`` for a top-level module)."""
    return path.rsplit(".", 1)[0] if "." in path else ""


def spectral_delta(weight: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
    """Relative update magnitude of a weight matrix.

    ``||weight - base_weight||_F / ||base_weight||_F`` — the mode-2 (largest
    singular value) component of the full SVD distance between the two
    matrices, and a parameter-free stand-in for the paper's subspace-distance
    criterion.
    """
    delta = weight.detach() - base_weight.detach()
    denominator = base_weight.detach().norm()
    if denominator == 0:
        return delta.norm()
    return delta.norm() / denominator


def head_importance(
    model: NNsight,
    base_model: NNsight,
    n_heads: int = None,
) -> Dict[str, torch.Tensor]:
    """Score every attention head by how much CPT moved its projection rows.

    For each discovered output projection ``W`` (shape ``[hidden, n_heads *
    head_dim]``), the row block ``[i * head_dim : (i + 1) * head_dim]``
    produces head ``i``'s contribution to the residual stream, so the
    relative Frobenius norm of that block's update is a per-head importance
    score (higher = the head was updated more during adaptation).

    Args:
        model: The adapted (continually pre-trained) model, wrapped in NNsight.
        base_model: The pre-trained model with the same architecture. Only its
            parameters are read.
        n_heads: Head count to assume for every projection. If ``None``, it is
            discovered per-attention-module (falling back to the model config).

    Returns:
        Mapping from projection path (e.g. ``"model.layers.5.self_attn.o_proj"``)
        to a ``[n_heads]`` float tensor of importance scores.
    """
    scores: Dict[str, torch.Tensor] = {}
    projections = _iter_output_projections(model)

    for path, module in projections:
        base_module = _resolve(base_model, path)
        base_weight = base_module.weight

        heads = n_heads or _head_count(
            _resolve(model, _parent(path)), getattr(model, "config", None)
        )
        out_features = module.weight.shape[0]
        if out_features % heads != 0:
            raise ValueError(
                f"`{path}` has {out_features} output features, not divisible "
                f"by {heads} heads. Pass `n_heads=` explicitly."
            )
        head_dim = out_features // heads

        per_head = torch.empty(heads)
        for i in range(heads):
            rows = slice(i * head_dim, (i + 1) * head_dim)
            per_head[i] = spectral_delta(module.weight[rows], base_weight[rows])
        scores[path] = per_head

    return scores


def low_importance_heads(
    scores: Dict[str, torch.Tensor],
    fraction: float = 0.6,
) -> Dict[str, List[int]]:
    """Select the lowest-``fraction`` of heads per projection.

    Mirrors the paper's finding that up to 60% of head updates can be removed
    without measurable quality loss; the default fraction matches it.
    """
    selection: Dict[str, List[int]] = {}
    for path, per_head in scores.items():
        k = max(0, min(len(per_head), round(fraction * len(per_head))))
        order = torch.argsort(per_head)
        selection[path] = sorted(int(i) for i in order[:k])
    return selection


def rewind_heads(
    model: NNsight,
    base_model: NNsight,
    scores: Dict[str, torch.Tensor],
    fraction: Union[float, Dict[str, List[int]]] = 0.6,
    n_heads: int = None,
) -> NNsight:
    """Rewind low-importance attention heads to their pre-trained weights.

    Runs the row-block copies inside a ``model.edit()`` context and returns the
    edited model handle, so the rewound model behaves like any other edited
    nnsight model. Note that — like all edits — the handle shares the
    underlying ``torch.nn.Module`` with ``model``: the weight copy is applied
    in place, and the returned handle simply carries the edit state.

    Args:
        model: The adapted model to rewind.
        base_model: The pre-trained model whose head weights are restored.
        scores: Per-projection head importance from :func:`head_importance`.
        fraction: Either a float — rewind the lowest ``fraction`` of heads per
            projection — or an explicit ``{path: [head indices]}`` selection
            from :func:`low_importance_heads`.
        n_heads: Head count, as in :func:`head_importance`.

    Returns:
        The edited model (a shallow copy of ``model`` sharing its weights).
    """
    if isinstance(fraction, dict):
        selection = fraction
    else:
        selection = low_importance_heads(scores, fraction)

    with model.edit() as edited:
        for path, heads in selection.items():
            module = _resolve(model, path)
            base_weight = _resolve(base_model, path).weight

            resolved_heads = n_heads or _head_count(
                _resolve(model, _parent(path)), getattr(model, "config", None)
            )
            head_dim = module.weight.shape[0] // resolved_heads
            for i in heads:
                rows = slice(i * head_dim, (i + 1) * head_dim)
                module.weight.data[rows].copy_(base_weight[rows])

    return edited
