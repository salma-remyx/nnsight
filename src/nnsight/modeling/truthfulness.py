"""Depth-averaged truth-signal reading over cached layer activations.

Adapted from *HalluTracer: Hallucination Detection via Depth-Averaging Truth
Signals* (arXiv:2608.16353).  The paper's core observation is that
truthfulness evidence in an LLM is not concentrated at one depth: per-layer
probes are only weakly correlated, so **averaging the per-layer scores across
the whole forward pass** suppresses layer-specific noise and recovers nearly
all of the linearly accessible signal.  Hallucination detection is therefore a
depth-*aggregation* problem rather than a layer-*selection* problem.

This module implements that aggregation on top of the activation cache
(:meth:`nnsight.intervention.tracing.tracer.InterleavingTracer.cache`), which
already records every module's output across the full forward depth:

>>> with model.trace(prompt) as tracer:
...     cache = tracer.cache()                      # every layer, one call
>>> states = layer_states(cache, "transformer.h")
>>> detector = fit_truth_detector(states, labels)
>>> score, curve = truth_score(detector, states)    # depth-averaged

Scope decisions relative to the paper: the learned per-layer probes are plain
ridge-regression linear maps (the paper's "linearly separable truthfulness
signals"), trained on the unembeddings-free hidden state at each depth.  The
paper's benchmark suite, its six-model sweep, and its geometric sparsity
analysis are out of scope here -- :func:`layer_agreement` keeps only the
diagnostic the aggregation actually rests on (how weakly correlated the
per-layer signals are, i.e. how much depth averaging buys you).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "TruthDetector",
    "fit_truth_detector",
    "layer_agreement",
    "layer_states",
    "truth_score",
]


def _entry_outputs(entry: Any) -> List[Any]:
    """Unwrap a cache entry into a list of per-hit outputs.

    A cached path holds a single :class:`Cache.Entry` when the module fired
    once and a ``list[Cache.Entry]`` when it fired more than once (e.g. across
    generation steps).  Transformer blocks may also return tuples, in which
    case the residual stream is the first element.
    """
    entries = entry if isinstance(entry, list) else [entry]
    outputs: List[Any] = []
    for e in entries:
        value = e.output if hasattr(e, "output") else e
        if isinstance(value, tuple):
            value = value[0]
        if value is not None:
            outputs.append(value)
    return outputs


def layer_states(
    cache: Any,
    pattern: Optional[str] = None,
    paths: Optional[Sequence[str]] = None,
    position: Union[int, str, None] = "last",
    step: Optional[int] = None,
) -> torch.Tensor:
    """Stack cached layer outputs into a ``[depth, batch, hidden]`` tensor.

    Args:
        cache: A :class:`~nnsight.intervention.tracing.tracer.Cache.CacheDict`
            as returned by ``tracer.cache()``.
        pattern: Substring used to pick depth-ordered layer paths, e.g.
            ``"transformer.h"`` for GPT-2 or ``"model.layers"`` for Llama.
            Paths are sorted by their numeric index, so depth order matches
            forward order.  Mutually exclusive with ``paths``.
        paths: Explicit depth-ordered list of cache keys.  Use this when the
            layer naming is not index-sortable.
        position: Which sequence position to read.  ``"last"`` takes the
            final position (the next-token prediction site, which is what the
            paper probes before any answer token is emitted); ``"mean"``
            averages over positions; an ``int`` indexes directly.
        step: Generation step to read when a path accumulated multiple
            entries.  ``None`` takes the first hit.

    Returns:
        Float tensor of shape ``[depth, batch, hidden]``.
    """

    if (pattern is None) == (paths is None):
        raise ValueError("Give exactly one of `pattern` or `paths`.")

    if pattern is not None:
        selected = [key for key in cache if pattern in key]
        selected.sort(key=lambda key: _trailing_index(key))
        paths = selected

    states: List[torch.Tensor] = []
    for path in paths:
        outputs = _entry_outputs(cache[path])

        if not outputs:
            raise KeyError(f"Cache has no output recorded for {path!r}.")

        value = outputs[0] if step is None else outputs[step]

        if value.dim() == 2:
            # [batch, hidden] -- no sequence axis to select a position from.
            states.append(value.to(torch.float32))
            continue

        if value.dim() != 3:
            raise ValueError(
                f"Cached output for {path!r} has {value.dim()} dim(s); "
                "expected [batch, hidden] or [batch, sequence, hidden]."
            )

        if position == "last":
            value = value[:, -1, :]
        elif position == "mean":
            value = value.mean(dim=1)
        elif isinstance(position, int):
            value = value[:, position, :]
        else:
            raise ValueError(f"Unknown position {position!r}.")

        states.append(value.to(torch.float32))

    if not states:
        raise ValueError(f"No cached paths matched {pattern!r}.")

    hidden = states[0].shape[-1]
    for depth, state in enumerate(states):
        if state.shape[-1] != hidden:
            raise ValueError(
                f"Depth {depth} has hidden size {state.shape[-1]}, "
                f"expected {hidden}; the selected paths are not all the same "
                "kind of layer."
            )

    return torch.stack(states, dim=0)


def _trailing_index(path: str) -> Tuple[int, ...]:
    """Sort key pulling every integer out of a path, e.g. ``"a.h.10.attn"``."""

    parts: List[int] = []
    token = ""
    for char in path:
        if char.isdigit():
            token += char
        elif token:
            parts.append(int(token))
            token = ""
    if token:
        parts.append(int(token))
    return tuple(parts)


def _probe(
    states: torch.Tensor, labels: torch.Tensor, ridge: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit one least-squares probe ``label ~ state @ w + b`` for a single depth.

    ``states`` is that depth's ``[batch, hidden]`` slice.
    """

    flat = states.reshape(states.shape[0], -1)
    targets = labels.reshape(-1, 1).to(flat.dtype)

    design = torch.cat([flat, torch.ones(flat.shape[0], 1, dtype=flat.dtype)], dim=1)
    gram = design.T @ design
    ridge_eye = ridge * torch.eye(design.shape[1], dtype=flat.dtype)
    gram.diagonal().add_(ridge_eye.diagonal())

    weights = torch.linalg.lstsq(gram, design.T @ targets).solution

    return weights[:-1].squeeze(-1), weights[-1].item()


@dataclass
class TruthDetector:
    """Per-depth linear probes plus the depth-averaged score they define.

    Attributes:
        weights: ``[depth, hidden]`` probe directions, one per layer.
        biases: ``[depth]`` probe intercepts.
        paths: Depth-ordered cache keys the probes were fit on.  Kept so a
            detector can be re-applied to a new cache by pattern.
        threshold: Score at which a prediction flips to ``hallucinated``.
            Defaults to the midpoint of the fitted scores, i.e. the
            accuracy-maximizing split on the training data.
        metrics: Fit-time diagnostics (per-layer and averaged accuracy).
    """

    weights: torch.Tensor
    biases: torch.Tensor
    paths: List[str] = field(default_factory=list)
    threshold: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        """Number of layers the detector reads."""

        return self.weights.shape[0]

    def __repr__(self) -> str:
        per_layer = self.metrics.get("per_layer_accuracy")
        agreement = self.metrics.get("mean_layer_agreement")

        return (
            f"{self.__class__.__name__}(depth={self.depth}, "
            f"threshold={self.threshold:.4f}"
            + (
                f", per_layer_accuracy={per_layer:.4f}"
                if per_layer is not None
                else ""
            )
            + (
                f", mean_layer_agreement={agreement:.4f}"
                if agreement is not None
                else ""
            )
            + ")"
        )


def fit_truth_detector(
    states: torch.Tensor,
    labels: torch.Tensor,
    paths: Optional[Sequence[str]] = None,
    ridge: float = 1e-2,
) -> TruthDetector:
    """Fit a per-depth linear truth probe and its depth-averaged score.

    Args:
        states: ``[depth, batch, hidden]`` hidden states from
            :func:`layer_states`.
        labels: ``[batch]`` with ``1`` for truthful and ``0`` for hallucinated
            examples.
        paths: Optional depth-ordered cache keys, stored on the detector.
        ridge: Small L2 term keeping the per-layer least-squares fits stable
            when a depth has fewer dimensions than examples.

    Returns:
        A :class:`TruthDetector`.  Its ``threshold`` is the midpoint of the
        depth-averaged training scores, the split that maximizes training
        accuracy for a symmetric scorer.
    """

    if states.dim() != 3:
        raise ValueError(
            f"`states` must be [depth, batch, hidden]; got {tuple(states.shape)}."
        )
    if labels.dim() != 1 or labels.shape[0] != states.shape[1]:
        raise ValueError(
            f"`labels` must be [batch={states.shape[1]}]; got {tuple(labels.shape)}."
        )

    depth = states.shape[0]
    weights = torch.empty(depth, states.shape[-1], dtype=states.dtype)
    biases = torch.empty(depth, dtype=states.dtype)

    for layer in range(depth):
        weights[layer], biases[layer] = _probe(
            states[layer], labels.to(states.dtype), ridge
        )

    # Each layer scored by its own probe, then averaged over depth.
    averaged = (
        torch.einsum("lbh,lh->lb", states, weights).mean(dim=0) + biases.mean()
    )

    detector = TruthDetector(
        weights=weights,
        biases=biases,
        paths=list(paths) if paths is not None else [],
        threshold=float((averaged.max() + averaged.min()) / 2),
    )

    per_layer = torch.stack(
        [
            ((states[layer] @ weights[layer] + biases[layer]) > 0).eq(
                labels.to(states.dtype) == 1
            )
            .float()
            .mean()
            for layer in range(depth)
        ]
    )
    averaged_accuracy = (
        (averaged > detector.threshold).eq(labels == 1).float().mean().item()
    )

    detector.metrics = {
        "averaged_accuracy": averaged_accuracy,
        "per_layer_accuracy": per_layer.mean().item(),
        "best_layer_accuracy": per_layer.max().item(),
        "per_layer_accuracy_curve": per_layer,
    }

    return detector


def truth_score(
    detector: TruthDetector,
    states: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score cached hidden states with a fitted detector.

    Args:
        detector: A detector from :func:`fit_truth_detector`.
        states: ``[depth, batch, hidden]`` states for the prompts to score.

    Returns:
        ``(scores, curve)`` where ``scores`` is the depth-averaged truth score
        per batch element (higher = more truthful; compare against
        ``detector.threshold``) and ``curve`` is the raw ``[depth, batch]``
        per-layer score matrix.
    """

    if states.dim() != 3:
        raise ValueError(
            f"`states` must be [depth, batch, hidden]; got {tuple(states.shape)}."
        )
    if states.shape[0] != detector.depth:
        raise ValueError(
            f"Detector reads {detector.depth} layers; got {states.shape[0]}."
        )
    if states.shape[-1] != detector.weights.shape[-1]:
        raise ValueError(
            f"Detector expects hidden size {detector.weights.shape[-1]}; "
            f"got {states.shape[-1]}."
        )

    # Layer l scored by probe l only, giving the [depth, batch] curve.
    curve = (
        torch.einsum("lbh,lh->lb", states, detector.weights)
        + detector.biases.unsqueeze(1)
    )

    return curve.mean(dim=0), curve


def layer_agreement(curve: torch.Tensor) -> torch.Tensor:
    """Pairwise agreement between per-layer score signs.

    The paper's geometric finding is that per-layer truth signals are weakly
    correlated, which is *why* depth averaging helps: independent errors
    cancel.  This returns the fraction of agreeing layer pairs per batch
    element.  Near ``1.0`` means the layers are redundant and averaging adds
    little; near ``0.5`` means they disagree like independent voters and
    averaging is doing real work.
    """

    if curve.dim() != 2:
        raise ValueError(f"`curve` must be [depth, batch]; got {tuple(curve.shape)}.")

    depth = curve.shape[0]
    if depth < 2:
        raise ValueError("Need at least two layers to measure agreement.")

    signs = (curve > 0).float().T
    # [batch, depth, depth]: element b's layer i vs layer j, so the
    # off-diagonal mean is that element's pairwise agreement.
    pairs = signs.unsqueeze(1) == signs.unsqueeze(2)

    mask = ~torch.eye(depth, dtype=torch.bool, device=curve.device)
    total = mask.sum().item()

    return (pairs * mask).sum(dim=(1, 2)).div(float(total))
