"""Per-token routing signals from Mixture-of-Experts router logits.

MoE routers expose a signal dense transformers do not have: for every token,
the full distribution over experts. ``"Mixture-of-Expert Blocks Contain
Strong Hallucination Detection Signals"`` (InnerExpert, arXiv:2608.17687)
shows that routing-level statistics of that distribution — router entropy,
top-k disagreement, expert usage — carry per-token signal about whether the
model is about to produce unsupported content, and that they combine with
ordinary transformer signals into cheap single-forward-pass features.

This module computes those routing statistics. It does **not** ship the
paper's trained detector: the features here are the parameter-free part of
the method, ready to be correlated against whatever labels you have.

The one wrinkle worth a helper is that there is no single "router output"
shape. Routers disagree on what they return:

=========================  =====================  ==================
module                    returns                layout
=========================  =====================  ==================
Qwen3-MoE / Mixtral       ``(logits, scores,     logits last
``*TopKRouter``           indices)``
GraniteMoe                ``(indices, weights,   logits last,
``GraniteMoeTopKRouter``  logits)``              order flipped
DeepSeek                  ``(topk_idx, topk_w,   version-dependent
                          router_l)``
vLLM ``mlp.gate``         ``(logits, ...)`` or   logits first
(Replicated)              bare logits
=========================  =====================  ==================

:func:`normalize_router_output` recovers ``(logits, topk_weights,
topk_indices)`` from any of them by shape: indices are integer-valued,
weights are top-k-sized, logits are the only full-width float entry. Once
normalized, :func:`routing_features` turns a router output into named
per-token statistics and :func:`find_routers` walks an envoy tree so you
don't have to remember whether a given architecture calls it ``mlp.gate``,
``block_sparse_moe.router``, or ``gate``.

Example:
    >>> from nnsight import LanguageModel
    >>> from nnsight.modeling.moe_routing import routing_features, find_routers
    >>> model = LanguageModel("ibm/PowerMoE-3b", device_map="cpu", dispatch=True)
    >>> routers = find_routers(model)
    >>> with model.trace("The Eiffel Tower is located in the city of"):
    ...     feats = routing_features(routers[0].output)
    ...     entropy = feats["router_entropy"].save()
    >>> entropy.shape   # one value per token
    torch.Size([12])
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..intervention.envoy import Envoy

__all__ = [
    "ROUTING_FEATURE_NAMES",
    "find_routers",
    "normalize_router_output",
    "routing_features",
    "expert_usage",
]

#: Ordered names of the columns :func:`routing_features` produces. Layer-wise
#: features are prefixed ``"<name>@<layer>"``; this tuple is the bare name.
#: ``top1_expert`` is the one integer column — the winning expert id per token.
ROUTING_FEATURE_NAMES = (
    "router_entropy",
    "topk_margin",
    "topk_weight_mass",
    "topk_entropy",
    "expert_load",
    "top1_expert",
)


def _num_experts_from_logits(logits: torch.Tensor) -> int:
    """Width of the router-logit axis."""
    return int(logits.shape[-1])


def _flatten(value: Any) -> List[torch.Tensor]:
    """Spread a router output (tensor, tuple, or nested) into flat tensors."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        out: List[torch.Tensor] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return []


def normalize_router_output(
    output: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Recover ``(logits, topk_weights, topk_indices)`` from any router output.

    Args:
        output: Whatever the router module returned inside the trace — a bare
            tensor or a tuple of them. Trailing non-tensor entries are dropped.

    Returns:
        A ``(logits, topk_weights, topk_indices)`` tuple. ``logits`` is always
        present; the other two are ``None`` when the router did not return
        them (both are re-derivable from ``logits``, which is what
        :func:`routing_features` falls back to).

    Raises:
        ValueError: If no candidate looks like router logits — i.e. no
            float tensor wide enough to be a per-expert score.
    """
    tensors = [t for t in _flatten(output) if isinstance(t, torch.Tensor)]

    logits = None
    indices = None
    weights = None
    k: Optional[int] = None

    for tensor in tensors:
        if tensor.dim() < 1 or tensor.shape[-1] < 2:
            continue
        if not tensor.is_floating_point():
            # Integer-valued and top-k-sized: the chosen-expert indices.
            if indices is None:
                indices = tensor
                k = int(tensor.shape[-1])
            continue
        width = _num_experts_from_logits(tensor)
        if logits is None or width > _num_experts_from_logits(logits):
            logits = tensor

    if logits is None:
        shapes = [tuple(t.shape) for t in tensors]
        raise ValueError(
            "No router logits found in router output. Expected a float tensor "
            f"of shape (tokens, experts); got shapes {shapes}"
        )

    # The float entry with top-k width (and not full expert width) is the
    # normalized routing weights. Checked after `logits` so a full-width
    # tensor never claims both roles.
    for tensor in tensors:
        if not tensor.is_floating_point() or tensor is logits:
            continue
        if tensor.dim() < 1:
            continue
        if k is not None and tensor.shape[-1] == k:
            weights = tensor
            break

    return logits, weights, indices


def _topk_from_logits(
    probs: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    topk_indices: Optional[torch.Tensor],
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized top-k probabilities and their expert ids.

    Prefers the router's own selection when it returned one, so this matches
    what the experts actually computed rather than a re-derivation.
    """
    if (
        topk_weights is not None
        and topk_indices is not None
        and int(topk_indices.shape[-1]) == k
    ):
        weights = topk_weights.float()
        total = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(total > 0, weights / total.clamp_min(1e-12), weights)
        return weights, topk_indices
    return torch.topk(probs, k, dim=-1)


def routing_features(
    router_output: Any,
    top_k: Optional[int] = None,
    layer: Optional[Union[int, str]] = None,
) -> Dict[str, torch.Tensor]:
    """Per-token routing statistics for one MoE layer's router output.

    Computes, per token:

    - ``router_entropy`` — Shannon entropy of the full softmax over experts.
      Low = the router committed hard to one expert. InnerExpert's headline
      routing signal.
    - ``topk_entropy`` — entropy of the normalized top-k routing weights.
      Near 0 = one expert dominates the selected set.
    - ``topk_margin`` — gap between the two strongest routing weights. The
      "expert disagreement" signal: a router that can't decide routes with a
      small margin.
    - ``topk_weight_mass`` — how much probability mass the selected experts
      carry (1.0 when ``norm_topk_prob`` renormalizes, < 1.0 otherwise).
    - ``expert_load`` — how many tokens this step routed to this token's
      top expert. A per-token view of the layer's expert-usage histogram:
      high where a token is being processed by a crowded expert.
    - ``top1_expert`` — id of the winning expert. The one integer column;
      useful as a grouping key when aggregating over a corpus.

    Args:
        router_output: The router module's ``.output`` inside a trace, in any
            layout (see :func:`normalize_router_output`).
        top_k: Routing width. Defaults to the router's own top-k when it
            returned indices, else 2.
        layer: Layer label folded into the returned keys. ``3`` produces
            ``"router_entropy@3"``; ``None`` leaves keys bare. Pass a layer id
            when collecting several layers so the dicts merge without
            colliding.

    Returns:
        Dict of named float tensors, each shaped like the router's token axis.
    """
    logits, weights, indices = normalize_router_output(router_output)

    if top_k is not None:
        k = top_k
    elif indices is not None:
        k = int(indices.shape[-1])
    else:
        k = 2
    k = max(2, min(k, logits.shape[-1]))

    probs = torch.softmax(logits.float(), dim=-1)
    router_entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

    topk_w, topk_i = _topk_from_logits(probs, weights, indices, k)

    topk_entropy = -(topk_w * torch.log(topk_w.clamp_min(1e-12))).sum(dim=-1)
    topk_margin = topk_w[..., 0] - topk_w[..., 1]
    topk_mass = topk_w.sum(dim=-1)

    flat_experts = topk_i.reshape(-1)
    usage = torch.bincount(flat_experts, minlength=logits.shape[-1])
    # Per-token: how crowded this token's winning expert is this step.
    expert_load = usage[topk_i[..., 0].reshape(-1)].reshape(topk_i[..., 0].shape)
    expert_load = expert_load.to(logits.dtype)

    features = {
        "router_entropy": router_entropy,
        "topk_margin": topk_margin,
        "topk_weight_mass": topk_mass,
        "topk_entropy": topk_entropy,
        "expert_load": expert_load,
        "top1_expert": topk_i[..., 0],
    }

    if layer is not None:
        features = {f"{name}@{layer}": value for name, value in features.items()}

    return features


def expert_usage(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """A layer's per-expert token counts from its :func:`routing_features`.

    Rebuilds the router's usage histogram from the ``top1_expert`` column.
    Useful as a per-layer balance metric: a collapsed router funnels most
    tokens to one expert, which is itself a routing-degenerate signal.

    Args:
        features: One layer's :func:`routing_features` output. Keys may carry
            a ``@layer`` suffix.

    Returns:
        ``[num_experts]`` tensor of token counts per expert.
    """
    for name, value in features.items():
        if name.split("@")[0] == "top1_expert":
            experts = value.to(torch.long).reshape(-1)
            return torch.bincount(experts, minlength=int(experts.max()) + 1)
    raise KeyError("no 'top1_expert' entry; pass an unfiltered routing_features() dict")


def find_routers(model: Envoy) -> List[Envoy]:
    """Every MoE router in an envoy tree, in module order.

    Matches on module shape rather than names so it works across
    architectures and backends: a router is a module named ``gate`` or
    ``router`` whose own weight (or that of a single inner projection) is
    ``[num_experts, hidden]`` — wider than it is tall — sitting directly
    under a module that also owns an experts container. That is exactly
    ``mlp.gate`` / ``block_sparse_moe.router`` in ``transformers`` MoE models
    and ``mlp.gate`` (a vLLM ``ReplicatedLinear``) in vLLM.

    Args:
        model: Any nnsight model — a root :class:`~nnsight.modeling.base.NNsight`,
            :class:`~nnsight.modeling.language.LanguageModel`,
            :class:`~nnsight.modeling.vllm.VLLM`, ...

    Returns:
        Envoy for each router, ordered by module path.
    """
    routers: List[Envoy] = []

    def _visit(envoy: Envoy) -> None:
        for child in envoy._children:
            if _is_router(child) and _has_expert_sibling(envoy):
                routers.append(child)
            _visit(child)

    _visit(model)
    return routers


def _router_weight(module: torch.nn.Module) -> Optional[torch.Tensor]:
    """The ``[num_experts, hidden]`` projection weight of a router module.

    ``transformers`` routers expose it directly; some wrap the projection in
    a child (vLLM's ``ReplicatedLinear`` under ``mlp.gate``).
    """
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    for child in module.children():
        weight = getattr(child, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight
    return None


def _is_router(envoy: Envoy) -> bool:
    """Is this Envoy a router: a wide ``[experts, hidden]`` projection?"""
    name = envoy.path.rsplit(".", 1)[-1]
    if name not in ("gate", "router"):
        return False
    module = envoy._module
    if not isinstance(module, torch.nn.Module):
        return False
    weight = _router_weight(module)
    if weight is None or weight.dim() != 2:
        return False
    num_experts, hidden = int(weight.shape[0]), int(weight.shape[1])
    # A real expert count (Qwen3-MoE 40/128, Mixtral 8, PowerMoE 40) that is
    # still smaller than the hidden dim, so tall head projections are out.
    return num_experts >= 4 and num_experts < hidden


def _has_expert_sibling(parent: Envoy) -> bool:
    """Does ``parent`` own an experts container alongside a router?

    Guards the name+shape heuristic: without it, a module that happens to
    call a wide projection ``gate`` (a gated dense MLP) would false-positive.
    """
    return any(
        "expert" in child.path.rsplit(".", 1)[-1].lower() for child in parent._children
    )
