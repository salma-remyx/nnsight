"""Routing-statistics readout for MoE models.

Turns the router logits a mixture-of-experts model already computes into a
small, interpretable per-token feature block: gate entropy, top-k margin,
top-k weight mass, and per-expert participation. The paper's J64 frame is
learned from reasoning states; its routing-side half (R64) is a ridge
regression on exactly these native statistics, which is what makes it
cheap enough to evaluate during generation. This module provides the
statistics half — the parameter-free signal — so a user can log them per
fire, aggregate them over a window, and fit whatever readout they want on
top, without instrumenting anything but ``<...>.mlp.gate.output``.

Adapted from "Beyond the Trace: Coupling an Interpretable Reasoning-State
Readout to Native MoE Routing" (arXiv:2608.17638). The paper's learned
J-space / ridge weights are model-specific artifacts and are intentionally
out of scope: this is the target-native statistics layer they consume.

Works on any trace value whose last dim is expert logits — the vLLM router
(``model.model.layers[i].mlp.gate.output``, full and identical on every
rank under both expert layouts) and the HF ``mlp.gate`` module alike.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import torch

__all__ = ["RoutingReadout", "collect_router_logits", "routing_features"]

# Ordering of the scalar block every layer contributes. Kept as a module
# constant so downstream consumers can index features by name.
SCALAR_FEATURES = ("entropy", "margin", "topk_mass", "spread")


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """View ``[..., n_experts]`` as ``[tokens, n_experts]``.

    Positional dims beyond the first (batch, ...) are folded into the token
    axis so the readout stays per-token regardless of how the caller batched.
    """
    if x.ndim == 0:
        return x.reshape(1, 1)
    return x.reshape(-1, x.shape[-1])


def routing_features(logits: torch.Tensor, top_k: int) -> Dict[str, torch.Tensor]:
    """Compute the routing statistics for one layer's router logits.

    Args:
        logits: Router logits with experts on the last dim. Extra leading
            dims are folded into a single token axis.
        top_k: The router's active-expert count (e.g. 8 for gpt-oss, 4 for
            Qwen3-MoE). Only the top ``top_k`` experts are routed to; the
            rest of the distribution is dead weight for routing purposes.

    Returns:
        Dict of named statistics, each ``[tokens]``:

        - ``probs`` — softmax over experts, ``[tokens, n_experts]``.
        - ``entropy`` — entropy of the full routing distribution, in nats.
          Low entropy = the router is committing to a small expert set.
        - ``margin`` — gap between the strongest and the k-th strongest
          expert logit. Wide margin = confident routing, no near-ties.
        - ``topk_mass`` — probability mass on the active ``top_k`` experts.
        - ``spread`` — number of active experts carrying non-negligible
          mass (each above a floor set just under the uniform 1/n), as a
          float tensor. 1.0 means one expert dominates; ``k`` means the
          load is evenly split across the active set.
    """
    x = _ensure_2d(logits).float()
    n_experts = x.shape[-1]
    k = max(1, min(int(top_k), n_experts))

    probs = torch.softmax(x, dim=-1)
    entropy = -(
        probs * torch.clamp(probs, min=torch.finfo(probs.dtype).eps).log()
    ).sum(dim=-1)

    sorted_logits, _ = torch.sort(x, dim=-1, descending=True)
    margin = sorted_logits[..., 0] - sorted_logits[..., k - 1]

    topk_probs, _ = torch.topk(probs, k, dim=-1)
    topk_mass = topk_probs.sum(dim=-1)
    # A perfectly even k-way split puts 1/n on each active expert; the floor
    # sits just under that so a uniform router scores k, not 0.
    floor = (1.0 / n_experts) - 1e-6
    spread = (topk_probs > floor).sum(dim=-1).float()

    return {
        "probs": probs,
        "entropy": entropy,
        "margin": margin,
        "topk_mass": topk_mass,
        "spread": spread,
    }


class RoutingReadout:
    """Aggregate routing statistics from one or more MoE layers.

    Feed it router logits inside a trace and it accumulates the scalar
    block per layer; read ``.vector`` (or ``.means()``) to get the
    concatenated feature vector the R64-style readout regresses against.

    Example (vLLM MoE model)::

        from nnsight.modeling.moe import RoutingReadout

        readout = RoutingReadout(top_k=8)
        with model.trace(prompt, temperature=0.0, max_tokens=32) as tracer:
            values = readout.attach(
                [model.model.layers[i].mlp.gate.output for i in (8, 16, 24)]
            )
            steps = list().save()
            for _ in tracer.iter[:]:
                steps.append(values[:])

        # steps[t][l] -> [entropy, margin, topk_mass, spread] per layer

    The object is plain Python; ``attach`` returns a list of per-layer
    statistic tensors that can be saved like any other trace value.
    """

    def __init__(self, top_k: int, layers: Optional[Sequence[str]] = None):
        self.top_k = int(top_k)
        self.layers: List[str] = list(layers) if layers is not None else []
        self._values: List[torch.Tensor] = []

    def attach(self, gate_outputs: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the scalar block for each layer's router logits.

        Args:
            gate_outputs: Router-logit tensors, one per MoE layer, in the
                order you want them in :attr:`vector`. Inside a trace these
                are the real tensors — ``mlp.gate.output`` resolves to the
                live value when the router fires.

        Returns:
            List of ``[tokens, 4]`` tensors, one per layer, stacked as
            ``[entropy, margin, topk_mass, spread]``.
        """
        self._values = []
        for gate_output in gate_outputs:
            feats = routing_features(gate_output, self.top_k)
            block = torch.stack(
                [feats[name] for name in SCALAR_FEATURES], dim=-1
            )
            self._values.append(block)
        return self._values

    @property
    def values(self) -> List[torch.Tensor]:
        """Per-layer statistic tensors, ``[tokens, 4]`` each."""
        return self._values

    def means(self) -> torch.Tensor:
        """Token-mean of each statistic per layer, ``[n_layers, 4]``.

        Requires :meth:`attach` to have run. Pooling over tokens matches how
        the paper aggregates a readout window before applying it.
        """
        if not self._values:
            raise RuntimeError("Nothing to aggregate — call attach() first.")
        return torch.stack([v.float().mean(dim=0) for v in self._values], dim=0)

    @property
    def vector(self) -> torch.Tensor:
        """Flattened readout vector, ``[n_layers * 4]``.

        The routing-side feature vector an R64-style ridge readout consumes:
        one entropy / margin / topk_mass / spread group per layer.
        """
        return self.means().flatten()

    def describe(self) -> str:
        """Human-readable per-layer summary of the current readout."""
        means = self.means()
        names = [f"L{i}" for i in range(means.shape[0])]
        if self.layers:
            names = [
                self.layers[i] if i < len(self.layers) else f"L{i}"
                for i in range(means.shape[0])
            ]
        rows = [
            f"  {name}: " + "  ".join(
                f"{feat}={means[i, j]:.4f}" for j, feat in enumerate(SCALAR_FEATURES)
            )
            for i, name in enumerate(names)
        ]
        return "RoutingReadout (token-mean)\n" + "\n".join(rows)


def collect_router_logits(
    gates: Iterable[torch.Tensor],
) -> List[torch.Tensor]:
    """Pull the raw router logits out of ``mlp.gate`` envoys, in order.

    A convenience for the common call site: reading ``.gate.output`` on each
    MoE layer inside a trace registers the access (the value resolves when
    the router fires) and returns the tensors ready for
    :meth:`RoutingReadout.attach`. Order of access matters inside a single
    invoke — pass the layers in forward-pass order.

    Args:
        gates: ``mlp.gate`` envoys (or any objects exposing ``.output``),
            in forward-pass order.

    Returns:
        List of the layers' ``.output`` values.
    """
    return [gate.output for gate in gates]
