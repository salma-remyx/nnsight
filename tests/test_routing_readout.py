"""
Tests for the MoE routing-statistics readout.

Covers the statistics themselves (against hand-computed values) and the
integration: the readout is exported from ``nnsight.modeling`` and runs
inside a real ``NNsight`` trace on a tiny MoE-shaped module, reading the
same ``<...>.mlp.gate.output`` surface the vLLM MoE path exposes.
"""

import pytest
import torch
from torch import nn

import nnsight
from nnsight import NNsight
from nnsight.modeling import RoutingReadout, routing_features


# =============================================================================
# Fixtures — a tiny MoE block so the tests run on CPU without vllm
# =============================================================================


class TinyMoE(nn.Module):
    """Minimal MoE stack: two layers, each a router over 6 experts.

    Mirrors the module surface the readout targets: ``layers[i].mlp.gate``
    holding the router logits. The expert compute itself is a stub — the
    readout only consumes ``gate.output``.
    """

    class Layer(nn.Module):
        def __init__(self, d_model: int, n_experts: int):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.gate = nn.Linear(d_model, n_experts)

        def forward(self, x):
            # Route to the top-2 experts by weight (output unused beyond
            # keeping the graph alive).
            logits = self.mlp.gate(x)
            return logits.softmax(dim=-1)[..., :2].sum(dim=-1, keepdim=True) * x

    def __init__(self, d_model: int = 8, n_layers: int = 2, n_experts: int = 6):
        super().__init__()
        self.layers = nn.ModuleList(
            [self.Layer(d_model, n_experts) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture(scope="module")
def moe_model(device: str):
    """Tiny two-layer MoE wrapped with NNsight."""
    torch.manual_seed(0)
    net = TinyMoE()
    # Deterministic routing: give each gate a distinct, fixed bias so the
    # softmax is far from uniform without being one-hot.
    with torch.no_grad():
        for i, layer in enumerate(net.layers):
            layer.mlp.gate.bias.copy_(torch.linspace(0.5, 2.5, 6) * (1 + i))
    return NNsight(net).to(device)


@pytest.fixture
def moe_input(device: str):
    return torch.rand(1, 3, 8, device=device)


# =============================================================================
# Statistics — hand-computed expectations
# =============================================================================


class TestRoutingFeatures:
    def test_entropy_uniform_is_log_nexperts(self):
        logits = torch.zeros(4, 8)
        feats = routing_features(logits, top_k=2)
        assert feats["entropy"].shape == (4,)
        assert torch.allclose(feats["entropy"], torch.log(torch.tensor(8.0)), atol=1e-5)

    def test_entropy_one_hot_is_zero(self):
        logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
        feats = routing_features(logits, top_k=2)
        assert feats["entropy"][0].item() == pytest.approx(0.0, abs=1e-4)

    def test_margin_is_top_minus_kth(self):
        logits = torch.tensor([[5.0, 4.0, 3.0, 1.0]])
        feats = routing_features(logits, top_k=2)
        # top logit 5.0, k-th (2nd) logit 4.0 -> margin 1.0
        assert feats["margin"][0].item() == pytest.approx(1.0)

    def test_topk_mass_sums_selected_experts(self):
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
        feats = routing_features(logits, top_k=2)
        probs = feats["probs"][0]
        assert feats["topk_mass"][0].item() == pytest.approx(
            (probs[0] + probs[1]).item()
        )
        assert feats["topk_mass"][0].item() <= 1.0

    def test_spread_counts_contributing_experts(self):
        # One dominant expert -> spread 1.
        sharp = routing_features(torch.tensor([[10.0, 0.0, 0.0, 0.0]]), top_k=2)
        assert sharp["spread"][0].item() == pytest.approx(1.0)
        # Uniform logits -> all k experts above the 1/(2k) floor.
        flat = routing_features(torch.zeros(1, 4), top_k=2)
        assert flat["spread"][0].item() == pytest.approx(2.0)

    def test_top_k_clamped_to_n_experts(self):
        # Asking for more experts than exist must not raise.
        feats = routing_features(torch.zeros(3, 4), top_k=100)
        assert feats["probs"].shape == (3, 4)
        assert torch.allclose(feats["topk_mass"], torch.ones(3), atol=1e-5)

    def test_extra_leading_dims_fold_into_tokens(self):
        logits = torch.zeros(2, 5, 7)  # [batch, seq, experts]
        feats = routing_features(logits, top_k=2)
        assert feats["entropy"].shape == (10,)


# =============================================================================
# Readout aggregation
# =============================================================================


class TestRoutingReadout:
    def test_attach_returns_one_block_per_layer(self):
        readout = RoutingReadout(top_k=2)
        blocks = readout.attach([torch.randn(3, 6), torch.randn(3, 6)])
        assert len(blocks) == 2
        for block in blocks:
            assert block.shape == (3, 4)

    def test_means_pools_over_tokens(self):
        logits = torch.zeros(1, 4)  # uniform over 4 experts
        readout = RoutingReadout(top_k=2)
        readout.attach([logits])
        means = readout.means()
        assert means.shape == (1, 4)
        assert means[0, 0].item() == pytest.approx(torch.log(torch.tensor(4.0)).item())

    def test_vector_length_is_layers_times_four(self):
        readout = RoutingReadout(top_k=2)
        readout.attach([torch.randn(5, 6) for _ in range(3)])
        assert readout.vector.shape == (12,)

    def test_means_before_attach_raises(self):
        with pytest.raises(RuntimeError):
            RoutingReadout(top_k=2).means()

    def test_describe_lists_every_layer(self):
        readout = RoutingReadout(top_k=2, layers=["layers.0.mlp.gate"])
        readout.attach([torch.randn(2, 6)])
        text = readout.describe()
        assert "layers.0.mlp.gate" in text
        for feat in ("entropy", "margin", "topk_mass", "spread"):
            assert feat in text

    def test_confident_routing_lowers_entropy(self):
        readout = RoutingReadout(top_k=2)
        readout.attach([torch.tensor([[10.0, 0.0, 0.0, 0.0]])])
        confident = readout.means()[0, 0].item()
        readout.attach([torch.zeros(1, 4)])
        uniform = readout.means()[0, 0].item()
        assert confident < uniform


# =============================================================================
# Integration — readout inside a real nnsight trace
# =============================================================================


class TestTraceIntegration:
    def test_readout_runs_inside_trace_on_gate_output(self, moe_model, moe_input):
        """The wired call site: gate.output read through the tracing layer."""
        readout = RoutingReadout(top_k=2)

        with torch.no_grad():
            with moe_model.trace(moe_input):
                gates = [
                    moe_model.layers[i].mlp.gate.output for i in range(2)
                ]
                blocks = nnsight.save(readout.attach(gates))
                vector = nnsight.save(readout.vector)

        assert len(blocks) == 2
        for block in blocks:
            assert isinstance(block, torch.Tensor)
            assert block.shape == (3, 4)  # [tokens, scalar features]
        assert vector.shape == (8,)  # 2 layers x 4 scalars
        # Entropy is finite and positive on this non-degenerate routing.
        assert torch.isfinite(vector).all()
        assert (blocks[0][:, 0] > 0).all()

    def test_readout_tracks_a_routed_perturbation(self, moe_model, moe_input):
        """Pushing the router toward one expert must move the readout.

        The entropy / topk_mass coordinates are the ones the paper's
        routing-side readout leans on; a large negative bias on all-but-one
        expert drops entropy and raises topk_mass together.

        Modules are visited in forward-pass order, one visit each — reading
        a layer's gate again after a later layer has fired would be an
        out-of-order access.
        """
        base_readout = RoutingReadout(top_k=2)
        steered_readout = RoutingReadout(top_k=2)

        with torch.no_grad():
            with moe_model.trace(moe_input):
                gates = [moe_model.layers[i].mlp.gate.output for i in range(2)]
                nnsight.save(base_readout.attach(gates))
                base = nnsight.save(base_readout.vector)

            with moe_model.trace(moe_input):
                steered_blocks = []
                for i in range(2):
                    gate = moe_model.layers[i].mlp.gate
                    # Collapse the routing onto expert 0.
                    gate.output[..., 1:] -= 50.0
                    steered_readout.attach([gate.output])
                    steered_blocks.append(steered_readout.values[0])
                steered = nnsight.save(
                    torch.stack(
                        [b.float().mean(dim=0) for b in steered_blocks], dim=0
                    ).flatten()
                )

        # Each layer contributes [entropy, margin, topk_mass, spread]; layer 0
        # occupies the first four slots.
        assert steered[0] < base[0]          # entropy drops
        assert steered[2] > base[2]          # topk_mass rises
        assert steered[3] < base[3]          # spread collapses to ~1

    def test_collect_router_logits_reads_gate_envoys(self, moe_model, moe_input):
        from nnsight.modeling import collect_router_logits

        with torch.no_grad():
            with moe_model.trace(moe_input):
                logits = nnsight.save(
                    collect_router_logits(
                        [moe_model.layers[i].mlp.gate for i in range(2)]
                    )
                )

        assert len(logits) == 2
        for value in logits:
            assert isinstance(value, torch.Tensor)
            assert value.shape[-1] == 6  # n_experts on the last dim
