"""Tests for MoE routing-signal extraction (nnsight.modeling.moe_routing).

Two layers:

- Layout tests against hand-built tensors covering the router return shapes
  the real architectures produce (Qwen3-MoE/Mixtral, GraniteMoe, bare-logits).
  These run anywhere, no model download.
- Integration tests driving a synthetic MoE through a real nnsight trace via
  ``NNsight``, and the real ``LanguageModel`` MoE path when weights are
  reachable.
"""

import pytest
import torch

from nnsight import NNsight
from nnsight.modeling import (
    expert_usage,
    find_routers,
    normalize_router_output,
    routing_features,
)
from nnsight.modeling.moe_routing import ROUTING_FEATURE_NAMES


# =============================================================================
# Synthetic MoE — mirrors the module shapes transformers MoE models build
# =============================================================================


class QwenStyleRouter(torch.nn.Module):
    """``(logits, scores, indices)`` — Qwen3-MoE / Mixtral ``*TopKRouter``."""

    def __init__(self, d=8, e=5, k=2):
        super().__init__()
        self.k = k
        self.lin = torch.nn.Linear(d, e)

    def forward(self, h):
        logits = self.lin(h).float()
        probs = torch.softmax(logits, -1)
        top_v, top_i = torch.topk(probs, self.k, -1)
        top_v = top_v / top_v.sum(-1, keepdim=True)
        return logits, top_v, top_i


class GraniteStyleRouter(torch.nn.Module):
    """``(indices, weights, logits)`` — GraniteMoe ``GraniteMoeTopKRouter``."""

    def __init__(self, d=8, e=5, k=2):
        super().__init__()
        self.k = k
        self.lin = torch.nn.Linear(d, e)

    def forward(self, h):
        logits = self.lin(h).float()
        top_l, top_i = torch.topk(logits, self.k, -1)
        top_w = torch.softmax(top_l, -1)
        return top_i, top_w, logits


class MoEBlock(torch.nn.Module):
    """A shape-faithful stand-in for a transformers MoE block.

    The point is the module *tree* — ``gate`` next to ``experts`` — and the
    router's return layout, so ``find_routers`` and ``normalize_router_output``
    are exercised against the shapes the real architectures produce. The
    expert math is a loop over the selected experts.
    """

    def __init__(self, d=8, e=5, router_cls=QwenStyleRouter):
        super().__init__()
        self.gate = router_cls(d, e)
        self.experts = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(e)])

    def forward(self, h):
        lead = h.shape[:-1]
        flat = h.reshape(-1, h.shape[-1])
        logits, weights, indices = self.gate(flat)

        out = torch.zeros_like(flat)
        for slot in range(weights.shape[1]):
            for expert, layer in enumerate(self.experts):
                mask = indices[:, slot] == expert
                if mask.any():
                    weighted = weights[:, slot].unsqueeze(-1) * layer(flat)
                    out = out + torch.where(mask.unsqueeze(-1), weighted, 0.0)
        return (flat + out).reshape(*lead, h.shape[-1])


class MoENet(torch.nn.Module):
    def __init__(self, d=8, e=5, n_layers=2, router_cls=QwenStyleRouter):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [MoEBlock(d, e, router_cls) for _ in range(n_layers)]
        )
        self.head = torch.nn.Linear(d, 3)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# =============================================================================
# normalize_router_output — layout recovery
# =============================================================================


class TestNormalizeRouterOutput:
    def test_qwen_style_logits_last(self):
        logits = torch.randn(4, 5)
        weights = torch.softmax(logits, -1)[:, :2]
        indices = torch.zeros(4, 2, dtype=torch.long)

        out_logits, out_w, out_i = normalize_router_output((logits, weights, indices))

        assert out_logits is logits
        assert out_w is weights
        assert out_i is indices

    def test_granite_style_logits_last_order_flipped(self):
        indices = torch.zeros(4, 2, dtype=torch.long)
        weights = torch.rand(4, 2)
        logits = torch.randn(4, 5)

        out_logits, out_w, out_i = normalize_router_output((indices, weights, logits))

        assert out_logits is logits
        assert out_i is indices
        assert out_w is weights

    def test_bare_logits_only(self):
        logits = torch.randn(4, 5)

        out_logits, out_w, out_i = normalize_router_output(logits)

        assert out_logits is logits
        assert out_w is None
        assert out_i is None

    def test_rejects_no_float_candidate(self):
        with pytest.raises(ValueError):
            normalize_router_output(torch.zeros(4, 2, dtype=torch.long))

    def test_prefers_widest_float_tensor_as_logits(self):
        # A tiny scalar-ish float must not shadow the real logit matrix.
        logits = torch.randn(4, 8)
        bias = torch.tensor(0.5)

        out_logits, _, _ = normalize_router_output((bias, logits))

        assert out_logits is logits


# =============================================================================
# routing_features — the per-token statistics
# =============================================================================


class TestRoutingFeatures:
    def _uniform_logits(self, tokens=3, experts=4):
        return torch.zeros(tokens, experts)

    def test_uniform_routing_gives_max_entropy(self):
        feats = routing_features(self._uniform_logits(), top_k=2)

        import math

        assert feats["router_entropy"][0] == pytest.approx(math.log(4), rel=1e-5)
        assert feats["topk_margin"][0] == pytest.approx(0.0, abs=1e-6)
        assert feats["topk_entropy"][0] == pytest.approx(math.log(2), rel=1e-5)

    def test_confident_routing_gives_low_entropy(self):
        logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]])

        feats = routing_features(logits, top_k=2)

        assert feats["router_entropy"][0] < 0.01
        assert feats["topk_margin"][0] > 0.99

    def test_feature_names_and_shapes_match_token_axis(self):
        logits = torch.randn(7, 6)

        feats = routing_features(logits, top_k=2)

        assert set(feats) == set(ROUTING_FEATURE_NAMES)
        for value in feats.values():
            assert value.shape == (7,)

    def test_layer_label_suffixes_keys(self):
        feats = routing_features(torch.randn(3, 6), top_k=2, layer=11)

        assert set(feats) == {f"{name}@11" for name in ROUTING_FEATURE_NAMES}

    def test_all_features_finite(self):
        logits = torch.randn(16, 9) * 50

        feats = routing_features(logits, top_k=3)

        for name, value in feats.items():
            assert torch.isfinite(value).all(), name

    def test_uses_router_selection_when_present(self):
        # A router that renormalized its top-k: weights sum to 1, so mass is 1.
        logits = torch.randn(5, 6)
        probs = torch.softmax(logits, -1)
        top_w, top_i = torch.topk(probs, 2, -1)
        top_w = top_w / top_w.sum(-1, keepdim=True)

        feats = routing_features((logits, top_w, top_i))

        assert torch.allclose(feats["topk_weight_mass"], torch.ones(5), atol=1e-5)

    def test_expert_usage_recovers_histogram(self):
        logits = torch.tensor([[10.0, 0.0], [10.0, 0.0], [0.0, 10.0]])

        feats = routing_features(logits, top_k=2)
        usage = expert_usage(feats)

        assert usage.shape == (2,)
        # Token 0 and 1 lead with expert 0, token 2 with expert 1.
        assert usage.argmax() == 0

    def test_expert_usage_raises_without_load(self):
        with pytest.raises(KeyError):
            expert_usage({"router_entropy": torch.zeros(3)})

    def test_entropy_ranks_mixed_confidence(self):
        sharp = torch.tensor([[12.0, 0.0, 0.0, 0.0]])
        flat = torch.zeros(1, 4)

        feats = routing_features(torch.cat([sharp, flat]), top_k=2)

        assert feats["router_entropy"][1] > feats["router_entropy"][0]


# =============================================================================
# find_routers + trace integration — exercises the real nnsight path
# =============================================================================


class TestFindRouters:
    def test_finds_all_layers_on_synthetic_moe(self):
        model = NNsight(MoENet())

        routers = find_routers(model)

        assert len(routers) == 2
        assert [r.path for r in routers] == [
            "model.blocks.0.gate",
            "model.blocks.1.gate",
        ]

    def test_ignores_dense_projections(self):
        # head/out projections are wide linears too; only gate-with-experts
        # siblings count.
        model = NNsight(MoENet())

        routers = find_routers(model)

        assert all(r.path.endswith(".gate") for r in routers)


class TestTraceIntegration:
    def test_features_inside_a_trace_are_real_tensors(self, device: str):
        model = NNsight(MoENet()).to(device)
        router = model.blocks[0].gate

        # NOTE: values must be saved via plain assignments, not comprehensions
        # — a comprehension body runs in its own frame, so its locals don't
        # propagate out of the trace (see docs/gotchas/save.md).
        entropy = None
        margin = None
        with model.trace(torch.rand(1, 5, 8, device=device)):
            feats = routing_features(router.output)
            entropy = feats["router_entropy"].save()
            margin = feats["topk_margin"].save()

        assert isinstance(entropy, torch.Tensor)
        assert entropy.shape == (5,)
        assert margin.shape == (5,)

    def test_multi_layer_collection_merges_without_collision(self, device: str):
        model = NNsight(MoENet()).to(device)

        layer0 = {}
        layer1 = {}
        with model.trace(torch.rand(1, 4, 8, device=device)):
            for i, router in enumerate(find_routers(model)):
                feats = routing_features(router.output, layer=i)
                target = layer0 if i == 0 else layer1
                for name in ROUTING_FEATURE_NAMES:
                    target[f"{name}@{i}"] = feats[f"{name}@{i}"].save()

        merged = {**layer0, **layer1}
        assert len(merged) == 2 * len(ROUTING_FEATURE_NAMES)
        # Every layer labels its own rows, so nothing collided away.
        assert merged["router_entropy@0"].shape == (4,)
        assert merged["router_entropy@1"].shape == (4,)

    def test_find_routers_returns_traceable_envoys(self, device: str):
        # What find_routers hands back must be directly usable as a trace
        # target — the whole point of the helper.
        model = NNsight(MoENet()).to(device)
        routers = find_routers(model)

        rows = []
        with model.trace(torch.rand(1, 3, 8, device=device)):
            for router in routers:
                feats = routing_features(router.output)
                rows.append(feats["router_entropy"].save())

        assert len(rows) == 2
        assert all(isinstance(row, torch.Tensor) for row in rows)


# =============================================================================
# Real HF MoE — the ``mlp.gate`` path Qwen3-MoE / Mixtral expose
# =============================================================================


TINY_MOE = "hf-internal-testing/tiny-random-Qwen3MoeForCausalLM"


@pytest.fixture(scope="module")
def moe(device: str):
    """A real (tiny, random-weight) Qwen3-MoE; skips when offline."""
    try:
        from nnsight import LanguageModel
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"LanguageModel unavailable: {exc}")

    try:
        return LanguageModel(TINY_MOE, device_map=device, dispatch=True)
    except Exception as exc:
        pytest.skip(f"could not fetch {TINY_MOE}: {exc}")


class TestRealMoEModel:
    """End-to-end against a real Qwen3-MoE through ``LanguageModel``."""

    def test_find_routers_hits_mlp_gate(self, moe):
        routers = find_routers(moe)

        assert len(routers) == 2
        assert all(r.path.endswith(".mlp.gate") for r in routers)

    def test_features_match_token_count(self, moe):
        prompt = "The Eiffel Tower is in"
        n_tokens = len(moe.tokenizer(prompt).input_ids)
        router = find_routers(moe)[0]

        entropy = None
        with moe.trace(prompt):
            feats = routing_features(router.output)
            entropy = feats["router_entropy"].save()

        assert entropy.shape == (n_tokens,)
        assert torch.isfinite(entropy).all()
