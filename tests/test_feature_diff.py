"""
Tests for feature-level model diffing and control.

Covers:
- ``feature_diff.feature_activation_diff`` (feature isolation)
- ``feature_diff.contrastive_feature_scores`` (task-specific detection)
- ``feature_diff.steer_features`` (feature-level control)
- ``VisionLanguageModel.diff_features`` wiring (dual trace -> FeatureDiff)
"""

import pytest
import torch
import nnsight

from nnsight.intervention.feature_diff import (
    FeatureDiff,
    contrastive_feature_scores,
    feature_activation_diff,
    steer_features,
)


# =============================================================================
# Feature isolation: diffing SAE activations between two models
# =============================================================================


class TestFeatureActivationDiff:
    def test_ranks_changed_features(self):
        """Features with the largest activation change rank highest."""
        source = torch.zeros(4, 8)
        target = torch.zeros(4, 8)
        target[:, 3] = 2.0  # feature 3 gained activation in the target model
        target[:, 6] = -1.0  # feature 6 lost activation

        result = feature_activation_diff(source, target)

        assert isinstance(result, FeatureDiff)
        assert result.diff.shape == (8,)
        top = result.topk(1)
        assert top.indices.item() == 3
        assert top.values.item() == pytest.approx(2.0)
        bottom = result.topk(1, largest=False)
        assert bottom.indices.item() == 6

    def test_firing_rates(self):
        """Firing rates reflect the fraction of positions with activation > 0."""
        source = torch.zeros(2, 4)
        target = torch.zeros(2, 4)
        target[0, 1] = 5.0  # fires on 1/2 positions

        result = feature_activation_diff(source, target)

        assert result.target_rate[1].item() == pytest.approx(0.5)
        assert result.source_rate[1].item() == pytest.approx(0.0)

    def test_batched_and_sequential_dims_flatten(self):
        """[batch, seq, n_features] inputs reduce over batch and sequence."""
        source = torch.ones(2, 3, 5)
        target = torch.ones(2, 3, 5) * 3

        result = feature_activation_diff(source, target)

        assert result.diff.shape == (5,)
        assert torch.allclose(result.diff, torch.full((5,), 2.0))

    def test_mismatched_feature_dims_raise(self):
        with pytest.raises(ValueError, match="feature dims differ"):
            feature_activation_diff(torch.zeros(4, 8), torch.zeros(4, 9))

    def test_contrastive_scores_match_positive_minus_negative(self):
        """Contrastive detection diffs the positive set against the negative set."""
        positive = torch.zeros(3, 6)
        negative = torch.zeros(3, 6)
        positive[:, 2] = 4.0  # task-specific feature fires only on positives

        result = contrastive_feature_scores(positive, negative)

        assert result.diff[2].item() == pytest.approx(4.0)
        assert result.target_rate[2].item() == pytest.approx(1.0)
        assert result.source_rate[2].item() == pytest.approx(0.0)


# =============================================================================
# Feature-level control: removing and steering feature directions
# =============================================================================


class TestSteerFeatures:
    def test_steer_adds_direction(self):
        hidden = torch.zeros(1, 2, 4)
        directions = torch.eye(6, 4)  # 6 features, d_model=4

        out = steer_features(hidden, directions, indices=[0, 2], alpha=2.0)

        expected = 2.0 * (directions[0] + directions[2])
        assert torch.allclose(out, expected.expand_as(hidden))
        # Original hidden states are untouched (new tensor returned).
        assert torch.all(hidden == 0)

    def test_remove_subtracts_feature_contribution(self):
        hidden = torch.zeros(1, 3, 4)
        directions = torch.eye(5, 4)
        features = torch.zeros(1, 3, 5)
        features[..., 1] = 3.0  # feature 1 active with strength 3

        out = steer_features(
            hidden, directions, indices=1, features=features, mode="remove"
        )

        assert torch.allclose(out, hidden - 3.0 * directions[1].expand_as(hidden))

    def test_remove_requires_features(self):
        with pytest.raises(ValueError, match="requires feature activations"):
            steer_features(
                torch.zeros(1, 4), torch.eye(5, 4), indices=0, mode="remove"
            )

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown mode"):
            steer_features(torch.zeros(1, 4), torch.eye(5, 4), indices=0, mode="zero")


# =============================================================================
# VisionLanguageModel.diff_features wiring
# =============================================================================


class TestDiffFeaturesWiring:
    """Exercise the VisionLanguageModel.diff_features call site end to end.

    ``diff_features`` only relies on ``.trace()`` and attribute resolution,
    so a LanguageModel instance (gpt2) can stand in for the VLM: GPT-2
    blocks return tuple outputs, which also covers the tuple-unwrap branch.
    """

    @torch.no_grad()
    def test_diff_features_against_self_is_zero(self, gpt2: nnsight.LanguageModel):
        """Diffing a model against itself yields a zero diff with correct shape.

        Uses the identity map as the encoder so "features" are the hidden
        dimensions themselves; the wiring (dual trace, path resolution,
        tuple handling, FeatureDiff assembly) is what is under test.
        """
        from nnsight.modeling.vlm import VisionLanguageModel

        result = VisionLanguageModel.diff_features(
            gpt2,
            gpt2,
            "The Eiffel Tower is located in the city of",
            encoder=lambda hidden: hidden,
            layer="transformer.h.0",
        )

        assert isinstance(result, FeatureDiff)
        d_model = result.diff.shape[0]
        assert d_model == gpt2.config.n_embd
        assert torch.allclose(result.diff, torch.zeros(d_model), atol=1e-5)
        assert torch.allclose(result.source_mean, result.target_mean, atol=1e-5)

    @torch.no_grad()
    def test_diff_features_detects_planted_change(self, gpt2: nnsight.LanguageModel):
        """A planted per-feature offset shows up as the top-ranked diff."""
        with gpt2.trace("The Eiffel Tower is located in the city of"):
            hidden = gpt2.transformer.h[0].output[0].save()

        def boosted_encoder(h):
            out = h.clone()
            out[..., 0] += 10.0  # plant a change in feature 0
            return out

        result = feature_activation_diff(hidden, boosted_encoder(hidden))

        top = result.topk(1)
        assert top.indices.item() == 0
        assert top.values.item() == pytest.approx(10.0)

    @torch.no_grad()
    def test_diff_features_on_vlm(self, request):
        """Real VLM self-diff, text-only (needs the optional image stack)."""
        pytest.importorskip("PIL", reason="VLM processor requires Pillow")
        vlm = request.getfixturevalue("vlm")
        result = vlm.diff_features(
            vlm,
            "The Eiffel Tower is located in the city of",
            encoder=lambda hidden: hidden,
            layer="model.language_model.layers.0",
        )

        assert isinstance(result, FeatureDiff)
        assert torch.allclose(result.diff, torch.zeros_like(result.diff), atol=1e-5)

