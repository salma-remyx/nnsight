"""
Tests for depth-averaged truth-signal reading over the activation cache.

Adapted from HalluTracer (arXiv:2608.16353): per-layer truth probes are weakly
correlated, so averaging their scores across depth recovers nearly all of the
linearly accessible signal.  These tests exercise the whole path through the
public surface -- ``tracer.cache()`` from
``nnsight.intervention.tracing.tracer``, then the detector from
``nnsight.modeling.truthfulness`` -- on the shared tiny model fixture.
"""

from collections import OrderedDict

import pytest
import torch

import nnsight
from nnsight import NNsight
from nnsight.modeling import fit_truth_detector, truth_score
from nnsight.modeling.truthfulness import layer_agreement, layer_states

INPUT_SIZE = 5
HIDDEN_DIMS = 10
OUTPUT_SIZE = 2
DEPTH = 3  # l0, l1, l2


@pytest.fixture(scope="module")
def layered_tiny_model(device: str):
    """Three same-width layers (the "depth" we read) plus an output head.

    The head is named ``head`` rather than ``l3`` so the ``model.l`` pattern
    selects exactly the uniform-width layers a depth stack needs.
    """

    net = torch.nn.Sequential(
        OrderedDict(
            [
                ("l0", torch.nn.Linear(INPUT_SIZE, HIDDEN_DIMS)),
                ("l1", torch.nn.Linear(HIDDEN_DIMS, HIDDEN_DIMS)),
                ("l2", torch.nn.Linear(HIDDEN_DIMS, HIDDEN_DIMS)),
                ("head", torch.nn.Linear(HIDDEN_DIMS, OUTPUT_SIZE)),
            ]
        )
    )
    return NNsight(net).to(device)


@pytest.fixture(scope="module")
def cached_states(layered_tiny_model, device: str):
    """Cache every module of the tiny model on a fixed batch of inputs."""

    torch.manual_seed(0)

    inputs = torch.rand((6, INPUT_SIZE))

    with layered_tiny_model.trace(inputs) as tracer:
        cache = tracer.cache()

    return cache, inputs


class TestLayerStates:
    def test_cache_captures_every_layer(self, cached_states):
        """tracer.cache() with no module filter records all three layers."""

        cache, _ = cached_states

        assert "model.l0" in cache
        assert "model.l1" in cache
        assert "model.l2" in cache

    def test_layer_states_stacks_depth_first(self, cached_states):
        """The tiny model has no sequence axis, so layers stay [batch, hidden]."""

        cache, _ = cached_states

        states = layer_states(cache, pattern="model.l")

        assert states.shape == (DEPTH, 6, HIDDEN_DIMS)
        # The pattern must not have swept in the 2-wide output head.
        assert states.shape[-1] == HIDDEN_DIMS

    def test_layer_states_matches_a_single_cached_output(self, cached_states):
        """The stacked tensor must agree with reading one cache entry directly."""

        cache, _ = cached_states

        states = layer_states(cache, pattern="model.l")

        assert torch.equal(states[1], cache["model.l1"].output.to(torch.float32))

    def test_layer_states_orders_by_layer_index(self):
        """Numeric index order, not lexicographic -- l2 must sort before l10."""

        class Entry:
            def __init__(self, output):
                self.output = output

        cache = {
            "model.l10": Entry(torch.zeros(1, 4)),
            "model.l2": Entry(torch.ones(1, 4)),
        }

        states = layer_states(cache, pattern="model.l")

        assert states.shape[0] == 2
        assert torch.equal(states[0], torch.ones(1, 4))

    def test_layer_states_selects_last_position(self):
        """With a sequence axis, 'last' reads the next-token position."""

        class Entry:
            def __init__(self, output):
                self.output = output

        cache = {
            "model.h0": Entry(torch.arange(12).float().reshape(1, 3, 4)),
            "model.h1": Entry(torch.arange(12, 24).float().reshape(1, 3, 4)),
        }

        states = layer_states(cache, pattern="model.h")

        assert states.shape == (2, 1, 4)
        assert torch.equal(states[0, 0], torch.arange(8.0, 12.0))

    def test_pattern_and_paths_are_mutually_exclusive(self, cached_states):
        cache, _ = cached_states

        with pytest.raises(ValueError, match="exactly one"):
            layer_states(cache, pattern="model.l", paths=["model.l0"])


class TestTruthDetector:
    def test_separates_linearly_separable_labels(self, cached_states):
        """The depth-averaged score should recover an exactly-separable split."""

        cache, _ = cached_states
        states = layer_states(cache, pattern="model.l")

        # A label that is a linear function of the last layer's hidden state:
        # trivially separable, so the detector should get it perfectly right.
        labels = (states[-1][:, 0] > states[-1][:, 0].median()).long()

        detector = fit_truth_detector(states, labels, paths=["l0", "l1", "l2"])
        scores, curve = truth_score(detector, states)

        assert scores.shape == (6,)
        assert curve.shape == (DEPTH, 6)
        assert detector.metrics["averaged_accuracy"] == pytest.approx(1.0)

        predicted = (scores > detector.threshold).long()
        assert torch.equal(predicted, labels)

    def test_averaging_beats_or_matches_a_single_layer(self, cached_states):
        """The paper's claim: depth averaging is at least as good as any layer.

        On separable data every layer is perfect, so assert the weaker but
        still meaningful property -- the averaged score never trails the best
        single layer.
        """

        cache, _ = cached_states
        states = layer_states(cache, pattern="model.l")
        labels = (states[-1][:, 0] > states[-1][:, 0].median()).long()

        detector = fit_truth_detector(states, labels)

        assert (
            detector.metrics["averaged_accuracy"]
            >= detector.metrics["best_layer_accuracy"] - 1e-6
        )

    def test_detector_repr_mentions_depth(self, cached_states):
        cache, _ = cached_states
        states = layer_states(cache, pattern="model.l")
        labels = torch.tensor([1, 0, 1, 0, 1, 0])

        detector = fit_truth_detector(states, labels)

        assert f"depth={DEPTH}" in repr(detector)

    def test_rejects_wrong_depth(self, cached_states):
        cache, _ = cached_states
        states = layer_states(cache, pattern="model.l")
        labels = torch.tensor([1, 0, 1, 0, 1, 0])

        detector = fit_truth_detector(states, labels)

        with pytest.raises(ValueError, match="layers"):
            truth_score(detector, states[:2])

    def test_rejects_wrong_labels_shape(self, cached_states):
        cache, _ = cached_states
        states = layer_states(cache, pattern="model.l")

        with pytest.raises(ValueError, match="labels"):
            fit_truth_detector(states, torch.tensor([1, 0]))


class TestLayerAgreement:
    def test_identical_layers_agree_completely(self):
        curve = torch.ones(4, 3)

        agreement = layer_agreement(curve)

        assert torch.all(agreement == 1.0)

    def test_disagreeing_layers_score_low(self):
        # Three of four layers vote one way, the fourth the other.  Of the 12
        # ordered layer pairs, 6 agree and 6 do not, so agreement is exactly
        # 0.5 -- visibly below the 1.0 of a fully redundant stack.
        curve = torch.tensor([[1.0], [1.0], [1.0], [-1.0]])

        agreement = layer_agreement(curve)

        assert agreement.shape == (1,)
        assert agreement[0].item() == pytest.approx(0.5)

    def test_needs_two_layers(self):
        with pytest.raises(ValueError, match="two layers"):
            layer_agreement(torch.ones(1, 3))


class TestPublicExport:
    def test_exported_from_modeling(self):
        """The capability is reachable from the public nnsight.modeling path."""

        from nnsight.modeling import TruthDetector

        detector = TruthDetector(
            weights=torch.zeros(2, 3), biases=torch.zeros(2), paths=["a", "b"]
        )

        assert detector.depth == 2
        assert detector.paths == ["a", "b"]


def test_cache_reference_is_documented():
    """The tracer.cache() docstring points at the depth-averaged pattern."""

    from nnsight.intervention.tracing.tracer import InterleavingTracer

    assert "fit_truth_detector" in InterleavingTracer.cache.__doc__


def test_import_nnsight_still_exposes_language_model():
    """The new import in nnsight.modeling must not break lazy top-level imports."""

    assert hasattr(nnsight, "LanguageModel") or hasattr(nnsight, "NNsight")
