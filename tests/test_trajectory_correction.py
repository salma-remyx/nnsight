"""
Tests for mid-generation trajectory correction.

These cover the TrajectoryCorrector controller and its drift supervisors
(``cosine_drift``, ``token_overlap_drift``), exported from
``nnsight.modeling``:

- Public export surface (integration with the wiring in
  ``src/nnsight/modeling/__init__.py``)
- Supervisor scoring semantics
- Correction gating, step decay, direction, and norm clamping
- End-to-end use inside a real ``NNsight`` trace: assess + correct across
  an iterated latent state via ``for step in tracer.iter[...]``

The end-to-end tests use a small recurrent ``nn.Module`` (see
``tests/test_iter_edge_cases.py`` for the same harness) so they run on
CPU with no downloads. The diffusion call-site itself (``DiffusionModel``
iterating over ``unet.output[0]``) is exercised in
``docs/patterns/trajectory-correction.md``; it needs ``diffusers`` and is
covered by the existing diffusion suite.
"""

import pytest
import torch
import torch.nn as nn

import nnsight
from nnsight import NNsight
from nnsight.modeling import (
    DriftReport,
    TrajectoryCorrector,
    cosine_drift,
    token_overlap_drift,
)


# ---------------------------------------------------------------------------
# Test harness — recurrent module, its inner submodule fires N times per trace
# ---------------------------------------------------------------------------


class RecurrentInner(nn.Module):
    """A module whose forward calls an inner submodule in sequence.

    Wrapped with NNsight, each call to ``self.linear`` is one iteration of
    ``tracer.iter[...]`` — the same shape as a diffusion denoiser firing
    once per scheduler step.
    """

    def __init__(self, dim: int = 4, inner_calls: int = 4):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.inner_calls = inner_calls

    def forward(self, x):
        for _ in range(self.inner_calls):
            x = self.linear(x)
        return x


@pytest.fixture
def recurrent_model(device: str):
    torch.manual_seed(0)
    return NNsight(RecurrentInner(dim=4, inner_calls=4)).to(device)


@pytest.fixture
def small_input():
    torch.manual_seed(0)
    return torch.randn(1, 4)


# ---------------------------------------------------------------------------
# Export surface
# ---------------------------------------------------------------------------


class TestExportSurface:
    """The capability is importable from nnsight.modeling (the wiring edit)."""

    def test_names_exported_from_modeling(self):
        for name in (
            "DriftReport",
            "TrajectoryCorrector",
            "cosine_drift",
            "token_overlap_drift",
        ):
            assert hasattr(nnsight.modeling, name)

    def test_module_imports_without_optional_deps(self):
        """The capability has no diffusers/transformers import of its own."""
        from nnsight.modeling import trajectory_correction

        assert trajectory_correction.__all__


# ---------------------------------------------------------------------------
# Supervisors
# ---------------------------------------------------------------------------


class TestCosineDrift:
    """``cosine_drift`` reports 1 - cos(state, reference)."""

    def test_identical_state_is_zero_drift(self):
        state = torch.randn(2, 3)
        supervisor = cosine_drift(reference=state)
        assert supervisor(state, step=0).drift == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_state_is_unit_drift(self):
        reference = torch.tensor([1.0, 0.0])
        orthogonal = torch.tensor([0.0, 5.0])  # direction matters, not scale
        supervisor = cosine_drift(reference=reference)
        assert supervisor(orthogonal, step=0).drift == pytest.approx(1.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        supervisor = cosine_drift(reference=torch.zeros(3))
        with pytest.raises(ValueError, match="does not match reference shape"):
            supervisor(torch.zeros(4), step=0)

    def test_accepts_tuple_states(self):
        """Multi-output modules (UNet-style tuples) score by concatenation."""
        reference = (torch.ones(2), torch.zeros(2))
        supervisor = cosine_drift(reference=reference)
        report = supervisor((torch.ones(2), torch.zeros(2)), step=0)
        assert report.drift == pytest.approx(0.0, abs=1e-6)


class TestTokenOverlapDrift:
    """``token_overlap_drift`` reports 1 - prompt-word coverage."""

    def test_full_overlap_is_zero_drift(self):
        supervisor = token_overlap_drift(
            prompt="A cat", describe=lambda state: "a CAT sitting"
        )
        assert supervisor(torch.zeros(1), step=0).drift == pytest.approx(0.0)

    def test_no_overlap_is_unit_drift(self):
        supervisor = token_overlap_drift(
            prompt="A cat", describe=lambda state: "a dog"
        )
        assert supervisor(torch.zeros(1), step=0).drift == pytest.approx(1.0)

    def test_partial_overlap(self):
        supervisor = token_overlap_drift(
            prompt="a red panda eating bamboo",
            describe=lambda state: "red panda sleeping",
        )
        # Content words: red, panda, eating, bamboo -> 2 of 4 present.
        assert supervisor(torch.zeros(1), step=0).drift == pytest.approx(0.5)

    def test_stopwords_are_dropped(self):
        supervisor = token_overlap_drift(
            prompt="the cat with the hat", describe=lambda state: "cat hat"
        )
        assert supervisor(torch.zeros(1), step=0).drift == pytest.approx(0.0)

    def test_default_decoder_describes_tensor(self):
        """The shipped stand-in decoder is deterministic in the state."""
        supervisor = token_overlap_drift(prompt="anything")
        flat = torch.linspace(0.0, 1.0, 32)
        first = supervisor(flat, step=0).info["description"]
        second = supervisor(flat.clone(), step=0).info["description"]
        assert first == second
        assert first  # non-empty for a non-degenerate tensor

    def test_empty_prompt_is_zero_drift(self):
        supervisor = token_overlap_drift(prompt="", describe=lambda state: "cat")
        assert supervisor(torch.zeros(1), step=0).drift == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Controller validation
# ---------------------------------------------------------------------------


class TestDriftReport:
    def test_rejects_negative_drift(self):
        with pytest.raises(ValueError, match="non-negative"):
            DriftReport(drift=-0.1)

    def test_rejects_out_of_range_weight(self):
        with pytest.raises(ValueError, match=r"weight must be in \[0, 1\]"):
            DriftReport(drift=0.5, weight=1.5)

    def test_info_defaults_to_empty_dict(self):
        assert DriftReport(drift=0.5).info == {}


class TestCorrectorValidation:
    def test_rejects_bad_total_steps(self):
        with pytest.raises(ValueError, match="total_steps"):
            TrajectoryCorrector(scorer=cosine_drift(torch.zeros(2)), total_steps=0)

    def test_rejects_bad_direction(self):
        with pytest.raises(ValueError, match="direction"):
            TrajectoryCorrector(
                scorer=cosine_drift(torch.zeros(2)), total_steps=4, direction="up"
            )

    def test_correct_before_assess_raises(self):
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(torch.zeros(2)), total_steps=4
        )
        with pytest.raises(RuntimeError, match="before assess"):
            corrector.correct(torch.zeros(2), step=0)


class TestDecay:
    """Step-decayed gain: 1 at step 0, 0 at total_steps."""

    def test_decay_endpoints(self):
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(torch.zeros(2)), total_steps=10
        )
        assert corrector.decay(0) == pytest.approx(1.0)
        assert corrector.decay(10) == pytest.approx(0.0)

    def test_decay_is_monotone_nonincreasing(self):
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(torch.zeros(2)), total_steps=10
        )
        values = [corrector.decay(step) for step in range(11)]
        assert all(a >= b for a, b in zip(values, values[1:]))

    def test_decay_rejects_negative_step(self):
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(torch.zeros(2)), total_steps=4
        )
        with pytest.raises(ValueError, match="non-negative"):
            corrector.decay(-1)


class TestCorrectionBehavior:
    """Gating, direction, dtype preservation, and the norm cap."""

    def _corrector(self, anchor, **kwargs):
        return TrajectoryCorrector(
            scorer=cosine_drift(reference=anchor),
            total_steps=10,
            anchor=anchor,
            **kwargs,
        )

    def test_zero_drift_returns_state_unchanged(self):
        state = torch.randn(2, 2)
        corrector = self._corrector(anchor=state)
        corrector.assess(state, step=0)
        out = corrector.correct(state, step=0)
        assert out is state  # identity, not a modified copy

    def test_threshold_gates_correction(self):
        state = torch.randn(2, 2)
        anchor = -state  # max drift
        corrector = self._corrector(anchor=anchor, threshold=2.0)
        corrector.assess(state, step=0)
        assert corrector.correct(state, step=0) is state

    def test_correction_moves_toward_anchor(self):
        state = torch.tensor([[1.0, 0.0]])
        anchor = torch.tensor([[0.0, 1.0]])
        corrector = self._corrector(anchor=anchor, gain=10.0)
        corrector.assess(state, step=0)
        out = corrector.correct(state, step=0)
        similarity = torch.nn.functional.cosine_similarity(
            out.flatten(), anchor.flatten(), dim=0
        )
        assert similarity > torch.nn.functional.cosine_similarity(
            state.flatten(), anchor.flatten(), dim=0
        )

    def test_correction_preserves_shape_and_dtype(self):
        state = torch.randn(3, 4, 5, dtype=torch.float16)
        anchor = torch.randn(3, 4, 5, dtype=torch.float16)
        corrector = self._corrector(anchor=anchor)
        corrector.assess(state, step=0)
        out = corrector.correct(state, step=0)
        assert out.shape == state.shape
        assert out.dtype == state.dtype

    def test_norm_cap_bounds_output(self):
        """Corrected norm never exceeds max_norm x original norm."""
        state = torch.randn(8)
        anchor = -state
        corrector = self._corrector(anchor=anchor, gain=100.0, max_norm=1.0)
        corrector.assess(state, step=0)
        out = corrector.correct(state, step=0)
        assert out.norm().item() <= state.norm().item() * (1 + 1e-3)

    def test_negative_direction_reduces_norm(self):
        """With gain < 1 the 'negative' direction shrinks the state's norm."""
        state = torch.randn(8)
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(reference=torch.zeros(8)),
            total_steps=10,
            gain=0.5,
            direction="negative",
        )
        corrector.assess(state, step=0)
        out = corrector.correct(state, step=0)
        # Anti-parallel anchor -> drift 1.0, so strength is decay * gain = 0.5
        # and the correction halves the norm.
        assert out.norm().item() == pytest.approx(0.5 * state.norm().item(), rel=1e-3)

    def test_norm_cap_clamps_an_aggressive_correction(self):
        """A huge gain still cannot push the norm past the cap."""
        state = torch.randn(8)
        anchor = -state
        corrector = self._corrector(anchor=anchor, gain=1000.0, max_norm=1.0)
        corrector.assess(state, step=0)
        out = corrector.correct(state, step=0)
        assert out.norm().item() == pytest.approx(state.norm().item(), rel=1e-3)
        # Clamped to the cap along the correction direction, i.e. flipped.
        assert torch.allclose(out, -state, atol=1e-3)

    def test_anchor_defaults_to_first_assessed_state(self):
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(reference=torch.zeros(2)), total_steps=4
        )
        assert corrector.anchor is None
        first = torch.tensor([1.0, 0.0])
        corrector.assess(first, step=0)
        assert torch.equal(corrector.anchor, first.reshape(-1))

    def test_history_records_every_assessment(self):
        corrector = self._corrector(anchor=torch.zeros(4))
        for step in range(3):
            corrector.assess(torch.randn(4), step=step)
        assert len(corrector.history) == 3

    def test_reset_clears_history_and_anchor(self):
        corrector = self._corrector(anchor=torch.zeros(4))
        corrector.assess(torch.randn(4), step=0)
        corrector.reset()
        assert corrector.history == []
        assert corrector.anchor is None


class TestSummary:
    def test_summary_reports_counts_and_strength(self):
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(reference=torch.zeros(4)),
            total_steps=10,
            anchor=torch.zeros(4),
        )
        for step in range(4):
            corrector.assess(torch.randn(4), step=step)
        summary = nnsight.modeling.correction_summary(corrector)
        assert summary["steps"] == 4
        assert 0 <= summary["corrected_steps"] <= 4
        assert summary["mean_drift"] >= 0.0
        assert summary["total_correction_strength"] >= 0.0

    def test_summary_on_empty_history(self):
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(reference=torch.zeros(4)), total_steps=4
        )
        summary = nnsight.modeling.correction_summary(corrector)
        assert summary["steps"] == 0
        assert summary["mean_drift"] == 0.0


# ---------------------------------------------------------------------------
# End-to-end: inside a real nnsight trace
# ---------------------------------------------------------------------------


class TestTraceIntegration:
    """The controller drives a real intervention across an iter loop."""

    @torch.no_grad()
    def test_assess_correct_across_iter_loop(self, recurrent_model, small_input):
        """Reading and correcting the latent at every step changes the output."""
        # Baseline: observe the trajectory without correcting.
        with recurrent_model.trace(small_input) as tracer:
            baseline_latents = list().save()
            for step in tracer.iter[:4]:
                baseline_latents.append(recurrent_model.linear.output.clone())
            baseline_out = recurrent_model.output.clone().save()

        # The inner module fires once per iter step, so we get 4 latents.
        assert len(baseline_latents) == 4

        # Supervised run: anchor to the first observed latent, then pull the
        # trajectory back toward it on every step.
        anchor = baseline_latents[0].clone()
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(reference=anchor), total_steps=4, anchor=anchor,
            gain=1.0,
        )

        with recurrent_model.trace(small_input) as tracer:
            reports = list().save()
            for step in tracer.iter[:4]:
                latent = recurrent_model.linear.output
                reports.append(corrector.assess(latent, step))
                recurrent_model.linear.output[:] = corrector.correct(latent, step)
            corrected_out = recurrent_model.output.clone().save()

        assert len(reports) == 4
        assert all(isinstance(report, DriftReport) for report in reports)
        # The supervisor sees real drift in an uncorrected linear chain.
        assert max(report.drift for report in reports) > 0.1
        # ... and the intervention actually moved the final output.
        assert not torch.allclose(baseline_out, corrected_out)

    @torch.no_grad()
    def test_supervised_run_stays_closer_to_anchor(self, recurrent_model, small_input):
        """Corrected latents are more anchor-aligned than the baseline's."""
        with recurrent_model.trace(small_input) as tracer:
            baseline_latents = list().save()
            for step in tracer.iter[:4]:
                baseline_latents.append(recurrent_model.linear.output.clone())

        anchor = baseline_latents[0].clone()
        corrector = TrajectoryCorrector(
            scorer=cosine_drift(reference=anchor), total_steps=4, anchor=anchor
        )

        with recurrent_model.trace(small_input) as tracer:
            corrected_latents = list().save()
            for step in tracer.iter[:4]:
                latent = recurrent_model.linear.output
                corrector.assess(latent, step)
                corrected = corrector.correct(latent, step)
                recurrent_model.linear.output[:] = corrected
                corrected_latents.append(recurrent_model.linear.output.clone())

        def alignment(tensor):
            return torch.nn.functional.cosine_similarity(
                tensor.flatten().float(), anchor.flatten().float(), dim=0
            ).item()

        # Skip step 0: the baseline's step-0 latent IS the anchor, so the
        # comparison is only meaningful from the first drifted step on.
        for baseline, corrected in zip(
            baseline_latents[1:], corrected_latents[1:]
        ):
            assert alignment(corrected) > alignment(baseline)

    @torch.no_grad()
    def test_null_corrector_is_identity(self, recurrent_model, small_input):
        """Threshold above any drift leaves the trajectory untouched."""
        with recurrent_model.trace(small_input) as tracer:
            baseline_out = recurrent_model.output.clone().save()

        corrector = TrajectoryCorrector(
            scorer=cosine_drift(reference=torch.zeros(1, 4)),
            total_steps=4,
            threshold=10.0,  # never exceeded -> never corrects
        )

        with recurrent_model.trace(small_input) as tracer:
            for step in tracer.iter[:4]:
                latent = recurrent_model.linear.output
                corrector.assess(latent, step)
                recurrent_model.linear.output[:] = corrector.correct(latent, step)
            null_out = recurrent_model.output.clone().save()

        assert torch.allclose(baseline_out, null_out)

    @torch.no_grad()
    def test_token_overlap_supervisor_runs_in_trace(
        self, recurrent_model, small_input
    ):
        """The lexical supervisor also composes with the loop, verbatim."""
        corrector = TrajectoryCorrector(
            scorer=token_overlap_drift(prompt="red panda bamboo"),
            total_steps=4,
        )

        with recurrent_model.trace(small_input) as tracer:
            reports = list().save()
            for step in tracer.iter[:4]:
                latent = recurrent_model.linear.output
                reports.append(corrector.assess(latent, step))
                recurrent_model.linear.output[:] = corrector.correct(latent, step)
            out = recurrent_model.output.clone().save()

        assert len(reports) == 4
        assert out.shape == (1, 4)
