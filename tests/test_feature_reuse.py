"""
Tests for timestep feature reuse (``nnsight.modeling.feature_reuse``).

These tests cover the feature-caching loop built on ``tracer.iter`` and
``module.skip``:
- Reuse actually bypasses the cached module's forward
- Modules outside the plan are untouched
- The predictor is exact on polynomial features and safe on bad timestep grids
- ``NNsight.feature_reuse`` resolves module paths onto envoys
- Multi-tensor (tuple) outputs keep their structure through the cache
"""

import pytest
import torch

from nnsight import NNsight
from nnsight.modeling import FeatureReusePlan, apply_feature_reuse
from nnsight.modeling.base import NNsight as NNsightBase
from nnsight.modeling.feature_reuse import CurveFitter, _extrapolate


class CountingBlock(torch.nn.Module):
    """Residual block that counts how many times its forward really ran."""

    def __init__(self, d):
        super().__init__()
        self.l = torch.nn.Linear(d, d)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.l(x) + x


class IterativeModel(torch.nn.Module):
    """Model whose forward repeats a stack of blocks ``steps`` times.

    Stands in for a diffusion denoiser: the same modules fire once per
    iteration, which is the shape ``tracer.iter`` steps through.
    """

    def __init__(self, d=16, n=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList([CountingBlock(d) for _ in range(n)])
        self.head = torch.nn.Linear(d, d)

    def forward(self, x, steps=4):
        for _ in range(steps):
            for b in self.blocks:
                x = b(x)
        return self.head(x)


@pytest.fixture(scope="module")
def reuse_model(device: str):
    """Iterative model wrapped with NNsight, on the test device."""
    torch.manual_seed(0)
    return NNsight(IterativeModel()).to(device)


@pytest.fixture
def counts(reuse_model: NNsight):
    """Zero the per-block forward counters and hand back the raw blocks."""
    raw = reuse_model._module.blocks
    for b in raw:
        b.calls = 0
    return raw


class TestCurveFitter:
    """The parameter-free predictor that stands in for the learned one."""

    def test_exact_on_quadratic(self):
        """Three samples of a quadratic are reproduced exactly anywhere."""
        f = CurveFitter(order=3)
        for t, v in [(0.0, 1.0), (1.0, 4.0), (2.0, 9.0)]:
            f.observe(t, torch.tensor([v]))
        # f(t) = (t + 1) ** 2 through all three nodes.
        for t, expected in [(0.5, 2.25), (1.5, 6.25), (3.0, 16.0)]:
            assert f.predict(t).item() == pytest.approx(expected, abs=1e-5)

    def test_linear_and_constant(self):
        """Two nodes give a line; one node holds the value."""
        f = CurveFitter(order=2)
        f.observe(0.0, torch.tensor([0.0]))
        f.observe(1.0, torch.tensor([10.0]))
        assert f.predict(0.5).item() == pytest.approx(5.0)

        g = CurveFitter(order=1)
        g.observe(0.0, torch.tensor([7.0]))
        assert g.predict(100.0).item() == pytest.approx(7.0)

    def test_rolling_window_drops_old_samples(self):
        """Only the last ``order`` samples are retained."""
        f = CurveFitter(order=2)
        for i in range(5):
            f.observe(float(i), torch.tensor([float(i) ** 2]))
        assert f.times == [3.0, 4.0]
        # Linear through (3, 9) and (4, 16).
        assert f.predict(5.0).item() == pytest.approx(23.0)

    def test_duplicate_timesteps_do_not_divide_by_zero(self):
        """Repeated timesteps degrade to the fittable subset instead of raising."""
        f = CurveFitter(order=3)
        f.observe(1.0, torch.tensor([3.0]))
        f.observe(1.0, torch.tensor([3.0]))
        f.observe(2.0, torch.tensor([5.0]))
        # Collapses to the line through (1, 3) and (2, 5).
        assert f.predict(3.0).item() == pytest.approx(7.0)

        all_dup = CurveFitter(order=2)
        all_dup.observe(4.0, torch.tensor([2.0]))
        all_dup.observe(4.0, torch.tensor([2.0]))
        assert all_dup.predict(9.0).item() == pytest.approx(2.0)

    def test_non_tensor_output_is_rejected(self):
        """A module with no tensors in its output is left on the compute path."""
        f = CurveFitter(order=2)
        assert f.observe(0.0, {"not": "a tensor"}) is False
        assert f.ready is False

    def test_tuple_output_round_trips(self):
        """Tuple outputs predict back as a tuple, single tensors as a tensor."""
        f = CurveFitter(order=2)
        f.observe(0.0, (torch.tensor([1.0]), torch.tensor([2.0])))
        f.observe(1.0, (torch.tensor([3.0]), torch.tensor([5.0])))
        out = f.predict(2.0)
        assert isinstance(out, tuple)
        assert out[0].item() == pytest.approx(5.0)
        assert out[1].item() == pytest.approx(8.0)

    def test_extrapolate_nonfinite_falls_back(self):
        """A non-finite result returns the nearest cached sample."""
        ts = [0.0, 1.0]
        vs = [torch.tensor([1.0]), torch.tensor([float("inf")])]
        assert _extrapolate(ts, vs, 5.0).item() == pytest.approx(1.0)


class TestFeatureReusePlan:
    """The stride / order schedule."""

    def test_stride_grid(self):
        plan = FeatureReusePlan(stride=4)
        assert [s for s in range(9) if plan.computes_at(s)] == [0, 4, 8]

    def test_stride_one_computes_everything(self):
        plan = FeatureReusePlan(stride=1)
        assert all(plan.computes_at(s) for s in range(5))

    def test_start_and_stop_bounds_the_grid(self):
        plan = FeatureReusePlan(stride=3, start=1, stop=7)
        assert [s for s in range(9) if plan.computes_at(s)] == [0, 1, 4, 7, 8]

    def test_invalid_stride_rejected(self):
        with pytest.raises(ValueError):
            FeatureReusePlan(stride=0)

    def test_reuse_ratio_and_report(self):
        plan = FeatureReusePlan(stride=2)
        plan.executed = [0, 2]
        plan.predicted = [1, 3]
        assert plan.reuse_ratio == 0.5
        assert "computed 2" in plan.report("block")
        assert "50% reused" in plan.report("block")
        assert "stride=2" in repr(plan)


class TestApplyFeatureReuse:
    """The reuse loop driving a real trace."""

    def test_cached_module_forward_is_bypassed(
        self, reuse_model: NNsight, counts, device: str
    ):
        """A block in the plan runs only on compute timesteps.

        This is the core claim: ``skip()`` must genuinely bypass the forward,
        not merely overwrite its output afterwards. The counter can only stay
        at 3 if the block's ``forward`` never ran for the other 6 steps.
        """
        STEPS = 9
        x = torch.rand(1, 16, device=device)

        report = {}
        with reuse_model.trace(x, steps=STEPS) as tracer:
            report["plans"] = apply_feature_reuse(
                tracer,
                *reuse_model.feature_reuse({"blocks.1": (3, 2)}),
                num_steps=STEPS,
            )
            output = reuse_model.output.save()

        assert isinstance(output, torch.Tensor)
        # Not in the plan -> every step computed.
        assert counts[0].calls == STEPS
        assert counts[2].calls == STEPS
        # In the plan with stride 3 -> only steps 0, 3, 6.
        assert counts[1].calls == 3

        plan = report["plans"]["blocks.1"]
        assert plan.executed == [0, 3, 6]
        assert plan.predicted == [1, 2, 4, 5, 7, 8]
        assert plan.reuse_ratio == pytest.approx(6 / 9)

    def test_first_step_always_computes(
        self, reuse_model: NNsight, counts, device: str
    ):
        """With no cached samples yet the block computes rather than predict nothing."""
        STEPS = 4
        x = torch.rand(1, 16, device=device)

        with reuse_model.trace(x, steps=STEPS) as tracer:
            apply_feature_reuse(
                tracer,
                *reuse_model.feature_reuse({"blocks.2": FeatureReusePlan(stride=5)}),
                num_steps=STEPS,
            )

        # Stride 5 would compute only at step 0 within 4 steps anyway.
        assert counts[2].calls == 1
        assert counts[2].calls < STEPS

    def test_stride_one_is_a_no_op(
        self, reuse_model: NNsight, counts, device: str
    ):
        """``stride=1`` disables reuse, so the trace must match a plain run."""
        STEPS = 5
        x = torch.rand(1, 16, device=device)

        with reuse_model.trace(x, steps=STEPS):
            baseline = reuse_model.output.save()
        for b in counts:
            b.calls = 0

        with reuse_model.trace(x, steps=STEPS) as tracer:
            apply_feature_reuse(
                tracer,
                *reuse_model.feature_reuse({"blocks.1": 1}),
                num_steps=STEPS,
            )
            reused = reuse_model.output.save()

        assert all(b.calls == STEPS for b in counts)
        assert torch.allclose(baseline, reused)

    def test_per_module_plans(self, reuse_model: NNsight, counts, device: str):
        """Different modules can carry different strides and orders."""
        STEPS = 9
        x = torch.rand(1, 16, device=device)

        with reuse_model.trace(x, steps=STEPS) as tracer:
            apply_feature_reuse(
                tracer,
                *reuse_model.feature_reuse(
                    {"blocks.0": (3, 2), "blocks.1": (2, 3), "blocks.2": 1}
                ),
                num_steps=STEPS,
            )

        assert counts[0].calls == 3  # stride 3
        assert counts[1].calls == 5  # stride 2
        assert counts[2].calls == STEPS  # stride 1 -> no reuse

    def test_plan_dict_missing_module_key_raises(
        self, reuse_model: NNsight, device: str
    ):
        """A plan dict that omits a requested module is a configuration error."""
        x = torch.rand(1, 16, device=device)
        modules, _ = reuse_model.feature_reuse({"blocks.0": 2})

        with reuse_model.trace(x, steps=2) as tracer:
            with pytest.raises(ValueError, match="missing an entry"):
                apply_feature_reuse(tracer, modules, {"other": 2}, num_steps=2)

    def test_bad_spec_type_raises(self, reuse_model: NNsight, device: str):
        """Plan entries must be a plan, an int stride, or a (stride, order) tuple."""
        x = torch.rand(1, 16, device=device)
        with reuse_model.trace(x, steps=2) as tracer:
            with pytest.raises(TypeError, match="Reuse spec"):
                apply_feature_reuse(
                    tracer,
                    *reuse_model.feature_reuse({"blocks.0": "fast"}),
                    num_steps=2,
                )


class TestFeatureReusePathResolution:
    """``NNsight.feature_reuse`` module-path resolution."""

    def test_resolves_to_envoys(self, reuse_model: NNsight):
        """Resolved values must carry ``.output`` / ``.skip``."""
        modules, plan = reuse_model.feature_reuse({"blocks.0": 2, "blocks.1": (3, 2)})

        assert set(modules) == {"blocks.0", "blocks.1"}
        for envoy in modules.values():
            assert isinstance(envoy, NNsightBase) or hasattr(envoy, "skip")
        assert plan == {"blocks.0": 2, "blocks.1": (3, 2)}

    def test_unknown_path_raises(self, reuse_model: NNsight):
        with pytest.raises(AttributeError, match="does not resolve"):
            reuse_model.feature_reuse({"not.a.module": 2})

    def test_empty_spec_is_empty(self, reuse_model: NNsight):
        modules, plan = reuse_model.feature_reuse(None)
        assert modules == {} and plan == {}

        modules, plan = reuse_model.feature_reuse({})
        assert modules == {} and plan == {}


class TestTupleOutputReuse:
    """Blocks returning ``(hidden, ...)`` tuples survive the cache."""

    def test_tuple_returning_block_reuses(self, device: str):
        class TupleBlock(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.l = torch.nn.Linear(d, d)
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return (self.l(x) + x, None)

        class TupleModel(torch.nn.Module):
            def __init__(self, d=16):
                super().__init__()
                self.block = TupleBlock(d)
                self.head = torch.nn.Linear(d, d)

            def forward(self, x, steps=4):
                for _ in range(steps):
                    x = self.block(x)[0]
                return self.head(x)

        torch.manual_seed(0)
        raw = TupleModel()
        model = NNsight(raw).to(device)

        STEPS = 5
        x = torch.rand(1, 16, device=device)
        with model.trace(x, steps=STEPS) as tracer:
            apply_feature_reuse(
                tracer, *model.feature_reuse({"block": (2, 2)}), num_steps=STEPS
            )
            output = model.output.save()

        assert isinstance(output, torch.Tensor)
        # Stride 2 over 5 steps computes at 0, 2, 4.
        assert raw.block.calls == 3
