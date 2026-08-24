"""Timestep feature caching for iterative models.

Adapted from `LinCa: Accelerating Diffusion Models via Learnable Decomposed
Feature Caching <https://arxiv.org/abs/2608.17973>`_.  LinCa decomposes cached
features into sub-components with distinct continuity properties and applies a
differentiated prediction order to each, then reconstructs losslessly — a
"Decompose-Predict-Reconstruct" pipeline that reaches 5-7x speedups at
near-lossless quality.

The part of that pipeline that matters to nnsight is not the learned
decomposition itself (that ships trained weights for FLUX / Qwen-Image /
HunyuanVideo and belongs with those checkpoints).  It is the *loop*: on
timesteps where a block's feature is highly continuous, do not run the block at
all — predict its output from earlier timesteps and splice the prediction in.

nnsight already exposes every primitive that loop needs.  ``tracer.iter[:]``
steps through the iterations; ``module.output`` reads a computed feature; and
``module.skip(replacement)`` bypasses the module's forward entirely so the
prediction replaces the block without the block ever executing.  This module
turns that loop into one call.

Adaptations (Mode 2):

- LinCa's learned invertible decomposition is replaced by a parameter-free
  predictor over the timestep axis: Lagrange interpolation through the last
  ``order`` cached samples of each feature channel.  "Distinct prediction
  orders matched to each component" survives as a per-module ``order`` —
  low-variance channels interpolate well at higher order, noisy ones need
  lower order.
- Separate per-model / per-timestep-segment predictors are replaced by a
  per-module rolling window, which adapts to local timestep dynamics for free
  and needs no training run.
- Strict invertibility is out of scope: nothing here decomposes a feature, so
  nothing needs to be reconstructed back.  The cache *is* the feature space.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

__all__ = ["CurveFitter", "FeatureReusePlan", "apply_feature_reuse"]


def _flatten(value: Any) -> Optional[List[torch.Tensor]]:
    """Flatten a module output into the tensors that will be fitted.

    Blocks wrapped by nnsight commonly return ``(hidden_states, ...)`` tuples
    where the trailing entries are ``None`` or empty — attention masks,
    past-key-value placeholders.  Only the tensors are fitted; non-tensor
    entries are dropped from the fit but the prediction still comes back as a
    tuple when the original had more than one, so downstream modules keep
    receiving the structure they expect.
    """
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)) and value:
        flat = []
        for item in value:
            if isinstance(item, torch.Tensor):
                flat.append(item)
            elif isinstance(item, (tuple, list)) and item:
                nested = _flatten(item)
                if nested is None:
                    return None
                flat.extend(nested)
        if flat:
            return flat
    return None


def _extrapolate(times: Sequence[float], values: Sequence[torch.Tensor], t: float):
    """Lagrange extrapolation through ``(times, values)`` evaluated at ``t``.

    With one sample the feature is held constant; with two it is linear; with
    ``order`` samples a degree ``order - 1`` polynomial.  Duplicate timesteps
    (an identity mapping, a repeated step, a mis-read scheduler attribute)
    would divide by zero, so those nodes are dropped before fitting; if the
    collapse leaves nothing fittable the nearest sample is held instead.
    """
    n = len(times)
    if n == 0:
        raise ValueError("No cached samples to extrapolate from.")
    if n == 1:
        return values[0]

    # Drop duplicate nodes — Lagrange denominators vanish on them.
    dedup_times = []
    dedup_values = []
    for i in range(n):
        if times[i] not in dedup_times:
            dedup_times.append(times[i])
            dedup_values.append(values[i])
    if len(dedup_times) == 1:
        return dedup_values[0]

    times, values, n = dedup_times, dedup_values, len(dedup_times)

    total = values[0] * 0.0
    for i in range(n):
        scale = 1.0
        for j in range(n):
            if i != j:
                scale *= (t - times[j]) / (times[i] - times[j])
        total = total + values[i] * scale

    # Guard both directions: a non-finite *input* poisons the sum (inf * 0 is
    # nan, but inf * nonzero is inf), and so does an exploding extrapolation.
    # Fall back to the newest finite sample so a poisoned cache entry can't
    # propagate an inf into the model in place of a real feature.
    for value in values:
        if not torch.isfinite(value).all():
            finite = [v for v in values if torch.isfinite(v).all()]
            return finite[-1] if finite else values[0] * 0.0
    if not torch.isfinite(total).all():
        return values[-1]
    return total


class CurveFitter:
    """Rolling per-module predictor over the timestep axis.

    Keeps the last ``order`` ``(timestep, feature)`` samples and extrapolates
    the next one.  This is the parameter-free stand-in for LinCa's learned
    per-component predictors: same interface (observe a timestep, predict the
    next), no trained weights.

    Args:
        order: Number of retained samples.  Higher order tracks smooth feature
            drift more closely but amplifies noise.
    """

    def __init__(self, order: int = 2) -> None:
        self.order = max(1, int(order))
        self.times: List[float] = []
        self.values: List[List[torch.Tensor]] = []

    def observe(self, t: float, value: Any) -> bool:
        """Record the feature computed at timestep ``t``.

        Returns ``False`` if the output held no tensors to fit, in which case
        the caller should leave the module on the compute path.
        """
        flat = _flatten(value)
        if not flat:
            return False
        self.times.append(float(t))
        self.values.append(flat)
        if len(self.times) > self.order:
            del self.times[: len(self.times) - self.order]
            del self.values[: len(self.values) - self.order]
        return True

    def predict(self, t: float) -> Any:
        """Predict the feature at timestep ``t`` from retained samples.

        Returns a bare tensor for single-tensor outputs and a tuple otherwise,
        matching the structure the module originally produced.
        """
        predicted = [
            _extrapolate(self.times, [row[i] for row in self.values], float(t))
            for i in range(len(self.values[0]))
        ]
        if len(predicted) == 1:
            return predicted[0]
        return tuple(predicted)

    @property
    def ready(self) -> bool:
        """Whether at least one sample has been observed."""
        return bool(self.values)

    def __len__(self) -> int:
        return len(self.times)


class FeatureReusePlan:
    """Which timesteps a block runs on, and at what order it is predicted.

    The schedule is a strided grid ``range(start, stop, stride)`` over timestep
    indices: a block *computes* on the grid and is *predicted* everywhere else.
    A block's execution fraction is therefore ``1 / stride``, and the whole
    denoiser's wall-clock saving is roughly the parameter-weighted mean of
    ``1 - 1 / stride`` across the blocks in the plan.

    Args:
        stride: Compute every ``stride``-th timestep, reusing in between.
            ``stride=1`` disables reuse for the module.
        order: Samples retained by that module's :class:`CurveFitter`.
        start: First timestep index the module computes on.
        stop: Exclusive upper bound on the compute grid (``None`` = unbounded).

    Attributes:
        executed: Timesteps the module actually computed.
        predicted: Timesteps served from the cache.
    """

    def __init__(
        self,
        stride: int = 1,
        order: int = 2,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.stride = int(stride)
        self.order = max(1, int(order))
        self.start = int(start)
        self.stop = stop
        self.executed: List[int] = []
        self.predicted: List[int] = []

    def computes_at(self, step: int) -> bool:
        """Whether the module should run its forward at ``step``."""
        if step < self.start or (self.stop is not None and step >= self.stop):
            return True
        return (step - self.start) % self.stride == 0

    @property
    def reuse_ratio(self) -> float:
        """Fraction of planned timesteps served from the cache."""
        total = len(self.executed) + len(self.predicted)
        if total == 0:
            return 0.0
        return len(self.predicted) / total

    def report(self, path: str) -> str:
        """One-line human summary of what happened to this module."""
        return (
            f"{path}: computed {len(self.executed)} timesteps "
            f"{self.executed}, predicted {len(self.predicted)} "
            f"({self.reuse_ratio:.0%} reused, order={self.order})"
        )

    def __repr__(self) -> str:
        stop = "" if self.stop is None else f", stop={self.stop}"
        return (
            f"FeatureReusePlan(stride={self.stride}, order={self.order}, "
            f"start={self.start}{stop})"
        )


def _normalize(spec: Union[FeatureReusePlan, int, Tuple[int, int]]) -> FeatureReusePlan:
    if isinstance(spec, FeatureReusePlan):
        return spec
    if isinstance(spec, int):
        return FeatureReusePlan(stride=spec)
    if isinstance(spec, tuple) and len(spec) == 2:
        return FeatureReusePlan(stride=spec[0], order=spec[1])
    raise TypeError(
        "Reuse spec must be a FeatureReusePlan, an int stride, or a "
        f"(stride, order) tuple — got {spec!r}"
    )


def apply_feature_reuse(
    tracer: Any,
    modules: Dict[str, Any],
    plan: Union[FeatureReusePlan, int, Tuple[int, int]],
    num_steps: Optional[int] = None,
    timestep: Optional[str] = None,
) -> Dict[str, FeatureReusePlan]:
    """Reuse cached block outputs across timesteps instead of recomputing them.

    Call this **inside** a trace or generate context, before the iteration
    loop it drives::

        with model.generate(prompt, num_inference_steps=20) as tracer:
            stats = apply_feature_reuse(
                tracer,
                {"unet": model.unet, "up_blocks.2": model.unet.up_blocks[2]},
                plan={"unet": 1, "up_blocks.2": (4, 3)},
            )
            for _ in tracer.iter[:]:
                pass
            output = tracer.output.save()

    On every step of ``tracer.iter[:]``, each module in ``plan`` either saves
    its computed output into a rolling :class:`CurveFitter` (compute timesteps)
    or is handed ``skip(predicted)``, so its forward is bypassed and the
    prediction flows onward in its place (reused timesteps).

    The returned plans carry per-module ``executed`` / ``predicted`` timestep
    lists plus a :attr:`FeatureReusePlan.reuse_ratio`, so a run can be
    reported without re-deriving the schedule.

    Args:
        tracer: The active tracer (``as ... tracer`` from the trace context).
        modules: Envoy modules to cache, keyed by report label.  Values are
            ``model.<path>`` envoys, e.g. ``{"unet": model.unet}``.
        plan: Schedule applied to every entry in ``modules`` — a
            :class:`FeatureReusePlan`, an int stride, or a ``(stride, order)``
            tuple.  Pass a dict keyed the same as ``modules`` to schedule
            modules differently.
        num_steps: Iteration count for ``tracer.iter[:num_steps]``.  Required
            unless a bound is set elsewhere (``generate`` sets one from
            ``num_inference_steps``).
        timestep: Attribute of the pipeline scheduler holding the current
            timestep value.  When omitted the integer step index is used,
            which is the correct grid for uniform schedulers.

    Returns:
        Per-module :class:`FeatureReusePlan` with the executed / predicted
        timestep bookkeeping filled in.

    Note:
        The return value is only visible inside the trace body.  Like any
        value created there, it does not survive the ``with`` block — stash it
        in a container defined outside the trace if you want the report after::

            report = {}
            with model.generate(prompt, num_inference_steps=20) as tracer:
                report["plans"] = apply_feature_reuse(tracer, modules, 4)

    Raises:
        ValueError: If ``plan`` is a dict missing one of ``modules``' keys.
        TypeError: If a plan entry is not a plan, int, or ``(stride, order)``.
    """
    if isinstance(plan, dict):
        plans = {}
        for key, module in modules.items():
            if key not in plan:
                raise ValueError(f"plan is missing an entry for module {key!r}")
            plans[key] = _normalize(plan[key])
    else:
        default = _normalize(plan)
        plans = {key: _normalize(plan) for key in modules}

    sched_timestep = None
    if timestep is not None:
        for module in modules.values():
            sched = getattr(_root_module(module), "scheduler", None)
            if sched is not None and hasattr(sched, timestep):
                sched_timestep = getattr(sched, timestep)
                break

    def _current_t(step: int):
        if sched_timestep is not None:
            value = getattr(sched_timestep, "value", sched_timestep)
            try:
                return float(value)
            except TypeError:
                return float(step)
        return float(step)

    fitters: Dict[int, CurveFitter] = {}

    steps = tracer.iter[:num_steps] if num_steps is not None else tracer.iter[:]
    for step in steps:
        t = _current_t(step)
        for key, module in modules.items():
            module_plan = plans[key]
            fitter = fitters.get(key)
            if fitter is None:
                fitter = CurveFitter(order=module_plan.order)
                fitters[key] = fitter

            if module_plan.computes_at(step) or not fitter.ready:
                # Compute step (or the first step, before anything is cached).
                observed = fitter.observe(t, module.output.save())
                if not observed:
                    # Output had no fittable tensors — keep computing silently.
                    continue
                module_plan.executed.append(step)
            else:
                # Reuse step: the block never runs, the prediction takes its place.
                module.skip(fitter.predict(t))
                module_plan.predicted.append(step)

    return plans


def _root_module(envoy: Any) -> Any:
    """Walk an envoy up to its root so pipeline attributes (``scheduler``) resolve.

    Envoy chains terminate at the model wrapper, which mirrors the pipeline's
    components (``scheduler`` among them) as attributes.
    """
    node = envoy
    root = envoy
    seen = {id(node)}
    while node is not None:
        node = getattr(node, "parent", None)
        if node is None or id(node) in seen:
            break
        seen.add(id(node))
        root = node
    return root
