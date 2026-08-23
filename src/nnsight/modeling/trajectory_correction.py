"""Mid-generation trajectory correction for iterated latent states.

Adapted from "MLLM-Guided Semantic Correction for Text-to-Video Generation"
(arXiv:2608.16513). That paper interleaves two roles inside a diffusion
sampling loop:

* a **Semantic Assessment Supervisor** that decodes intermediate latents to
  preview frames, scores them against the prompt, and diagnoses drift, and
* a **Semantic Modification Assistant** that turns each diagnosis into a
  *latent trajectory intervention* — a small, bounded correction applied to
  the latent state at the current step, scaled by remaining steps.

Both halves are already nnsight primitives: the supervisor is a read of a
mid-loop state (``module.output`` inside ``for step in tracer.iter[...]``),
and the assistant is an in-place write to the same state. This module gives
that loop a reusable controller so the pattern is three lines instead of a
hand-rolled scheduler.

What is kept at full fidelity
-----------------------------
* Per-step *assess-then-correct*: the supervisor is consulted on every
  monitored step and the correction it returns is applied before the next
  one, inside the same trace.
* Step-decayed gain: the paper scales interventions by remaining steps so
  late corrections are gentle and cannot wash out an almost-converged
  sample. ``TrajectoryCorrector`` implements this as
  ``max(0, 1 - step / total_steps)`` and exposes it as :meth:`decay`.
* Bounded, direction-and-norm interventions: the correction is a direction
  whose magnitude is capped at ``max_norm`` relative to the current state's
  own norm, so a mis-scored step cannot explode the trajectory.
* Parameter-free: like the paper, nothing is trained. All scorers here are
  cheap proxies standing in for the MLLM judge (see below).

What is substituted (Mode 2)
---------------------------
The paper's MLLM judge is replaced by any callable ``state -> DriftReport``.
Two zero-dependency proxies ship: :func:`cosine_drift` (geometric drift of
the latent itself, i.e. "the sample is drifting away from where the prompt
anchored it") and :func:`token_overlap_drift` (lexical alignment between
the prompt and a decoded description of the state, standing in for the
MLLM's semantic-alignment read). Supply your own supervisor — e.g. a real
VLM captioning a VAE-decoded preview — and the controller is unchanged.

The correction is a fixed direction (default: back toward the anchor state)
rather than a learned edit; the paper's learned components are out of scope
by design, since this runs inside a trace with no optimizer.

Example
-------
::

    from nnsight.modeling import TrajectoryCorrector, cosine_drift

    corrector = TrajectoryCorrector(
        scorer=cosine_drift(reference=anchor_state), total_steps=50
    )

    with sd.generate(prompt, num_inference_steps=50) as tracer:
        reports = list().save()
        for step in tracer.iter[:]:
            report = corrector.assess(sd.unet.output[0], step)
            sd.unet.output[0][:] = corrector.correct(sd.unet.output[0], step)
            reports.append(report)

``correct()`` is a no-op (returns the state untouched) whenever the
supervisor reports drift at or below ``threshold`` — the default threshold
of ``0.0`` means "always correct", matching the paper's always-on loop.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import torch

__all__ = [
    "DriftReport",
    "Supervisor",
    "cosine_drift",
    "token_overlap_drift",
    "TrajectoryCorrector",
    "correction_summary",
]


class DriftReport:
    """A supervisor's diagnosis of one monitored step.

    Attributes:
        drift: Non-negative drift score. Higher means the state has moved
            further from what the supervisor considers on-prompt. The
            controller corrects only when ``drift > threshold``.
        weight: Optional per-step strength multiplier in ``[0, 1]``. A
            supervisor that is confident about a step can pass ``1.0``; one
            that is unsure can down-weight it. ``None`` means "no opinion"
            and the controller uses ``1.0``.
        info: Arbitrary supervisor metadata (e.g. which prompt tokens were
            missing). Kept on :attr:`TrajectoryCorrector.history` so a
            research loop can inspect *why* each correction fired.
    """

    __slots__ = ("drift", "weight", "info")

    def __init__(
        self,
        drift: float,
        weight: Optional[float] = None,
        info: Optional[dict] = None,
    ) -> None:
        drift = float(drift)
        if drift < 0:
            raise ValueError(f"drift must be non-negative, got {drift}")
        if weight is not None:
            weight = float(weight)
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"weight must be in [0, 1], got {weight}")

        self.drift = drift
        self.weight = weight
        self.info = info if info is not None else {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DriftReport(drift={self.drift:.4f}, weight={self.weight})"


class Supervisor(Protocol):
    """Anything that can score a latent state for drift."""

    def __call__(self, state: Any, step: int) -> DriftReport:
        """Score ``state`` at ``step``.

        Args:
            state: The latent (or activation) the controller just read.
            step: Current step index within the trajectory.

        Returns:
            A :class:`DriftReport`.
        """
        ...


def _flatten(state: Any) -> torch.Tensor:
    """Coerce a state to a flat ``float32`` 1-D tensor for scoring."""
    if isinstance(state, torch.Tensor):
        tensor = state.detach().float().reshape(-1)
    elif isinstance(state, (tuple, list)):
        # Multi-output modules (UNet returns ``(sample,)`` in some diffusers
        # versions) — score the concatenation of tensor elements.
        tensor = torch.cat(
            [
                s.detach().float().reshape(-1)
                for s in state
                if isinstance(s, torch.Tensor)
            ]
        )
    else:
        raise TypeError(
            f"Expected a torch.Tensor or a tuple/list of tensors, got {type(state)!r}"
        )
    return tensor


def cosine_drift(
    reference: Any, eps: float = 1e-8
) -> Callable[[Any, int], DriftReport]:
    """Build a supervisor that reports cosine distance from ``reference``.

    The proxy for "the state has drifted off the prompt": the anchor is the
    state the trajectory started from (or a prompt-conditioned target), and
    drift is ``1 - cos(state, reference)``. Near-identical states score ~0;
    orthogonal ones score ~1.

    Args:
        reference: The anchor state (tensor, or tuple/list of tensors).
        eps: Numerical guard for the norms.

    Returns:
        A supervisor callable usable as ``TrajectoryCorrector(scorer=...)``.
    """
    reference_flat = _flatten(reference)

    def supervisor(state: Any, step: int) -> DriftReport:
        state_flat = _flatten(state)
        if state_flat.shape != reference_flat.shape:
            raise ValueError(
                "cosine_drift: state shape "
                f"{tuple(state_flat.shape)} does not match reference shape "
                f"{tuple(reference_flat.shape)}"
            )
        denominator = state_flat.norm() * reference_flat.norm()
        similarity = torch.dot(state_flat, reference_flat) / (denominator + eps)
        # Clamp before inverting: float32 cosine can land a hair above 1.0,
        # which would surface as a (invalid) negative drift.
        similarity = similarity.clamp(-1.0, 1.0)
        return DriftReport(drift=float(1.0 - similarity))

    return supervisor


def _normalize(text: str) -> list:
    """Lowercase, strip punctuation, drop stopwords — leaves content words."""
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "to", "was", "were", "will", "with",
    }
    tokens = []
    for chunk in text.lower().replace("-", " ").split():
        word = chunk.strip(".,;:!?()[]{}\"'")
        if word and word not in stop:
            tokens.append(word)
    return tokens


def token_overlap_drift(
    prompt: str,
    describe: Optional[Callable[[Any], str]] = None,
) -> Callable[[Any, int], DriftReport]:
    """Build a supervisor that reports lexical drift from ``prompt``.

    The proxy for the paper's MLLM semantic-alignment read: instead of
    asking a VLM "does this frame contain the prompt's subject?", compare
    the prompt's content words against a textual description of the state.
    Drift is ``1 - overlap``, where ``overlap`` is the fraction of prompt
    content words present in the description.

    Args:
        prompt: The conditioning prompt. Used only for its vocabulary.
        describe: Optional ``state -> str`` decoder. Defaults to
            :func:`tensor_token_description`, which hashes a state's
            sign/quantization pattern onto a fixed vocabulary — enough to
            make the *plumbing* testable offline; replace it with a real
            decoder (VAE decode + captioner) for actual semantic scoring.

    Returns:
        A supervisor callable usable as ``TrajectoryCorrector(scorer=...)``.
    """
    prompt_words = set(_normalize(prompt))
    if describe is None:
        describe = tensor_token_description

    def supervisor(state: Any, step: int) -> DriftReport:
        description = describe(state)
        described = set(_normalize(description))
        if not prompt_words:
            return DriftReport(drift=0.0)
        overlap = len(prompt_words & described) / len(prompt_words)
        return DriftReport(
            drift=float(1.0 - overlap), info={"description": description}
        )

    return supervisor


def tensor_token_description(state: Any, vocabulary_size: int = 64) -> str:
    """Deterministically describe a tensor state with a fixed vocabulary.

    Quantizes the state's flat values into ``vocabulary_size`` bins and
    emits one synthetic word per bin — ``token00 ... token63``. Two states
    with similar value distributions therefore produce overlapping
    descriptions. This is a stand-in decoder: it makes
    :func:`token_overlap_drift` runnable with no model and no downloads,
    and is intentionally content-free — swap it for a real decoder when you
    want semantic rather than distributional drift.
    """
    flat = _flatten(state)
    if flat.numel() == 0:
        return ""
    minimum = flat.min()
    maximum = flat.max()
    span = maximum - minimum
    if span.item() == 0:
        bins = torch.zeros_like(flat, dtype=torch.long)
    else:
        bins = ((flat - minimum) / span * (vocabulary_size - 1)).long()
    counts = torch.bincount(bins, minlength=vocabulary_size)
    present = (counts > 0).nonzero(as_tuple=True)[0]
    return " ".join(f"token{int(index):02d}" for index in present)


class TrajectoryCorrector:
    """Per-step assess-then-correct controller for an iterated latent state.

    Wraps the paper's two roles around an existing nnsight iter loop: call
    :meth:`assess` on the state you just read, then :meth:`correct` on the
    state you are about to write back. Both take the step index, because the
    correction strength decays with remaining steps.

    The correction is ``state + decay(step) * gain * drift * weight *
    direction``, where ``direction`` is ``normalize(anchor - state)`` — i.e.
    pull the trajectory back toward the anchor state by an amount set by how
    far the supervisor says it has strayed, then shrink that pull as the
    trajectory converges. The result is clamped so its norm never exceeds
    ``max_norm`` times the *original* state's norm, bounding worst-case
    damage from a mis-scoring supervisor.

    All state (anchor, history) lives on this object, so one instance per
    trajectory — reuse across traces is a caller bug, not a guard we can
    enforce cheaply.

    Args:
        scorer: The supervisor. Called as ``scorer(state, step)``.
        total_steps: Total number of steps in the trajectory. Used only for
            decay; corrections still apply (decayed) if the loop runs longer.
        anchor: The state corrections pull toward. If ``None``, the first
            state passed to :meth:`assess` becomes the anchor.
        gain: Multiplier on the drift score. ``1.0`` applies a correction
            whose magnitude is proportional to reported drift.
        threshold: Correct only when ``drift > threshold``. ``0.0`` corrects
            on every monitored step.
        max_norm: Norm cap on the corrected state, as a multiple of the
            incoming state's own norm. ``1.0`` forbids any growth.
        direction: ``"anchor"`` pulls toward :attr:`anchor`; ``"negative"``
            applies ``-normalize(state)`` (a pure norm reduction, useful
            when there is no meaningful anchor).
    """

    def __init__(
        self,
        scorer: Supervisor,
        total_steps: int,
        anchor: Any = None,
        gain: float = 1.0,
        threshold: float = 0.0,
        max_norm: float = 1.0,
        direction: str = "anchor",
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if gain < 0:
            raise ValueError(f"gain must be non-negative, got {gain}")
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}")
        if direction not in ("anchor", "negative"):
            raise ValueError(
                f"direction must be 'anchor' or 'negative', got {direction!r}"
            )

        self.scorer = scorer
        self.total_steps = int(total_steps)
        self.gain = float(gain)
        self.threshold = float(threshold)
        self.max_norm = float(max_norm)
        self.direction = direction
        #: Anchor state corrections pull toward. Set from the first assessed
        #: state when not given explicitly.
        self.anchor: Optional[torch.Tensor] = (
            _flatten(anchor) if anchor is not None else None
        )
        #: One :class:`DriftReport` per assessed step, in call order.
        self.history: list = []

    def decay(self, step: int) -> float:
        """Remaining-steps gain for ``step`` — 1 at step 0, 0 at the end."""
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")
        remaining = max(0, self.total_steps - step)
        return remaining / self.total_steps

    def assess(self, state: Any, step: int) -> DriftReport:
        """Run the supervisor on ``state`` and record the report.

        This is the read half of the loop — it never mutates the trajectory.
        The first call fixes the anchor when none was supplied.

        Returns:
            The recorded :class:`DriftReport`.
        """
        report = self.scorer(state, step)
        if self.anchor is None:
            self.anchor = _flatten(state)
        self.history.append(report)
        return report

    def correct(self, state: torch.Tensor, step: int) -> torch.Tensor:
        """Return the corrected state for ``step``.

        No-op unless the last :meth:`assess` reported drift above
        :attr:`threshold`; the caller writes the returned tensor back, e.g.
        ``module.output[0][:] = corrector.correct(module.output[0], step)``.

        Args:
            state: The state to correct.
            step: Current step index, used for decay.

        Returns:
            A tensor of ``state``'s shape and dtype. When no correction is
            due, ``state`` itself is returned unchanged (not a copy), so an
            identity assignment is free.
        """
        if not self.history:
            raise RuntimeError(
                "correct() called before assess() — the supervisor must score "
                "the state before it can be corrected."
            )
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}")

        report = self.history[-1]
        if report.drift <= self.threshold:
            return state

        weight = 1.0 if report.weight is None else report.weight
        strength = self.decay(step) * self.gain * report.drift * weight
        if strength <= 0.0:
            return state

        original = state.detach()
        flat = original.reshape(-1).float()
        norm = flat.norm()

        if self.direction == "anchor":
            if self.anchor is None:  # pragma: no cover - assess() always sets it
                return state
            if self.anchor.shape != flat.shape:
                raise ValueError(
                    "anchor shape "
                    f"{tuple(self.anchor.shape)} does not match state shape "
                    f"{tuple(flat.shape)}"
                )
            delta = self.anchor.to(flat.device) - flat
        else:
            delta = -flat

        direction_norm = delta.norm()
        if direction_norm.item() == 0.0:
            return state
        step_vector = delta / direction_norm * (strength * norm.item())

        corrected = flat + step_vector
        corrected_norm = corrected.norm()
        cap = self.max_norm * norm.item()
        if corrected_norm.item() > cap > 0.0:
            corrected = corrected * (cap / corrected_norm.item())

        return (
            corrected.reshape(original.shape).to(original.dtype).to(original.device)
        )

    def reset(self, anchor: Any = None) -> None:
        """Clear per-trajectory state (history and, optionally, the anchor).

        Args:
            anchor: New anchor state. If ``None``, the next :meth:`assess`
                call re-fixes the anchor.
        """
        self.history.clear()
        self.anchor = _flatten(anchor) if anchor is not None else None


def correction_summary(corrector: TrajectoryCorrector) -> dict:
    """Summarize a finished trajectory's supervision as plain numbers.

    Convenience for logging after a trace exits — the returned dict is
    plain Python, so it survives the trace (``.save()`` not required).

    Returns:
        Dict with ``steps``, ``corrected_steps``, ``mean_drift``,
        ``max_drift``, and ``total_correction_strength`` (the summed
        ``decay * gain * drift * weight`` the controller would have
        applied, whether or not each step was under threshold).
    """
    reports = corrector.history
    corrected = sum(
        1 for report in reports if report.drift > corrector.threshold
    )
    strengths = [
        corrector.decay(step) * corrector.gain * report.drift
        * (1.0 if report.weight is None else report.weight)
        for step, report in enumerate(reports)
    ]
    return {
        "steps": len(reports),
        "corrected_steps": corrected,
        "mean_drift": sum(report.drift for report in reports) / len(reports)
        if reports
        else 0.0,
        "max_drift": max((report.drift for report in reports), default=0.0),
        "total_correction_strength": float(sum(strengths)),
    }
