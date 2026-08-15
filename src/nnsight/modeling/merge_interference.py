"""Directional erasure of merged-model interference.

Adapted from "Orientation, not magnitude: the causal structure of
task-vector interference in merged language models"
(arXiv:2608.11797). The paper's core causal claim is that interference
between merged task vectors is carried by a *direction* in the residual
stream, not by activation magnitude: erasing that direction removes
expressed interference dose-dependently and saturates at exact erasure,
while norm-matched erasure along a wrong direction fails or backfires.

This module packages that intervention as an nnsight-friendly utility:
project the residual stream onto the orthogonal complement of one (or
more) unit "interference directions" at a chosen layer, optionally
rescaled by a coefficient, so the same code runs the causal arm
(``coefficient=1.0``) and the norm-matched control arm
(``coefficient=1.0`` along a shuffled direction) of the paper's design.

The utility functions are plain tensor ops and are importable without a
model; ``apply`` is meant to be called inside a trace context on a
module's ``.output``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..intervention.envoy import Envoy


def directions_from_contrast(
    clean: torch.Tensor, other: torch.Tensor
) -> torch.Tensor:
    """Return unit interference directions from a paired contrast set.

    Given activations of the same prompts under two model states (e.g. the
    base model and the task-merged model), the mean-difference direction at
    each sequence position is the residual-stream displacement the merge
    induces there. The paper's own ledger identifies the carried direction
    from cross-term measurements; the contrast mean-difference is the
    parameter-free proxy used here (Mode 2 substitution).

    Args:
        clean: activations under the reference state, shape
            ``(batch, seq, hidden)`` (or any shape ending in ``hidden``).
        other: activations under the merged state, same shape.

    Returns:
        Tensor of shape ``clean.shape[:-1] + (hidden,)`` with unit-norm
        directions along the last dimension.
    """
    if clean.shape != other.shape:
        raise ValueError(
            f"contrast shapes must match, got {tuple(clean.shape)} and "
            f"{tuple(other.shape)}"
        )
    delta = other - clean
    return unit_columns(delta)


def unit_columns(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize ``x`` along its last dimension (zero columns stay zero)."""
    norms = x.norm(dim=-1, keepdim=True)
    return x / norms.clamp_min(torch.finfo(x.dtype).tiny)


def shuffle_columns(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Permute the hidden dimension independently per row.

    Norm-matched wrong-direction control: each output row has the same norm
    as its input row (the entries are only permuted) but points along an
    unrelated direction, exactly the control condition the paper reports as
    failing to remove interference.
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    hidden = x.shape[-1]
    noise = torch.rand(x.shape[:-1] + (hidden,), device=x.device, generator=g)
    order = noise.argsort(dim=-1)
    return torch.gather(x, -1, order)


def erase_along(
    x: torch.Tensor,
    directions: torch.Tensor,
    coefficient: float = 1.0,
) -> torch.Tensor:
    """Project ``x`` off ``directions`` scaled by ``coefficient``.

    ``coefficient=1.0`` is exact erasure of the span of the directions;
    values in ``[0, 1)`` give the paper's dose-response curve; values
    greater than 1 over-project (the "backfire" regime of the norm-matched
    control).

    Args:
        x: activations, shape ``(..., hidden)``.
        directions: unit directions, shape ``(..., hidden)`` (same leading
            shape as ``x``) or broadcastable to it.
        coefficient: fraction of the projection removed.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    if x.shape[-1] != directions.shape[-1]:
        raise ValueError(
            f"hidden dims must match, got {x.shape[-1]} and "
            f"{directions.shape[-1]}"
        )
    projections = (x * directions).sum(dim=-1, keepdim=True) * directions
    out = x - coefficient * projections
    return out.to(dtype=x.dtype)


def erase_from_resid(
    layer: "Envoy",
    directions: torch.Tensor,
    coefficient: float = 1.0,
) -> None:
    """Erase ``directions`` from a block's residual stream, in place.

    Call inside a trace context:

        with model.trace(prompt):
            erase_from_resid(model.transformer.h[LAYER], direction)

    Transformer blocks return tuples, so the edit is applied to the first
    element only (the hidden state), leaving any auxiliary outputs
    untouched.
    """
    output = layer.output
    hidden = output[0] if isinstance(output, tuple) else output
    hidden[:] = erase_along(hidden, directions, coefficient)


def interference_dose(
    baseline: torch.Tensor, intervened: torch.Tensor
) -> torch.Tensor:
    """Fraction of a baseline activation removed by an intervention.

    The paper's continuous endpoint: how much of the expressed interference
    (here, the displacement of the intervened run from the baseline run)
    the erasure actually removed. Reported per sequence position as

        ||baseline - intervened|| / ||baseline||

    so 0 means "no effect" and 1 means "erasure moved the activation by
    the full baseline norm".
    """
    if baseline.shape != intervened.shape:
        raise ValueError(
            f"shapes must match, got {tuple(baseline.shape)} and "
            f"{tuple(intervened.shape)}"
        )
    delta = (intervened - baseline).norm(dim=-1)
    return delta / baseline.norm(dim=-1).clamp_min(
        torch.finfo(baseline.dtype).tiny
    )
