"""Subspace activation patching and the subspace-attribution caveat.

Subspace activation patching replaces only the component of an activation
that lies inside a hypothesized low-dimensional feature subspace, instead of
the whole activation vector as in standard activation patching. It is used
both to *manipulate* behavior and to *attribute* that behavior to the
subspace.

Makelov, Lange & Nanda (2023) show those two aims diverge: a subspace patch
can flip behavior through a dormant parallel pathway rather than through the
patched subspace itself, so a successful behavioral patch does not by itself
certify the subspace as causally load-bearing. The mitigation this module
bakes in is to run a same-rank control basis alongside the candidate one: a
candidate subspace is only interesting if it beats the control, not merely if
the patch works.

See docs/patterns/subspace-patching.md for the full recipe.
"""

from typing import Optional, Union

import torch

__all__ = ["orthonormalize", "project_component", "subspace_patch", "random_basis"]


def orthonormalize(basis: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis for the span of ``basis``.

    Args:
        basis: ``[d, k]`` matrix whose columns span the candidate subspace.

    Returns:
        ``[d, k']`` orthonormal matrix with the same column space (``k'`` may
        be smaller than ``k`` if the input was rank-deficient).
    """
    if basis.dim() != 2:
        raise ValueError(f"basis must be 2-D [d, k], got shape {tuple(basis.shape)}")
    q = torch.linalg.qr(basis).Q
    # torch.linalg.qr may return extra zero columns when rank-deficient; drop them.
    keep = q.norm(dim=0) > 1e-6
    return q[:, keep]


def project_component(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Component of ``x`` lying inside the subspace spanned by ``basis``.

    Args:
        x: ``[..., d]`` tensor.
        basis: ``[d, k]`` orthonormal (or plain) basis of the subspace.

    Returns:
        ``[..., d]`` tensor holding ``U (U^T x)`` — the projection of ``x``
        onto ``span(basis)``.
    """
    b = orthonormalize(basis)
    return (x @ b) @ b.T


def random_basis(
    dim: int,
    rank: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """A uniformly random orthonormal ``rank``-dim subspace of ``R^dim``.

    Used as the null control in subspace attribution: if patching a random
    same-rank subspace reproduces the candidate's behavioral effect, the
    effect is not evidence about the candidate subspace.
    """
    if not 0 < rank <= dim:
        raise ValueError(f"rank must be in (0, {dim}], got {rank}")
    noise = torch.randn(dim, rank, generator=generator, device=device)
    return orthonormalize(noise)


def subspace_patch(
    activation: torch.Tensor,
    source: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Return the source's subspace component, ready to add to ``activation``.

    Computes ``P_U(source) - P_U(activation)`` for subspace basis ``U``. In a
    trace this composes with an in-place add:

    .. code-block:: python

        model.transformer.h[LAYER].output[:, -1, :] += subspace_patch(
            model.transformer.h[LAYER].output[:, -1, :],
            clean_hs,
            basis,
        )

    which patches *only* the projection onto ``span(basis)`` from ``source``,
    leaving the orthogonal complement of the running activation untouched —
    the subspace analogue of the full-vector patch
    ``output[:, -1, :] = clean_hs``.

    Args:
        activation: the running activation to be modified, ``[..., d]``.
        source: the activation to patch *from* (e.g. the clean run), ``[..., d]``.
        basis: ``[d, k]`` basis of the hypothesized subspace.

    Returns:
        ``[..., d]`` delta that, added to ``activation``, makes its subspace
        component equal to the source's.
    """
    b = orthonormalize(basis)
    source_comp = (source @ b) @ b.T
    target_comp = (activation @ b) @ b.T
    return source_comp - target_comp
