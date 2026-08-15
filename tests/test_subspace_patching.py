"""
Tests for subspace activation patching (nnsight.modeling.subspace).

Covers the math helpers and the integrated two-invoke patching pattern from
docs/patterns/subspace-patching.md, including the same-rank random control
basis that guards against the interpretability illusion described by Makelov,
Lange & Nanda (2023).
"""

import pytest
import torch

import nnsight
from nnsight.modeling import (
    orthonormalize,
    project_component,
    random_basis,
    subspace_patch,
)


# =============================================================================
# Basis helpers
# =============================================================================


class TestBasisHelpers:
    """Pure-tensor tests for the projection / basis utilities."""

    def test_orthonormalize_columns_orthonormal(self):
        basis = torch.randn(16, 3)
        q = orthonormalize(basis)
        assert q.shape == (16, 3)
        assert torch.allclose(q.T @ q, torch.eye(3), atol=1e-5)

    def test_orthonormalize_idempotent(self):
        basis = torch.randn(8, 2)
        once = orthonormalize(basis)
        assert torch.allclose(once, orthonormalize(once), atol=1e-5)

    def test_orthonormalize_rejects_non_2d(self):
        with pytest.raises(ValueError):
            orthonormalize(torch.randn(4))

    def test_project_component_recovers_axis(self):
        basis = torch.eye(5)[:, :2]
        x = torch.randn(3, 5)
        proj = project_component(x, basis)
        assert torch.allclose(proj[..., :2], x[..., :2], atol=1e-5)
        assert torch.allclose(proj[..., 2:], torch.zeros(3, 3), atol=1e-5)

    def test_project_component_is_idempotent(self):
        basis = torch.linalg.qr(torch.randn(10, 3)).Q
        x = torch.randn(10)
        once = project_component(x, basis)
        assert torch.allclose(project_component(once, basis), once, atol=1e-5)

    def test_project_component_works_on_non_orthonormal_basis(self):
        # A spanning (but not orthonormal) basis must project identically:
        # right-multiplying by an invertible matrix preserves the column space.
        q = torch.linalg.qr(torch.randn(10, 2)).Q
        mix = torch.tensor([[1.0, 2.0], [0.0, 3.0]])
        x = torch.randn(10)
        assert torch.allclose(
            project_component(x, q @ mix), project_component(x, q), atol=1e-5
        )

    def test_random_basis_shapes_and_orthonormality(self):
        seed = torch.Generator().manual_seed(1)
        q = random_basis(32, 4, generator=seed)
        assert q.shape == (32, 4)
        assert torch.allclose(q.T @ q, torch.eye(4), atol=1e-5)

    def test_random_basis_reproducible(self):
        # Same seed -> same subspace (fresh generators, since a generator's
        # state advances on each draw).
        first = random_basis(16, 3, generator=torch.Generator().manual_seed(2))
        second = random_basis(16, 3, generator=torch.Generator().manual_seed(2))
        assert torch.allclose(first, second, atol=1e-6)

    def test_random_basis_rejects_bad_rank(self):
        with pytest.raises(ValueError):
            random_basis(8, 0)
        with pytest.raises(ValueError):
            random_basis(8, 9)

    def test_subspace_patch_zero_for_identical_vectors(self):
        basis = torch.linalg.qr(torch.randn(10, 2)).Q
        x = torch.randn(10)
        delta = subspace_patch(x, x.clone(), basis)
        assert torch.allclose(delta, torch.zeros(10), atol=1e-6)

    def test_subspace_patch_matches_full_vector_in_whole_space(self):
        # With basis = identity (full space), subspace patching reduces to
        # standard activation patching.
        dim = 6
        basis = torch.eye(dim)
        corrupt = torch.randn(2, dim)
        clean = torch.randn(2, dim)
        delta = subspace_patch(corrupt, clean, basis)
        assert torch.allclose(corrupt + delta, clean, atol=1e-5)

    def test_subspace_patch_leaves_complement_untouched(self):
        dim, rank = 12, 3
        basis = torch.linalg.qr(torch.randn(dim, rank)).Q
        # Complement basis: full space minus the patched subspace.
        comp = torch.linalg.qr(torch.randn(dim, dim - rank)).Q
        comp = comp - (basis @ (basis.T @ comp))
        corrupt = torch.randn(dim)
        clean = torch.randn(dim)
        patched = corrupt + subspace_patch(corrupt, clean, basis)
        # The component outside the patched subspace is unchanged.
        assert torch.allclose(
            (comp.T @ patched), (comp.T @ corrupt), atol=1e-5
        )
        # The component inside the subspace equals the clean run's.
        assert torch.allclose(basis.T @ patched, basis.T @ clean, atol=1e-5)


# =============================================================================
# Integrated patching on a real model
# =============================================================================


class TestSubspacePatchingOnModel:
    """The docs/patterns/subspace-patching.md recipe, end to end on GPT-2."""

    LAYER = 6
    RANK = 1

    def _last_token_probs(self, logits, token_ids):
        probs = logits.softmax(-1)[0]
        return [probs[t].item() for t in token_ids]

    @torch.no_grad()
    def test_subspace_patch_moves_logits_and_control_does_not(
        self, gpt2: nnsight.LanguageModel
    ):
        """Patch the clean-vs-corrupt delta into a rank-1 subspace.

        The aligned subspace (spanned by the clean/corrupt activation
        difference) must shift P(" Paris") above the unpatched baseline,
        while a random same-rank subspace must not — the null control from
        the paper.
        """
        clean = "The Eiffel Tower is in the city of"
        corrupt = "The Colosseum is in the city of"
        paris = gpt2.tokenizer.encode(" Paris")[0]

        # Step 1: capture both runs' last-token residuals to build the basis.
        with gpt2.trace() as tracer:
            with tracer.invoke(clean):
                clean_hs = gpt2.transformer.h[self.LAYER].output[:, -1, :].save()
            with tracer.invoke(corrupt):
                corrupt_hs = gpt2.transformer.h[self.LAYER].output[:, -1, :].save()

        direction = clean_hs[0] - corrupt_hs[0]
        aligned = orthonormalize(direction.unsqueeze(1))
        seed = torch.Generator().manual_seed(0)
        control = random_basis(direction.shape[0], self.RANK, generator=seed)

        assert aligned.shape == control.shape

        def patch_with(basis):
            with gpt2.trace() as tracer:
                barrier = tracer.barrier(2)
                with tracer.invoke(clean):
                    source = gpt2.transformer.h[self.LAYER].output[:, -1, :]
                    barrier()
                with tracer.invoke(corrupt):
                    barrier()
                    running = gpt2.transformer.h[self.LAYER].output[:, -1, :]
                    running += subspace_patch(running, source, basis)
                    patched = gpt2.lm_head.output[:, -1, :].save()
                with tracer.invoke(corrupt):
                    baseline = gpt2.lm_head.output[:, -1, :].save()
            return self._last_token_probs(patched, [paris])[0], self._last_token_probs(
                baseline, [paris]
            )[0]

        aligned_p, baseline_p = patch_with(aligned)
        control_p, baseline_p2 = patch_with(control)

        assert aligned_p > baseline_p, "aligned subspace patch should raise P(Paris)"
        assert control_p == pytest.approx(
            baseline_p2, abs=1e-5
        ), "random same-rank control should leave the logits unchanged"

    @torch.no_grad()
    def test_subspace_patch_preserves_complement_on_model(
        self, gpt2: nnsight.LanguageModel
    ):
        """Only the projected component of the corrupt activation changes."""
        clean = "The Eiffel Tower is in the city of"
        corrupt = "The Colosseum is in the city of"

        with gpt2.trace() as tracer:
            with tracer.invoke(clean):
                clean_hs = gpt2.transformer.h[self.LAYER].output[:, -1, :].save()
            with tracer.invoke(corrupt):
                corrupt_hs = gpt2.transformer.h[self.LAYER].output[:, -1, :].save()

        dim = clean_hs.shape[-1]
        direction = clean_hs[0] - corrupt_hs[0]
        aligned = orthonormalize(direction.unsqueeze(1))

        with gpt2.trace() as tracer:
            barrier = tracer.barrier(2)
            with tracer.invoke(clean):
                source = gpt2.transformer.h[self.LAYER].output[:, -1, :]
                barrier()
            with tracer.invoke(corrupt):
                barrier()
                running = gpt2.transformer.h[self.LAYER].output[:, -1, :]
                running += subspace_patch(running, source, aligned)
                patched_hs = gpt2.transformer.h[self.LAYER].output[:, -1, :].save()

        expected = corrupt_hs[0] + project_component(
            clean_hs[0] - corrupt_hs[0], aligned
        )
        assert torch.allclose(patched_hs[0], expected, atol=1e-4)
        # Rank-1 patch: the residual difference must be rank-1-shaped.
        diff = (patched_hs[0] - corrupt_hs[0]).abs()
        assert diff.sum() > 0
