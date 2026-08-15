"""
Tests for directional erasure of merged-model interference.

Exercises the public helpers exported from nnsight.modeling and applies
them through the standard trace interface on the tiny model, mirroring
the causal-vs-control design of the source paper: exact erasure along
the carried direction should undo the displacement, a norm-matched
wrong-direction control should not.
"""

import pytest
import torch

from nnsight.modeling import (
    directions_from_contrast,
    erase_along,
    erase_from_resid,
    interference_dose,
    shuffle_columns,
)


class TestEraseAlong:
    """Tensor-level properties of the projection."""

    def test_exact_erasure_removes_component(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 8)
        d = torch.randn(2, 3, 8)
        d = d / d.norm(dim=-1, keepdim=True)

        out = erase_along(x, d, coefficient=1.0)

        assert out.shape == x.shape
        # Orthogonal to the erased direction and norm shrunk, not grown.
        assert ((out * d).sum(dim=-1).abs().max()) < 1e-5
        assert (out.norm(dim=-1).max()) <= (x.norm(dim=-1).max()) + 1e-5

    def test_partial_dose_is_monotone(self):
        torch.manual_seed(1)
        x = torch.randn(4, 16)
        d = torch.randn(4, 16)
        d = d / d.norm(dim=-1, keepdim=True)

        removed = [
            (x - erase_along(x, d, coefficient=c)).norm(dim=-1).max()
            for c in (0.25, 0.5, 1.0)
        ]
        assert removed[0] < removed[1] < removed[2]

    def test_mismatched_hidden_dims_raise(self):
        with pytest.raises(ValueError):
            erase_along(torch.randn(2, 4), torch.randn(2, 5))


class TestDirectionalErasureTrace:
    """The causal-vs-control comparison through a real trace."""

    @torch.no_grad()
    def test_erasure_beats_matched_control(
        self, tiny_model, tiny_input: torch.Tensor
    ):
        other_input = torch.rand_like(tiny_input)

        with tiny_model.trace(tiny_input):
            clean_l1 = tiny_model.layer1.output.save()
            clean_out = tiny_model.output.save()

        with tiny_model.trace(other_input):
            other_l1 = tiny_model.layer1.output.save()
            other_out = tiny_model.output.save()

        # The merge proxy: displacement between the two states.
        directions = directions_from_contrast(clean_l1, other_l1)

        with tiny_model.trace(other_input):
            erase_from_resid(tiny_model.layer1, directions)
            erased = tiny_model.output.save()

        # Exact erasure along the carried direction removes the injected
        # displacement: the erased run should sit closer to the reference
        # trajectory than the untouched run.
        dose_untouched = interference_dose(clean_out, other_out).max()
        dose_erased = interference_dose(clean_out, erased).max()
        assert dose_erased < dose_untouched

        with tiny_model.trace(other_input):
            erase_from_resid(tiny_model.layer1, shuffle_columns(directions))
            control = tiny_model.output.save()

        dose_control = interference_dose(clean_out, control).max()
        assert dose_control > dose_erased + 1e-4
