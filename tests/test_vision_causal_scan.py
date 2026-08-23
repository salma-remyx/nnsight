"""
Tests for the vision-token causal tracing sweep.

These tests cover the pieces of
``nnsight.modeling.vision_causal_scan``:

- ``grid_regions`` tiling a patch grid into coarse regions
- ``mask_pixels`` / ``shuffle_pixels`` corruption operators
- ``CausalScan.capture`` / ``CausalScan.patch`` running the two-pass sweep

The sweep is model-agnostic, so the integration tests run it on GPT-2 via
``nnsight.LanguageModel`` (the same model the patterns cookbook uses) with
token-index groups standing in for image regions: region 0 is the subject
span that carries the answer's information, region 1 is an unrelated
position.  Patching the subject span back into a corrupted prompt should
help the answer more than patching the unrelated position -- the same
inside-vs-outside comparison the module reports for vision tokens.
"""

import pytest
import torch
import nnsight

from nnsight.modeling import CausalScan, grid_regions
from nnsight.modeling.vision_causal_scan import (
    CausalScan as _CausalScan,
    logprob_effect,
    mask_pixels,
    shuffle_pixels,
)


CLEAN = "The Eiffel Tower is in the city of"
CORRUPT = "The Colosseum is in the city of"

# Region 0 = " Eiffel Tower" (the subject carrying the answer); region 1 =
# " city" (an unrelated later position, the control).
SUBJECT_REGION = {"id": 0, "row": 0, "col": 0, "positions": [1, 2, 3]}
CONTROL_REGION = {"id": 1, "row": 0, "col": 1, "positions": [8]}
LAYERS = (5, 7, 9)


# =============================================================================
# Public wiring
# =============================================================================


class TestPublicExports:
    """The sweep is reachable from the ``nnsight.modeling`` package."""

    def test_exported_names(self):
        """``nnsight.modeling`` re-exports the sweep's entry points."""
        assert CausalScan is _CausalScan
        assert callable(grid_regions)

    def test_export_does_not_pull_transformers(self):
        """Importing the sweep's module alone does not import transformers.

        This is the property the lazy-import work protects: a light module
        in ``nnsight.modeling`` must not drag the heavy model classes in.
        """
        import subprocess
        import sys

        probe = (
            "import sys; "
            "from nnsight.modeling.vision_causal_scan import CausalScan; "
            "assert 'transformers' not in sys.modules; "
            "assert 'nnsight.modeling.language' not in sys.modules; "
            "print('clean')"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True
        )

        assert out.stdout.strip() == "clean", out.stderr


# =============================================================================
# Region tiling
# =============================================================================


class TestGridRegions:
    """Tests for mapping a patch grid onto sequence positions."""

    def test_region_count_and_positions(self):
        """A 6x6 patch grid at 3x3 regions gives 9 regions of 4 tokens."""
        regions = grid_regions(n_tokens=36, patch_grid=6, region_grid=3)

        assert len(regions) == 9
        assert all(len(region["positions"]) == 4 for region in regions)

    def test_positions_cover_token_span_exactly_once(self):
        """Every vision token belongs to exactly one region."""
        regions = grid_regions(n_tokens=36, patch_grid=6, region_grid=3)

        covered = sorted(p for region in regions for p in region["positions"])

        assert covered == list(range(36))

    def test_default_region_grid_is_one_region_per_patch(self):
        """Omitting region_grid gives one region per patch."""
        regions = grid_regions(n_tokens=16, patch_grid=4)

        assert len(regions) == 16
        assert regions[5]["positions"] == [5]

    def test_prefix_len_shifts_positions(self):
        """prefix_len accounts for special tokens before the vision span."""
        regions = grid_regions(n_tokens=16, patch_grid=4, prefix_len=7)

        assert min(p for p in regions[0]["positions"]) == 7
        assert max(p for p in regions[-1]["positions"]) == 7 + 15

    def test_row_major_layout(self):
        """Patch (r, c) sits at r * patch_grid + c, scanned row-major."""
        regions = grid_regions(n_tokens=36, patch_grid=6, region_grid=3)

        # Region in the top-right corner covers patches (0,4),(0,5),(1,4),(1,5).
        assert regions[2]["positions"] == [4, 5, 10, 11]

    @pytest.mark.parametrize(
        "n_tokens, patch_grid, region_grid",
        [(35, 6, 3), (36, 6, 4), (16, 4, 3), (16, 0, None)],
    )
    def test_invalid_grids_raise(self, n_tokens, patch_grid, region_grid):
        """Mismatched grid arguments are rejected up front."""
        with pytest.raises(ValueError):
            grid_regions(
                n_tokens=n_tokens, patch_grid=patch_grid, region_grid=region_grid
            )


# =============================================================================
# Corruption operators
# =============================================================================


class TestCorruptionOperators:
    """Tests for the pixel-space corruption operators."""

    def test_mask_pixels_zeros_only_the_region(self):
        """Masking writes zeros inside the region's pixel box and nothing else."""
        image = torch.arange(3 * 6 * 6, dtype=torch.float32).reshape(3, 6, 6)
        region = grid_regions(n_tokens=36, patch_grid=6, region_grid=3)[0]

        masked = mask_pixels(image.clone(), region, patch_grid=6)

        assert torch.equal(masked[:, :2, :2], torch.zeros(3, 2, 2))
        assert torch.equal(masked[:, 2:, :], image[:, 2:, :])
        assert torch.equal(masked[:, :, 2:], image[:, :, 2:])

    def test_mask_pixels_fill(self):
        """A nonzero fill gives a uniform-color mask."""
        image = torch.zeros(3, 6, 6)
        region = grid_regions(n_tokens=36, patch_grid=6, region_grid=3)[4]

        masked = mask_pixels(image, region, patch_grid=6, fill=0.5)

        assert torch.equal(masked[:, 2:4, 2:4], torch.full((3, 2, 2), 0.5))

    def test_shuffle_pixels_preserves_statistics(self):
        """Shuffling permutes the region's pixels, keeping its statistics."""
        image = torch.arange(3 * 6 * 6, dtype=torch.float32).reshape(3, 6, 6)
        region = grid_regions(n_tokens=36, patch_grid=6, region_grid=3)[4]

        generator = torch.Generator().manual_seed(0)
        shuffled = shuffle_pixels(
            image.clone(), region, patch_grid=6, generator=generator
        )

        assert shuffled.sum() == image.sum()
        assert not torch.equal(shuffled[:, 2:4, 2:4], image[:, 2:4, 2:4])
        # Outside the region, nothing moves.
        assert torch.equal(shuffled[:, :2, :], image[:, :2, :])


# =============================================================================
# Effect size
# =============================================================================


class TestLogprobEffect:
    """Tests for the answer-token effect size."""

    def test_zero_when_unchanged(self):
        """Identical logits give zero effect."""
        logits = torch.randn(2, 50)

        assert logprob_effect(logits, logits.clone(), answer=3) == 0.0

    def test_sign_follows_the_answer(self):
        """Raising the answer's logit gives a positive effect."""
        baseline = torch.zeros(1, 10)
        patched = torch.zeros(1, 10)
        patched[0, 4] = 5.0

        assert logprob_effect(baseline, patched, answer=4) > 0
        assert logprob_effect(baseline, patched, answer=5) < 0


# =============================================================================
# The sweep, end to end
# =============================================================================


class TestCausalScanSweep:
    """Integration tests running the two-pass sweep through a real model."""

    def test_capture_returns_saved_activations(self, gpt2: nnsight.LanguageModel):
        """The clean pass saves one tensor per (layer, region)."""
        scan = CausalScan(
            layer_path="transformer.h",
            layers=LAYERS,
            regions=[SUBJECT_REGION, CONTROL_REGION],
            answer=0,
        )

        captured = scan.capture(gpt2, prompt=CLEAN)

        assert sorted(captured.keys()) == list(LAYERS)
        for layer in LAYERS:
            assert sorted(captured[layer].keys()) == [0, 1]
            assert isinstance(captured[layer][0], torch.Tensor)
            assert captured[layer][0].shape == (1, 3, gpt2.config.n_embd)
            assert captured[layer][1].shape == (1, 1, gpt2.config.n_embd)

    def test_report_shape_and_accessors(self, gpt2: nnsight.LanguageModel):
        """The report exposes per-layer effects and the inside/outside split."""
        answer = gpt2.tokenizer.encode(" Paris")[0]
        scan = CausalScan(
            layer_path="transformer.h",
            layers=LAYERS,
            regions=[SUBJECT_REGION, CONTROL_REGION],
            answer=answer,
            target_region=0,
        )
        captured = scan.capture(gpt2, prompt=CLEAN)

        report = scan.patch(gpt2, captured, prompt=CORRUPT)

        assert sorted(report.effects.keys()) == list(LAYERS)
        for layer in LAYERS:
            assert sorted(report.effects[layer].keys()) == [0, 1]
            assert all(isinstance(v, float) for v in report.effects[layer].values())

        # One effect per layer on each side of the split.
        assert report.inside.shape == (len(LAYERS),)
        assert report.outside.shape == (len(LAYERS),)
        assert report.target_region == 0
        assert report.argmax_region(7)["id"] in (0, 1)

    def test_subject_region_beats_control(self, gpt2: nnsight.LanguageModel):
        """Restoring the subject span helps the answer more than the control.

        This is the signal the module exists to measure: patching the
        positions that carry the answer's information moves probability
        toward the answer, while patching an unrelated position does much
        less.
        """
        answer = gpt2.tokenizer.encode(" Paris")[0]
        scan = CausalScan(
            layer_path="transformer.h",
            layers=LAYERS,
            regions=[SUBJECT_REGION, CONTROL_REGION],
            answer=answer,
            target_region=0,
        )
        captured = scan.capture(gpt2, prompt=CLEAN)

        report = scan.patch(gpt2, captured, prompt=CORRUPT)

        for layer in LAYERS:
            assert report.effects[layer][0] > report.effects[layer][1]

        assert (report.inside > report.outside).all()

    def test_patching_the_whole_span_restores_the_answer(
        self, gpt2: nnsight.LanguageModel
    ):
        """Patching every position recovers a large share of the clean answer.

        A full-sequence patch is the upper bound on what any single region
        can achieve, so it should move the corrupt run much further toward
        the clean answer than any partial patch does.
        """
        answer = gpt2.tokenizer.encode(" Paris")[0]
        whole = {"id": 0, "row": 0, "col": 0, "positions": list(range(10))}
        scan = CausalScan(
            layer_path="transformer.h", layers=(3,), regions=[whole], answer=answer
        )
        captured = scan.capture(gpt2, prompt=CLEAN)

        report = scan.patch(gpt2, captured, prompt=CORRUPT)

        assert report.effects[3][0] > 1.0

    def test_validation(self, gpt2: nnsight.LanguageModel):
        """Empty layers / regions and missing 'positions' are rejected."""
        with pytest.raises(ValueError):
            CausalScan(
                layer_path="transformer.h",
                layers=(),
                regions=[SUBJECT_REGION],
                answer=0,
            )
        with pytest.raises(ValueError):
            CausalScan(layer_path="transformer.h", layers=LAYERS, regions=[], answer=0)
        with pytest.raises(KeyError):
            CausalScan(
                layer_path="transformer.h",
                layers=LAYERS,
                regions=[{"id": 0}],
                answer=0,
            )
