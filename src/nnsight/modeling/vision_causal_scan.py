"""Causal tracing of vision-token contributions in vision-language models.

Which visual patch actually drove the answer?  The naive assumption is that
the patches lying inside the object you asked about are the causal ones.
"Where To Look? Causal Tracing of Vision Encoders in VLM" (arXiv:2608.10758)
shows this assumption fails: corrupt the image, patch one *region's* vision
tokens from the clean run back into the corrupted run, and the tokens that
restore the answer are frequently outside the queried region.  Strong
captioning and spatially-localized causal representations are not the same
thing.

The experiment is a per-(region, layer) causal mediation sweep over the
vision tower, composed from the primitives nnsight already documents: a
clean run that captures activations (``docs/patterns/ablation.md`` calls
this "pass 1" of mean ablation), then a corrupted run that writes them back
in.  What is missing is the loop that turns "patch one activation" into a
map over an image, so that is what this module contributes.

Two implementation notes, both forced by where this code lives:

- The two passes are *separate* ``model.trace()`` blocks rather than two
  ``tracer.invoke`` calls joined by a ``tracer.barrier``.  Cross-invoke
  value sharing resolves against the caller's Python frame
  (``get_non_nnsight_frame``, ``intervention/tracing/util.py``), which
  skips frames belonging to the ``nnsight`` package itself -- so a barrier
  inside this module cannot hand a value between invokes.  Splitting the
  passes makes the captured activations ordinary Python values instead,
  which is also the shape the ablation recipe uses for mean ablation.
- The trace bodies are generated source, registered with ``linecache`` and
  executed, because nnsight recovers a ``with model.trace(...)`` block via
  ``inspect.getsourcelines`` on the frame that opened it.  This is the same
  trick ``intervention/serialization.py`` uses for deserialized functions.

Adapted port (Mode 2).  The per-(region, layer) patching sweep is kept at
full fidelity.  Substituted for target-native equivalents: the paper's VLM
benchmark suite is replaced by a next-token log-probability effect size on
whatever model the caller passes, so any nnsight model with a sequential
module list works -- including text-only ``LanguageModel``s where "regions"
are arbitrary token-index groups.

Example -- an LLaVA-style VLM, is the answer driven by the region you
asked about?::

    from nnsight import VisionLanguageModel
    from nnsight.modeling import CausalScan, grid_regions
    from nnsight.modeling.vision_causal_scan import mask_pixels

    model = VisionLanguageModel(
        "llava-hf/llava-interleave-qwen-0.5b-hf", device_map="auto", dispatch=True
    )
    # 336px image at 14px patches -> a 24x24 grid of 576 vision tokens.
    regions = grid_regions(n_tokens=576, patch_grid=24, region_grid=3)
    prompt = "<image>\nWhat color is the traffic light?"

    scan = CausalScan(
        layer_path="model.vision_tower.vision_model.encoder.layers",
        layers=(6, 12, 18),
        regions=regions,
        answer=model.tokenizer.encode(" red")[0],
        target_region=4,                       # the region you asked about
    )

    # Pass 1 -- capture every region's activations, every layer, clean run.
    clean = scan.capture(model, prompt=prompt, images=[img])

    # Pass 2 -- one corrupted image per region, patched back in.
    corrupt_kwargs = {
        region["id"]: {
            "prompt": prompt,
            "images": [mask_pixels(pixels.clone(), region, patch_grid=24)],
        }
        for region in regions
    }
    report = scan.patch(model, clean, corrupt_kwargs=corrupt_kwargs)
    print(report.inside.mean(), report.outside.mean())
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ..intervention.serialization import _register_source_with_linecache

__all__ = [
    "CausalScan",
    "CausalScanReport",
    "grid_regions",
    "logprob_effect",
    "mask_pixels",
    "shuffle_pixels",
]


# ---------------------------------------------------------------------------
# Regions: a part of an image expressed as a set of sequence positions
# ---------------------------------------------------------------------------


def grid_regions(
    n_tokens: int,
    patch_grid: int,
    region_grid: Optional[int] = None,
    prefix_len: int = 0,
) -> List[Dict[str, Any]]:
    """Tile ``n_tokens`` vision-token positions into a coarse grid of regions.

    Vision towers emit one token per image patch, laid out on a square grid.
    This maps that layout back to sequence positions so that a *region* of
    the image becomes the set of positions feeding it.

    Args:
        n_tokens: Number of vision tokens.  Must be ``patch_grid ** 2`` for
            towers with no patch merging; extra positions are left
            unassigned.
        patch_grid: Side length of the patch grid.  A 336px image at 14px
            patches gives ``patch_grid=18``.
        region_grid: Side length of the *region* grid.  Defaults to
            ``patch_grid`` (one region per patch).  ``region_grid=3`` over
            ``patch_grid=18`` yields 9 regions of 36 tokens each.
        prefix_len: Sequence positions before the first vision token (BOS,
            special tokens, leading text).  Added to every emitted position.
            Trailing text is *not* accounted for -- the region grid is
            counted from the first vision token.

    Returns:
        One dict per region with ``id``, ``row``, ``col``, the grid cell
        ``r0``/``r1``/``c0``/``c1`` (in patch-grid units), and ``positions``,
        the flat list of sequence positions in that cell.
    """
    if patch_grid < 1:
        raise ValueError(f"patch_grid must be >= 1, got {patch_grid}")
    if patch_grid**2 != n_tokens:
        raise ValueError(
            f"n_tokens ({n_tokens}) does not equal patch_grid**2 "
            f"({patch_grid**2}); pass the true grid side"
        )
    if region_grid is None:
        region_grid = patch_grid
    if not 1 <= region_grid <= patch_grid:
        raise ValueError(
            f"region_grid ({region_grid}) must be in [1, patch_grid={patch_grid}]"
        )
    if patch_grid % region_grid:
        raise ValueError(
            f"patch_grid ({patch_grid}) is not divisible by region_grid ({region_grid})"
        )

    step = patch_grid // region_grid
    regions: List[Dict[str, Any]] = []
    for rid, (r0, c0) in enumerate(
        itertools.product(range(0, patch_grid, step), range(0, patch_grid, step))
    ):
        # Patch tokens are row-major: patch (r, c) sits at r * patch_grid + c.
        positions = [
            r * patch_grid + c
            for r in range(r0, r0 + step)
            for c in range(c0, c0 + step)
            if r * patch_grid + c < n_tokens
        ]
        regions.append(
            {
                "id": rid,
                "row": r0 // step,
                "col": c0 // step,
                "r0": r0,
                "r1": r0 + step,
                "c0": c0,
                "c1": c0 + step,
                "positions": [p + prefix_len for p in positions],
            }
        )
    return regions


# ---------------------------------------------------------------------------
# Corruption operators -- the paper's masking / shuffling perturbations
# ---------------------------------------------------------------------------


def _pixel_box(
    image: torch.Tensor, region: Mapping[str, int], patch_grid: int
) -> Tuple[int, int, int, int]:
    """Map a region's grid cell onto a ``[y0:y1, x0:x1]`` pixel box."""
    h, w = image.shape[-2], image.shape[-1]
    return (
        round(region["r0"] * h / patch_grid),
        round(region["r1"] * h / patch_grid),
        round(region["c0"] * w / patch_grid),
        round(region["c1"] * w / patch_grid),
    )


def mask_pixels(
    image: torch.Tensor,
    region: Mapping[str, int],
    patch_grid: int,
    fill: float = 0.0,
) -> torch.Tensor:
    """Black out (or fill) one region of a ``[C, H, W]`` pixel tensor, in place.

    With ``fill=0`` the region is masked to black, the paper's masking
    corruption.  A nonzero ``fill`` gives a uniform-color mask.
    """
    y0, y1, x0, x1 = _pixel_box(image, region, patch_grid)
    image[..., y0:y1, x0:x1] = fill
    return image


def shuffle_pixels(
    image: torch.Tensor,
    region: Mapping[str, int],
    patch_grid: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Shuffle one region's pixels in place.

    Shuffling preserves the region's pixel statistics (so it is not pure
    drop-out) while destroying its spatial structure -- the paper uses this
    to separate "uses the content" from "uses the layout" of a region.
    """
    y0, y1, x0, x1 = _pixel_box(image, region, patch_grid)
    window = image[..., y0:y1, x0:x1]
    flat = window.reshape(window.shape[0], -1)
    perm = torch.randperm(flat.shape[-1], generator=generator)
    image[..., y0:y1, x0:x1] = flat.index_select(-1, perm).reshape(window.shape)
    return image


# ---------------------------------------------------------------------------
# Effect size: what did the patch do to the answer?
# ---------------------------------------------------------------------------


def logprob_effect(baseline: torch.Tensor, patched: torch.Tensor, answer: int) -> float:
    """Log-probability effect of a patch on one answer token.

    Positive means the patch moved probability mass *toward* the answer, so
    the patched region carried clean-answer information.  This is the
    paper's causal-tracing score restricted to a single target token; both
    inputs are read at the last position.
    """
    base_lp = torch.log_softmax(baseline[-1].float(), dim=-1)[answer]
    patched_lp = torch.log_softmax(patched[-1].float(), dim=-1)[answer]
    return (patched_lp - base_lp).item()


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CausalScanReport:
    """Per-(layer, region) causal effect of restoring a corrupted region.

    ``effects[layer][region_id]`` is the log-probability effect of patching
    that region's vision tokens from the clean run into the corrupted run.
    ``inside`` / ``outside`` split those effects by whether the region is the
    one the question was about -- the comparison the paper makes to show
    that highly causal vision tokens often lie outside the target region.
    """

    effects: Dict[int, Dict[int, float]]
    regions: List[Dict[str, Any]]
    layers: Tuple[int, ...]
    target_region: Optional[int]
    baselines: Dict[int, float] = dataclasses.field(default_factory=dict)

    def _split(self, match: bool) -> torch.Tensor:
        values = [
            effect
            for layer in self.layers
            for rid, effect in self.effects.get(layer, {}).items()
            if (rid == self.target_region) is match
        ]
        return torch.tensor(values or [0.0])

    @property
    def inside(self) -> torch.Tensor:
        """Effects for the region the question was about."""
        return self._split(match=True)

    @property
    def outside(self) -> torch.Tensor:
        """Effects for every other region."""
        return self._split(match=False)

    def top(self, layer: int, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """The ``k`` most causal regions at ``layer``, most causal first."""
        ranked = sorted(self.effects[layer].items(), key=lambda kv: -kv[1])
        return [(effect, self.regions[rid]) for rid, effect in ranked[:k]]

    def argmax_region(self, layer: int) -> Dict[str, Any]:
        """The single most causal region at ``layer``."""
        return self.top(layer, k=1)[0][1]


@dataclasses.dataclass
class CausalScan:
    """A (region, layer) causal patching sweep over a sequential module list.

    The sweep runs in two passes, matching the two-pass shape the ablation
    recipe documents for mean ablation (``docs/patterns/ablation.md``):

    1. :meth:`capture` -- one clean trace, saving every region's
       activations at every layer.
    2. :meth:`patch` -- one trace per corrupted condition, writing one
       region's captured activations back in and reading the effect on the
       answer.

    Splitting the passes avoids the cross-invoke barrier entirely: each
    pass is a plain ``with model.trace(...)`` block, so the captured
    activations travel as ordinary Python values rather than across
    ``tracer.invoke`` boundaries.

    Args:
        layer_path: Dotted path under the model to the sequential module
            holding the layers to sweep, e.g.
            ``"model.vision_tower.vision_model.encoder.layers"``.
        layers: Layer indices into that module (negative counts from the
            end).
        regions: Region descriptors from :func:`grid_regions`, or your own
            mappings with a ``positions`` list.
        answer: Target token id the patch should restore.
        readout_path: Dotted path to the module whose output is scored.
            Read at the last position.
        output_index: Index applied to a block's output before slicing
            positions, for architectures whose blocks return a tuple
            (GPT-2 style in ``transformers<5.0``).  ``None`` for tensor
            outputs.
        target_region: Region id the question is *about*, used to split
            ``report.inside`` from ``report.outside``.
    """

    layer_path: str
    layers: Sequence[int]
    regions: Sequence[Mapping[str, Any]]
    answer: int
    readout_path: str = "lm_head"
    output_index: Optional[int] = None
    target_region: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("regions must be non-empty")
        if not self.layers:
            raise ValueError("layers must be non-empty")
        self.layers = tuple(self.layers)
        self.regions = [dict(region) for region in self.regions]
        for region in self.regions:
            if "positions" not in region:
                raise KeyError(f"region {region.get('id')!r} has no 'positions'")

    # -- pass 1 ------------------------------------------------------------

    def capture(self, model: Any, **invoke_kwargs: Any) -> Dict[int, Dict[int, Any]]:
        """Run the clean pass and capture each region's activations.

        Args:
            model: Any nnsight model -- ``VisionLanguageModel`` for the
                paper's setting, ``LanguageModel`` for the text-only
                analogue.
            **invoke_kwargs: Forwarded to ``model.trace()`` (the prompt,
                ``images=``, ...).

        Returns:
            ``{layer: {region_id: tensor}}`` of saved activations.  Pass to
            :meth:`patch`.
        """
        return _capture_activations(
            model,
            layer_path=self.layer_path,
            layers=self.layers,
            regions=self.regions,
            output_index=self.output_index,
            **invoke_kwargs,
        )

    # -- pass 2 ------------------------------------------------------------

    def patch(
        self,
        model: Any,
        captured: Mapping[int, Mapping[int, Any]],
        baseline_kwargs: Optional[Mapping[str, Any]] = None,
        corrupt_kwargs: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **baseline_invoke_kwargs: Any,
    ) -> CausalScanReport:
        """Run the corrupted pass and score each (layer, region) patch.

        For each corrupted condition the patch, the patched readout, and a
        corrupt-only baseline are three invokes in a single trace -- the
        side-by-side shape from ``docs/patterns/multi-prompt-comparison.md``.
        The captured clean activations are already plain tensors by now, so
        no invoke needs to wait on another and no barrier is required.

        Args:
            model: The same model :meth:`capture` ran on.
            captured: The dict returned by :meth:`capture`.
            baseline_kwargs: Keyword arguments for the corrupt-only baseline
                invoke.  Defaults to the first condition's -- usually all
                conditions share one baseline because they differ only in
                which region is patched.
            corrupt_kwargs: One keyword-argument dict per region id for the
                corrupted invokes.  Defaults to ``baseline_kwargs`` for
                every region, which measures pure activation restoration
                (the "denoise" direction) rather than pixel corruption.
            **baseline_invoke_kwargs: Keyword arguments for the baseline
                invoke when ``baseline_kwargs`` is not given.

        Returns:
            :class:`CausalScanReport`.
        """
        if baseline_kwargs is None:
            baseline_kwargs = dict(baseline_invoke_kwargs)
        if corrupt_kwargs is None:
            corrupt_kwargs = {
                region["id"]: dict(baseline_kwargs) for region in self.regions
            }

        missing = [
            region["id"]
            for region in self.regions
            if region["id"] not in corrupt_kwargs
        ]
        if missing:
            raise KeyError(f"no corrupt_kwargs for region id(s) {missing}")

        effects: Dict[int, Dict[int, float]] = {}
        baselines: Dict[int, float] = {}

        for region in self.regions:
            positions = list(region["positions"])
            if not positions:
                continue

            for layer in self.layers:
                saved = captured[layer][region["id"]]
                effect, baseline = _patch_one(
                    model,
                    layer_path=self.layer_path,
                    layer=layer,
                    positions=positions,
                    saved=saved,
                    answer=self.answer,
                    readout_path=self.readout_path,
                    output_index=self.output_index,
                    baseline_kwargs=baseline_kwargs,
                    corrupt_kwargs=corrupt_kwargs[region["id"]],
                )
                effects.setdefault(layer, {})[region["id"]] = effect
                baselines.setdefault(layer, baseline)

        return CausalScanReport(
            effects=effects,
            regions=list(self.regions),
            layers=self.layers,
            target_region=self.target_region,
            baselines=baselines,
        )


def _capture_activations(
    model: Any,
    layer_path: str,
    layers: Sequence[int],
    regions: Sequence[Mapping[str, Any]],
    output_index: Optional[int],
    **invoke_kwargs: Any,
) -> Dict[int, Dict[int, Any]]:
    """Clean pass: save every region's activations at every layer.

    Implemented by emitting and executing a trace body, so that the module
    access expressions are evaluated in the *caller's* frame -- see the
    module docstring for why that matters.
    """
    layer_list = _resolve(model, layer_path)

    body: List[str] = []
    for layer in layers:
        body.append(f"    captured[{layer!r}] = {{}}")
        block = f"block_{layer}"
        body.append(f"    {block} = layers[{layer!r}]")
        for region in regions:
            positions = list(region["positions"])
            if not positions:
                continue
            name = f"saved_{layer}_{region['id']}"
            if output_index is None:
                src = f"{block}.output[:, {positions!r}, :]"
            else:
                src = f"{block}.output[{output_index!r}][:, {positions!r}, :]"
            body.append(f"    {name} = {src}.clone().save()")
            body.append(f"    captured[{layer!r}][{region['id']!r}] = {name}")

    source = "\n".join(
        [
            "captured = {}",
            "with model.trace(*args, **kwargs) as tracer:",
            *body,
        ]
    )
    args, kwargs = _split_input(invoke_kwargs)
    namespace = {
        "model": model,
        "layers": layer_list,
        "args": args,
        "kwargs": kwargs,
    }
    _run_trace_source(_CAPTURE_FILENAME, source, namespace)
    return namespace["captured"]


def _patch_one(
    model: Any,
    layer_path: str,
    layer: int,
    positions: Sequence[int],
    saved: torch.Tensor,
    answer: int,
    readout_path: str,
    output_index: Optional[int],
    baseline_kwargs: Mapping[str, Any],
    corrupt_kwargs: Mapping[str, Any],
) -> Tuple[float, float]:
    """One (region, layer): patch the saved activations into a corrupted run."""
    block = _resolve(model, f"{layer_path}.{layer}")
    readout = _resolve(model, readout_path)
    pos = list(positions)

    lines = [
        "with model.trace() as tracer:",
        "    with tracer.invoke(*corrupt_args, **corrupt_rest):",
    ]
    if output_index is None:
        lines.append(f"        block.output[:, {pos!r}, :] = saved")
    else:
        lines.append(f"        block.output[{output_index!r}][:, {pos!r}, :] = saved")
    lines += [
        "        patched = readout.output[:, -1, :].save()",
        "    with tracer.invoke(*base_args, **base_rest):",
        "        baseline = readout.output[:, -1, :].save()",
    ]

    base_args, base_rest = _split_input(baseline_kwargs)
    corrupt_args, corrupt_rest = _split_input(corrupt_kwargs)

    namespace = {
        "model": model,
        "block": block,
        "readout": readout,
        "saved": saved,
        "base_args": base_args,
        "base_rest": base_rest,
        "corrupt_args": corrupt_args,
        "corrupt_rest": corrupt_rest,
    }
    _run_trace_source(_PATCH_FILENAME, "\n".join(lines), namespace)

    baseline = namespace["baseline"]
    patched = namespace["patched"]
    baseline_lp = torch.log_softmax(baseline[-1].float(), dim=-1)[answer].item()
    return logprob_effect(baseline, patched, answer), baseline_lp


def _split_input(
    kwargs: Mapping[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Split an invoke's kwargs into ``(positional_input, forwarded_kwargs)``.

    The first entry is used as the positional prompt so that
    ``LanguageModel``'s single-positional-input contract holds; the rest are
    forwarded as keywords (``images=``, ``input_ids=``, ...).
    """
    rest = dict(kwargs)
    prompt = rest.pop("prompt", None)
    return () if prompt is None else (prompt,), rest


# Pseudo-filenames for the generated trace bodies.  They deliberately do NOT
# start with "<nnsight": nnsight's source capture treats such filenames as
# already-generated frames and refuses to look them up in linecache.  Each
# distinct body gets its own filename, so a shorter sweep registering after
# a longer one cannot leave stale lines behind in a shared linecache entry.
_CAPTURE_FILENAME = "<causal_scan_capture>"
_PATCH_FILENAME = "<causal_scan_patch>"


def _run_trace_source(filename: str, source: str, namespace: Dict[str, Any]) -> None:
    """Compile and run a generated trace body in ``namespace``.

    nnsight recovers the ``with model.trace(...)`` block by calling
    ``inspect.getsourcelines`` on the executing frame, so the generated
    source has to be reachable that way.  Registering it with ``linecache``
    (the same trick ``intervention/serialization.py`` uses for deserialized
    functions) makes that work without writing anything to disk.

    Because the body runs at module level of the generated "file", the
    values it produces land in ``namespace`` directly.
    """
    # One filename per distinct body: linecache merges entries that share a
    # filename, which would misalign line numbers across sweeps of
    # different sizes.
    key = abs(hash(source))
    _register_source_with_linecache(f"{filename}_{key}", source)
    exec(compile(source, f"{filename}_{key}", "exec"), namespace)


def _resolve(model: Any, path: str) -> Any:
    """Walk a dotted path under ``model``, indexing numeric segments."""
    node = model
    for part in path.split("."):
        node = node[int(part)] if part.lstrip("-").isdigit() else getattr(node, part)
    return node
