"""KV-cache steering: one-shot key/value interventions for generation.

Implements the steering surface of *cache steering* (arXiv 2507.08799,
"KV Cache Steering for Inducing Reasoning in Small Language Models"):
instead of nudging the residual stream on every generation step, a
precomputed direction is added **once** to the attention key/value
projections during the prefill step. Because every later decode step
attends over the prefilled cache, the intervention persists through the
rest of the generation with no per-step work — one intervention, effect
on every generated token.

The steering direction is computed the same way residual-stream
directions are (see ``docs/patterns/steering.md``): the difference of
mean projection outputs over a positive vs negative prompt set,
normalized to unit norm. The paper constructs its directions from
GPT-4o-generated reasoning traces vs direct answers; that data
collection is out of scope here — bring your own contrast sets.

Usage::

    from nnsight import LanguageModel
    from nnsight.modeling.kv_steering import KVSteering

    model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
    steering = KVSteering(model, layer=6)

    direction = steering.direction(
        positive=["Let me think step by step. 2 + 2 = 4."],
        negative=["The answer is 4."],
    )

    with model.generate("Q: What is 3 + 5? A:", max_new_tokens=20) as tracer:
        steering.apply(tracer, direction, coef=1.5)
        output = tracer.result.save()

    print(model.tokenizer.decode(output[0]))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Sequence

import torch

if TYPE_CHECKING:
    from .language import LanguageModel


def _proj_output(module):
    """The projection output as a plain ``[batch, seq, hidden]`` tensor.

    Linear outputs are bare tensors; a few architectures wrap them in a
    tuple, so unwrap when needed. The returned tensor aliases whatever
    the model holds onto, so in-place writes propagate.
    """
    out = module.output
    return out[0] if isinstance(out, tuple) else out


def _proj_view(output: torch.Tensor, idx) -> torch.Tensor:
    """The k or v chunk of a projection output, given a fused index."""
    if idx is None:
        return output
    hidden = output.shape[-1] // 3
    return output[..., idx * hidden : (idx + 1) * hidden]  # noqa: E203


def _mean_output(acts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean over prompts and positions of saved projection outputs."""
    return torch.cat([a.reshape(-1, a.shape[-1]) for a in acts], dim=0).mean(0)


class KVSteering:
    """Add precomputed directions to a layer's ``k_proj``/``v_proj`` outputs.

    Targets attention projections rather than the residual stream. The
    intervention fires during prefill (generation step 0), writing into
    the KV cache that all subsequent decode steps attend over.
    """

    def __init__(self, model: "LanguageModel", layer: int) -> None:
        """Set up steering for one decoder layer of ``model``.

        Args:
            model: A dispatched ``LanguageModel`` (or any ``TransformersModel``
                whose decoder attention exposes ``k_proj`` / ``v_proj``).
            layer: Decoder layer index to steer at.
        """
        self.model = model
        self.layer = layer

    def _attn(self):
        """The attention submodule for the configured layer.

        Resolves GPT-2-style ``transformer.h[L].attn`` and Llama-style
        ``model.layers[L].self_attn`` layouts.
        """
        if hasattr(self.model, "transformer"):
            return self.model.transformer.h[self.layer].attn
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer].self_attn
        raise AttributeError(
            f"Could not locate attention submodule at layer {self.layer} on "
            f"{type(self.model).__name__}. Use `print(model)` to find the "
            "decoder attention path and construct KVSteering around it."
        )

    def _projection(self, proj: str):
        """Resolve a projection module to steer and how to slice it.

        Returns ``(module, slice)`` where ``slice`` is ``None`` for
        architectures with dedicated ``k_proj``/``v_proj`` submodules
        (Llama-style), or a last-dim slice into a fused ``c_attn`` output
        (GPT-2-style, where keys/values live in the second/third of three
        concatenated chunks). In-place writes through either form propagate
        to the model's tensor.
        """
        attn = self._attn()
        if hasattr(attn, proj):
            return getattr(attn, proj), None
        if hasattr(attn, "c_attn"):
            idx = {"k_proj": 1, "v_proj": 2}[proj]
            return attn.c_attn, idx
        raise AttributeError(
            f"Attention module at layer {self.layer} exposes neither a "
            f"{proj} submodule nor a fused c_attn. Use `print(model)` to "
            "find the projection layout."
        )

    def direction(
        self,
        positive: Sequence[str],
        negative: Sequence[str],
        proj: str = "v_proj",
        cache: bool = False,
    ) -> torch.Tensor:
        """Compute a steering direction from contrast prompts.

        Difference of mean projection outputs (averaged over every prompt
        and position in each set) between ``positive`` and ``negative``,
        normalized to unit norm. Both prompt sets run through one trace so
        local execution pays one interleaved pass and remote execution pays
        one job.

        Args:
            positive: Prompts exhibiting the target behavior (e.g. explicit
                step-by-step reasoning traces).
            negative: Prompts lacking it (e.g. terse direct answers).
            proj: ``"v_proj"`` or ``"k_proj"`` — which cache to steer.
            cache: Persist the edits as a model default (``model.edit``)
                instead of computing them once. The direction is still
                returned; enabling this re-applies it on every future
                generation without calling :meth:`apply`.

        Returns:
            Unit-norm ``[hidden]`` direction tensor on the model's device.
        """
        if proj not in ("k_proj", "v_proj"):
            raise ValueError(f"proj must be 'k_proj' or 'v_proj', got {proj!r}")
        module, idx = self._projection(proj)

        pos_acts: List[torch.Tensor] = []
        neg_acts: List[torch.Tensor] = []

        def collect(prompts: Sequence[str], sink: List[torch.Tensor]):
            for p in prompts:
                with tracer.invoke(p):
                    sink.append(_proj_view(_proj_output(module), idx).save())

        trace_ctx = self.model.edit() if cache else self.model.trace()
        with trace_ctx as tracer:
            collect(positive, pos_acts)
            collect(negative, neg_acts)

        direction = _mean_output(pos_acts) - _mean_output(neg_acts)
        return direction / direction.norm()

    def apply(
        self,
        tracer,
        direction: torch.Tensor,
        coef: float = 1.0,
        proj: str = "v_proj",
    ) -> None:
        """Add ``direction * coef`` to the projection output during prefill.

        Call inside a ``model.generate(...)`` tracing context. The addition
        is scoped to generation step 0 (``tracer.iter[0]``), so it lands in
        the KV cache once and every decode step attends over the steered
        values.

        Args:
            tracer: The tracer yielded by ``model.generate(...)``.
            direction: ``[hidden]`` direction, e.g. from :meth:`direction`.
            coef: Steering strength. Sweep it together with ``layer``; large
                values degrade fluency.
            proj: ``"v_proj"`` or ``"k_proj"`` — must match the projection
                the direction was computed against.
        """
        if proj not in ("k_proj", "v_proj"):
            raise ValueError(f"proj must be 'k_proj' or 'v_proj', got {proj!r}")
        module, idx = self._projection(proj)
        with torch.no_grad():
            out = _proj_view(_proj_output(module), idx)
            delta = (direction / direction.norm()).to(out.device) * coef
            for _ in tracer.iter[0]:
                out[:, -1, :] += delta
