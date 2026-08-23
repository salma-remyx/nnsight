"""Test-time latent optimization: gradient descent on a mid-layer latent state.

Optimization-based latent reasoning adapts a model to a single input at
*test time* by optimizing instance-specific continuous states while the
model weights stay frozen. This module provides the pieces of that loop
as trace-body calls:

- :meth:`LatentOptimizer.score` adds a trainable ``delta`` to one
  transformer block's residual stream over the prompt span
  (``block.output = hs + delta``) and returns the reward-weighted
  continuation log-loss,
- :meth:`LatentOptimizer.step` backprops that loss and applies a
  plain-Python Adam update to the latent alone,
- :meth:`LatentOptimizer.apply` re-injects the optimized latent during
  ``model.generate(...)`` to steer decoding,
- :meth:`LatentOptimizer.attribute` credits each continuation token by
  the norm of its gradient with respect to the latent.

Teacher-forced continuation log-probabilities give every continuation
token a differentiable path back to the latent through the remaining
blocks — causal attention does the credit assignment.

nnsight trace contexts capture their ``with`` block from the caller's
frame, so the ``with model.session()`` / ``with model.trace(...)`` /
``with model.generate(...)`` statements below must be written in user
code; the optimizer's methods are called *inside* them::

    from nnsight import LanguageModel
    from nnsight.modeling.latent_optimization import LatentOptimizer

    model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
    opt = LatentOptimizer(model, "The Eiffel Tower is in the city of",
                          " Paris, the capital of France.", layer=6)

    with model.session():
        for _ in range(opt.steps):
            with model.trace(opt.text):
                opt.step(opt.score())

    result = opt.result()
    print(result.losses)        # descent with model weights frozen

    with model.generate(opt.prompt, max_new_tokens=8) as tracer:
        opt.apply()
        tokens = tracer.result.save()

Adapted from GradCuit (credit-assigned gradient flow for test-time latent
reasoning, arXiv:2608.02585). Adaptations: the latent is an additive delta
on the prompt span of an existing block rather than freshly inserted
sequence positions, and the reward is a callable over continuation token
log-probabilities (uniform by default) rather than a trained verifier.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

# Dotted envoy paths to a transformer's block list, most common first.
_BLOCK_PATHS = ("model.layers", "transformer.h", "decoder.layers", "layers")


def resolve_blocks(model):
    """Return the ``Envoy`` for a transformer's list of blocks.

    Tries the common HuggingFace layouts so callers do not have to know
    whether a checkpoint keeps its blocks at ``model.layers`` or
    ``transformer.h``.

    Args:
        model: Root envoy (e.g. ``LanguageModel``).

    Returns:
        Envoy wrapping a ``torch.nn.ModuleList`` of transformer blocks.

    Raises:
        ValueError: If no known path resolves to a non-empty block list.
    """
    for path in _BLOCK_PATHS:
        envoy = model
        for part in path.split("."):
            try:
                envoy = getattr(envoy, part)
            except AttributeError:
                envoy = None
                break
        if envoy is not None and len(envoy) > 0:
            return envoy
    raise ValueError(
        "Could not find a transformer block list on the model. Tried dotted "
        f"paths: {', '.join(_BLOCK_PATHS)}. Pass blocks= explicitly."
    )


@dataclass
class LatentResult:
    """Outcome of one test-time latent optimization run.

    Attributes:
        delta: The optimized latent, ``[batch, num_positions, hidden]``.
        layer: Index of the optimized block within the block list.
        num_positions: Number of leading positions ``delta`` spans.
        losses: Mean reward-weighted continuation NLL per step.
        grad_norms: L2 norm of the latent gradient per step.
        prompt: Prompt the latent was optimized for.
        continuation: Teacher-forced continuation used as the objective.
    """

    delta: Optional[torch.Tensor] = None
    layer: int = -1
    num_positions: int = 0
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    prompt: str = ""
    continuation: str = ""


class LatentOptimizer:
    """Optimizes a latent addition to one block's residual stream at test time.

    Holds the latent, its Adam state, and the per-step traces across the
    optimization loop. The trace contexts themselves belong to the caller
    (see the module docstring); each method below is designed to be called
    inside one.

    Args:
        model: ``LanguageModel`` (any model with ``.tokenizer`` and an
            ``lm_head``).
        prompt: Prompt text; the latent spans its token positions.
        continuation: Target text appended to ``prompt``; its token
            log-probabilities form the objective.
        layer: Block index to optimize at. Defaults to the middle block,
            the most effective optimization space per the paper's layer
            analysis.
        blocks: Block-list envoy, overriding :func:`resolve_blocks`.
        steps: Optimization steps (paper uses <= 10).
        lr: Learning rate (paper uses 1e-3).
        optimizer: ``"adam"`` or ``"sgd"``.
        reward: Callable mapping the continuation's per-token
            log-probabilities ``[batch, n_cont]`` to non-negative weights.
            Uniform (all ones) when ``None``.
    """

    def __init__(
        self,
        model,
        prompt: str,
        continuation: str,
        *,
        layer: Optional[int] = None,
        blocks: Optional[Sequence] = None,
        steps: int = 10,
        lr: float = 1e-3,
        optimizer: str = "adam",
        reward: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        if blocks is None:
            blocks = resolve_blocks(model)
        if layer is None:
            layer = len(blocks) // 2

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "LatentOptimizer needs a model with a .tokenizer (e.g. LanguageModel)."
            )

        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + continuation)
        n_prompt = len(prompt_ids)
        if n_prompt >= len(full_ids):
            raise ValueError("continuation must add at least one token to prompt.")
        if optimizer not in ("adam", "sgd"):
            raise ValueError(f"unknown optimizer {optimizer!r} (adam | sgd)")

        self.model = model
        self.blocks = blocks
        self.layer = layer
        self.prompt = prompt
        self.continuation = continuation
        self.steps = steps
        self.lr = lr
        self.optimizer = optimizer
        self.reward = reward
        self.text = prompt + continuation
        self.full_ids = full_ids
        self.n_prompt = n_prompt

        self._delta: Optional[torch.Tensor] = None
        self._adam_m: Optional[torch.Tensor] = None
        self._adam_v: Optional[torch.Tensor] = None
        self._adam_t = 0
        self.losses: List[float] = []
        self.grad_norms: List[float] = []

    @property
    def block(self):
        """Envoy of the block the latent is optimized at."""
        return self.blocks[self.layer]

    def score(self) -> torch.Tensor:
        """Inject the latent and return the continuation log-loss.

        Adds ``delta`` over the prompt span at ``self.block`` and computes
        the reward-weighted mean NLL of the continuation tokens. Call from
        inside ``with model.trace(self.text):`` — the block and
        ``lm_head`` are accessed in forward order.

        Returns:
            Scalar loss tensor, differentiable with respect to the latent.
        """
        hs = self.block.output
        if self._delta is None:
            self._delta = torch.zeros(
                hs.shape[0], self.n_prompt, hs.shape[-1],
                device=hs.device, dtype=hs.dtype,
            )
        delta = self._delta.detach().requires_grad_(True)

        # The latent spans prompt positions only; the generated tail keeps
        # its own (untouched) residual stream.
        padded = torch.zeros_like(hs)
        padded[:, : self.n_prompt] = padded[:, : self.n_prompt] + delta
        self.block.output = hs + padded

        logits = self.model.lm_head.output
        logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = torch.tensor([self.full_ids[1:]], device=logits.device)
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        cont_lp = token_lp[:, self.n_prompt - 1 :]
        if self.reward is not None:
            weights = self.reward(cont_lp)
        else:
            weights = torch.ones_like(cont_lp)
        self._scored_delta = delta
        return -(cont_lp * weights).sum() / weights.sum()

    def step(self, loss: torch.Tensor) -> None:
        """Backprop ``loss`` and update the latent, leaving weights frozen.

        Call inside the same trace as the :meth:`score` that produced
        ``loss``. The latent is a leaf tensor, so its gradient is read
        straight off autograd rather than through a
        ``with loss.backward():`` context — the backwards tracer can only
        capture ``with`` blocks written in the caller's frame.

        Args:
            loss: Scalar tensor from :meth:`score`.
        """
        loss.backward()
        delta = self._scored_delta
        grad = delta.grad

        self.losses.append(float(loss.detach()))
        self.grad_norms.append(float(grad.detach().norm()))

        if self.optimizer == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            if self._adam_m is None:
                self._adam_m = torch.zeros_like(grad)
                self._adam_v = torch.zeros_like(grad)
            self._adam_t += 1
            self._adam_m = beta1 * self._adam_m + (1 - beta1) * grad
            self._adam_v = beta2 * self._adam_v + (1 - beta2) * grad * grad
            m_hat = self._adam_m / (1 - beta1**self._adam_t)
            v_hat = self._adam_v / (1 - beta2**self._adam_t)
            self._delta = (delta - self.lr * m_hat / (v_hat.sqrt() + eps)).detach()
        else:
            self._delta = (delta - self.lr * grad).detach()

    def result(self) -> LatentResult:
        """Snapshot the optimized latent and its per-step traces."""
        return LatentResult(
            delta=self._delta,
            layer=self.layer,
            num_positions=self.n_prompt,
            losses=list(self.losses),
            grad_norms=list(self.grad_norms),
            prompt=self.prompt,
            continuation=self.continuation,
        )

    def apply(self, coefficient: float = 1.0) -> None:
        """Add the optimized latent at its block during generation.

        Call inside ``with model.generate(...)``; the in-place slice add
        applies the latent on every decode step. A latent that has not
        been optimized yet is initialized to zeros, so ``apply()`` before
        any ``step()`` reproduces the model's unmodified generation.

        Args:
            coefficient: Scale on the latent, for sweeping strength.
        """
        hs = self.block.output
        if self._delta is None:
            self._delta = torch.zeros(
                hs.shape[0], self.n_prompt, hs.shape[-1],
                device=hs.device, dtype=hs.dtype,
            )
        hs[:, : self.n_prompt] = hs[:, : self.n_prompt] + coefficient * self._delta

    def attribute(self) -> Tuple[List[str], torch.Tensor]:
        """Credit each continuation token by its gradient on the latent.

        Call inside ``with model.trace(self.text):``. Runs one backward per
        continuation position on a retained graph; a token's score is the
        L2 norm of the gradient of its log-probability with respect to the
        latent.

        Returns:
            ``(tokens, scores)`` — continuation token strings and their
            non-negative attribution scores, aligned index-for-index.
            Tuple assignment does not propagate out of a trace body, so
            callers should capture the return into a pre-existing
            container (``out[:] = opt.attribute()``).
        """
        if self._delta is None:
            self._delta = torch.zeros(
                1, self.n_prompt, self.model.config.n_embd,
            )
        hs = self.block.output
        self._delta = self._delta.to(device=hs.device, dtype=hs.dtype)
        delta = self._delta.detach().requires_grad_(True)

        padded = torch.zeros_like(hs)
        padded[:, : self.n_prompt] = padded[:, : self.n_prompt] + delta
        self.block.output = hs + padded

        logits = self.model.lm_head.output
        logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = torch.tensor([self.full_ids[1:]], device=logits.device)
        token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # One backward per continuation token from the same retained graph.
        # Leaf .grad accumulates across backwards, so clear it each pass.
        scores: List[torch.Tensor] = []
        for i in range(self.n_prompt - 1, len(self.full_ids) - 1):
            token_lp[0, i].backward(retain_graph=True)
            scores.append(delta.grad.detach().norm())
            delta.grad = None

        cont_ids = self.full_ids[self.n_prompt :]
        tokens = self.model.tokenizer.convert_ids_to_tokens(cont_ids)
        return tokens, torch.stack(scores)


def decode_continuation(model, tokens: torch.Tensor, n_prompt: int) -> str:
    """Decode the generated tail of a full token tensor.

    Args:
        model: Model whose tokenizer should decode.
        tokens: Token tensor whose leading ``n_prompt`` columns are the
            prompt.
        n_prompt: Number of prompt columns to skip.

    Returns:
        The decoded continuation string.
    """
    return model.tokenizer.decode(tokens[0][n_prompt:], skip_special_tokens=True)


__all__ = [
    "LatentOptimizer",
    "LatentResult",
    "decode_continuation",
    "resolve_blocks",
]
