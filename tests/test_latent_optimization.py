"""
Tests for test-time latent optimization (nnsight.modeling.latent_optimization).

Covers the optimize -> attribute -> apply loop end-to-end on GPT-2: gradient
descent on a mid-layer latent with model weights frozen, per-continuation-token
credit assignment, and re-injecting the optimized latent during generation.
"""

import pytest
import torch

import nnsight
from nnsight.modeling.latent_optimization import (
    LatentOptimizer,
    decode_continuation,
    resolve_blocks,
)


PROMPT = "The Eiffel Tower is in the city of"
CONTINUATION = " Paris, the capital of France."


def run(opt: LatentOptimizer, steps: int) -> None:
    """Drive ``steps`` optimization passes inside a session."""
    with opt.model.session():
        for _ in range(steps):
            with opt.model.trace(opt.text):
                opt.step(opt.score())


class TestResolveBlocks:
    def test_resolves_gpt2_blocks(self, gpt2: nnsight.LanguageModel):
        assert len(resolve_blocks(gpt2)) == gpt2.config.n_layer

    def test_unknown_layout_raises(self):
        from nnsight import NNsight

        net = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(ValueError, match="block list"):
            resolve_blocks(NNsight(net))


class TestLatentOptimizer:
    def test_default_layer_is_middle(self, gpt2: nnsight.LanguageModel):
        opt = LatentOptimizer(gpt2, PROMPT, CONTINUATION)
        assert opt.layer == gpt2.config.n_layer // 2

    def test_delta_shape_and_grad_flow(self, gpt2: nnsight.LanguageModel):
        n_prompt = len(gpt2.tokenizer.encode(PROMPT))

        opt = LatentOptimizer(gpt2, PROMPT, CONTINUATION, layer=6, steps=3)
        run(opt, 3)
        result = opt.result()

        assert result.delta is not None
        assert result.delta.shape == (1, n_prompt, gpt2.config.n_embd)
        assert result.num_positions == n_prompt
        assert len(result.losses) == 3
        assert len(result.grad_norms) == 3
        # Gradients reached the latent on every step.
        assert all(g > 0 for g in result.grad_norms)

    def test_loss_descends(self, gpt2: nnsight.LanguageModel):
        opt = LatentOptimizer(gpt2, PROMPT, CONTINUATION, layer=6, lr=1e-2, steps=8)
        run(opt, 8)

        assert opt.losses[-1] < opt.losses[0]

    def test_model_weights_frozen(self, gpt2: nnsight.LanguageModel):
        before = gpt2.transformer.h[6].mlp.c_fc.weight.detach().clone()

        opt = LatentOptimizer(gpt2, PROMPT, CONTINUATION, layer=6, steps=3)
        run(opt, 3)

        assert torch.equal(before, gpt2.transformer.h[6].mlp.c_fc.weight.detach())

    def test_sgd_optimizer(self, gpt2: nnsight.LanguageModel):
        opt = LatentOptimizer(
            gpt2, PROMPT, CONTINUATION, layer=6, steps=3, optimizer="sgd", lr=1e-1
        )
        run(opt, 3)

        assert opt.result().delta.abs().sum() > 0

    def test_reward_weighting_changes_objective(self, gpt2: nnsight.LanguageModel):
        def downweight_first(logprobs):
            weights = torch.ones_like(logprobs)
            weights[:, 0] = 0.0
            return weights

        plain = LatentOptimizer(gpt2, PROMPT, CONTINUATION, layer=6)
        weighted = LatentOptimizer(
            gpt2, PROMPT, CONTINUATION, layer=6, reward=downweight_first
        )
        run(plain, 1)
        run(weighted, 1)

        assert plain.losses[0] != weighted.losses[0]

    def test_unknown_optimizer_raises(self, gpt2: nnsight.LanguageModel):
        with pytest.raises(ValueError, match="unknown optimizer"):
            LatentOptimizer(gpt2, PROMPT, CONTINUATION, optimizer="adagrad")

    def test_empty_continuation_raises(self, gpt2: nnsight.LanguageModel):
        with pytest.raises(ValueError, match="continuation"):
            LatentOptimizer(gpt2, PROMPT, "", steps=1)


class TestApplyLatent:
    def test_latent_steer_generation(self, gpt2: nnsight.LanguageModel):
        # A zero latent must leave generation identical to baseline.
        opt = LatentOptimizer(gpt2, PROMPT, CONTINUATION, layer=6, steps=0)

        with gpt2.generate(PROMPT, max_new_tokens=4, do_sample=False) as tracer:
            baseline = tracer.result.save()

        with gpt2.generate(PROMPT, max_new_tokens=4, do_sample=False) as tracer:
            opt.apply()  # initializes the (zero) latent
            steered = tracer.result.save()

        assert torch.equal(baseline, steered)

    def test_strong_latent_changes_generation(self, gpt2: nnsight.LanguageModel):
        opt = LatentOptimizer(gpt2, PROMPT, CONTINUATION, layer=6, steps=0)

        with gpt2.generate(PROMPT, max_new_tokens=4, do_sample=False) as tracer:
            baseline = tracer.result.save()

        # A large coefficient on a random direction must flip the output.
        # (No Frobenius normalization: a unit-norm delta over 7680 elements
        # is numerically too small to move GPT-2 even at coefficient 20.)
        gen = torch.Generator().manual_seed(0)
        opt._delta = torch.randn(1, opt.n_prompt, gpt2.config.n_embd, generator=gen)

        with gpt2.generate(PROMPT, max_new_tokens=4, do_sample=False) as tracer:
            opt.apply(coefficient=20.0)
            steered = tracer.result.save()

        assert not torch.equal(baseline, steered)

    def test_decode_continuation_skips_prompt(self, gpt2: nnsight.LanguageModel):
        n_prompt = len(gpt2.tokenizer.encode(PROMPT))

        with gpt2.generate(PROMPT, max_new_tokens=4, do_sample=False) as tracer:
            tokens = tracer.result.save()

        text = decode_continuation(gpt2, tokens, n_prompt)

        assert PROMPT not in text
        assert len(text) > 0


class TestLatentAttribution:
    def test_scores_align_with_continuation_tokens(self, gpt2: nnsight.LanguageModel):
        opt = LatentOptimizer(gpt2, PROMPT, CONTINUATION, layer=6, steps=1)
        run(opt, 1)

        # Tuple assignment does not propagate out of a trace body; capture
        # into a pre-existing container instead.
        out: list = []
        with gpt2.trace(opt.text):
            out[:] = opt.attribute()

        tokens, scores = out
        n_cont = len(gpt2.tokenizer.encode(opt.text)) - opt.n_prompt
        assert len(tokens) == n_cont
        assert scores.shape == (n_cont,)
        assert (scores >= 0).all()
        # A differentiable path exists from every continuation token to the
        # latent, so no score should be exactly zero.
        assert (scores > 0).all()

    def test_attribution_concentrates_on_answer_tokens(self, gpt2):
        # The latent was optimized toward " Paris, the capital of France.";
        # its influence should fall on the content words, not the comma.
        opt = LatentOptimizer(gpt2, PROMPT, CONTINUATION, layer=6, steps=1)
        run(opt, 1)

        out: list = []
        with gpt2.trace(opt.text):
            out[:] = opt.attribute()

        tokens, scores = out
        by_token = dict(zip(tokens, scores.tolist()))
        assert by_token["ĠParis"] > by_token[","]
