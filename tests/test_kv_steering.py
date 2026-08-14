"""Tests for KVSteering — one-shot k/v projection steering during prefill.

Verifies the capability end-to-end through the public LanguageModel API:
the steering direction computed from contrasting prompt sets has the right
shape/norm, and the prefill-scoped addition actually changes the model's
generation output relative to an unsteered run.
"""

import torch

import nnsight
from nnsight.modeling.kv_steering import KVSteering

POSITIVE = [
    "Let me think step by step. First, 2 plus 2 is 4.",
    "To solve this, I will break it down step by step.",
]
NEGATIVE = [
    "The answer is 4.",
    "The answer is obvious.",
]


def test_direction_shape_and_norm(gpt2: nnsight.LanguageModel):
    steering = KVSteering(gpt2, layer=6)

    direction = steering.direction(POSITIVE, NEGATIVE)

    assert isinstance(direction, torch.Tensor)
    assert direction.shape == (gpt2.config.n_embd,)
    assert torch.isclose(direction.norm(), torch.tensor(1.0), atol=1e-5)


def test_direction_projection_choice(gpt2: nnsight.LanguageModel):
    steering = KVSteering(gpt2, layer=3)

    k_dir = steering.direction(POSITIVE, NEGATIVE, proj="k_proj")
    v_dir = steering.direction(POSITIVE, NEGATIVE, proj="v_proj")

    assert k_dir.shape == v_dir.shape == (gpt2.config.n_embd,)
    # The two caches carry different content, so their contrast
    # directions should not coincide.
    assert not torch.allclose(k_dir, v_dir, atol=1e-4)


def test_apply_changes_generation(gpt2: nnsight.LanguageModel):
    steering = KVSteering(gpt2, layer=6)
    direction = steering.direction(POSITIVE, NEGATIVE)
    prompt = "Q: What is 3 + 5? A:"

    with gpt2.generate(prompt, max_new_tokens=5) as tracer:
        baseline = tracer.result.save()

    with gpt2.generate(prompt, max_new_tokens=5) as tracer:
        steering.apply(tracer, direction, coef=300.0)
        steered = tracer.result.save()

    assert baseline.shape == steered.shape
    assert not torch.equal(baseline, steered)


def test_apply_zero_coef_is_identity(gpt2: nnsight.LanguageModel):
    steering = KVSteering(gpt2, layer=6)
    direction = steering.direction(POSITIVE, NEGATIVE)
    prompt = "Q: What is 3 + 5? A:"

    with gpt2.generate(prompt, max_new_tokens=5) as tracer:
        baseline = tracer.result.save()

    with gpt2.generate(prompt, max_new_tokens=5) as tracer:
        steering.apply(tracer, direction, coef=0.0)
        unsteered = tracer.result.save()

    assert torch.equal(baseline, unsteered)


def test_invalid_proj_rejected(gpt2: nnsight.LanguageModel):
    steering = KVSteering(gpt2, layer=6)

    try:
        steering.direction(POSITIVE, NEGATIVE, proj="q_proj")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid proj")
