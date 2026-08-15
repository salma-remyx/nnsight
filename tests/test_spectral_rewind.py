"""
Integration tests for spectral head scoring and selective rewinding.

Exercises `nnsight.modeling.spectral_rewind` (head_importance /
rewind_heads) against a real `NNsight`-wrapped transformer-like model with
attention output projections, going through the public `model.edit()` /
`model.trace()` surfaces. Adapted from the head-importance and selective
rewinding method of "Diffract: Spectral View of LLM Domain Adaptation"
(arXiv:2608.10850).
"""

import pytest
import torch
import torch.nn as nn

from nnsight import NNsight
from nnsight.modeling import head_importance, low_importance_heads, rewind_heads
from nnsight.modeling.spectral_rewind import spectral_delta


N_HEADS = 2
HEAD_DIM = 4
HIDDEN = N_HEADS * HEAD_DIM
SEQ = 3


class Attention(nn.Module):
    """Minimal attention whose output projection matches the HF naming."""

    num_heads = N_HEADS

    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x):
        return self.o_proj(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()

    def forward(self, x):
        return self.self_attn(x) + x


class TinyTransformer(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def base_net():
    torch.manual_seed(0)
    return TinyTransformer()


@pytest.fixture
def cpt_net(base_net):
    """`base_net` after a mock continual pre-training pass.

    Head 0 receives a small update in both layers, head 1 a large one — the
    low-importance / high-importance contrast the rewind selects on.
    """
    cpt = TinyTransformer()
    cpt.load_state_dict(base_net.state_dict())
    updates = [(0.01, 0, HEAD_DIM), (0.5, HEAD_DIM, N_HEADS * HEAD_DIM)]
    with torch.no_grad():
        for scale, start, stop in updates:
            for layer in cpt.layers:
                layer.self_attn.o_proj.weight[start:stop] += scale * torch.randn(
                    stop - start, HIDDEN
                )
    return cpt


def _envoys(net):
    return NNsight(net)


def test_spectral_delta_zero_for_identical_weights():
    w = torch.randn(4, 8)
    assert float(spectral_delta(w, w)) == 0.0
    assert float(spectral_delta(w + 1.0, w)) > 0.0


def test_head_importance_ranks_updated_heads(base_net, cpt_net):
    base, cpt = _envoys(base_net), _envoys(cpt_net)
    scores = head_importance(cpt, base)

    paths = set(scores)
    assert paths == {
        "layers.0.self_attn.o_proj",
        "layers.1.self_attn.o_proj",
    }

    for per_head in scores.values():
        assert per_head.shape == (N_HEADS,)

    # Head 1 received the larger update in both layers.
    for path in paths:
        assert float(scores[path][1]) > float(scores[path][0])


def test_low_importance_heads_selects_half_by_default(base_net, cpt_net):
    scores = head_importance(_envoys(cpt_net), _envoys(base_net))
    selection = low_importance_heads(scores, fraction=0.5)

    # Head 0 is the untouched (lowest-importance) head everywhere.
    assert selection == {
        "layers.0.self_attn.o_proj": [0],
        "layers.1.self_attn.o_proj": [0],
    }


def test_rewind_heads_restores_low_importance_rows(base_net, cpt_net):
    base, cpt = _envoys(base_net), _envoys(cpt_net)
    scores = head_importance(cpt, base)
    rewound = rewind_heads(cpt, base, scores, fraction=0.5)

    x = torch.randn(1, SEQ, HIDDEN)

    with torch.no_grad():
        before = cpt_net(x)
        # Head-0 rows rewound to base, head-1 rows kept from the adaptation.
        for layer_idx in range(2):
            w = cpt_net.layers[layer_idx].self_attn.o_proj.weight
            w[:HEAD_DIM] = base_net.layers[layer_idx].self_attn.o_proj.weight[
                :HEAD_DIM
            ]
        manually_rewound = cpt_net(x)

    # The edit ran through model.edit(): the rewound model's traces now match
    # the manually-rewound module, not the original CPT baseline.
    with rewound.trace(x):
        out = rewound.output.save()
    assert torch.allclose(out, manually_rewound)
    assert not torch.allclose(before, manually_rewound)


def test_rewind_heads_accepts_explicit_selection(base_net, cpt_net):
    base, cpt = _envoys(base_net), _envoys(cpt_net)
    scores = head_importance(cpt, base)
    selection = low_importance_heads(scores, fraction=0.5)

    rewound = rewind_heads(cpt, base, scores, fraction=selection)

    x = torch.randn(1, SEQ, HIDDEN)
    with rewound.trace(x):
        out = rewound.output.save()

    with torch.no_grad():
        expected = cpt_net(x)

    assert torch.allclose(out, expected)


def test_head_importance_requires_shared_head_count(base_net, cpt_net):
    base, cpt = _envoys(base_net), _envoys(cpt_net)
    with pytest.raises(ValueError):
        head_importance(cpt, base, n_heads=3)  # 8 rows not divisible by 3
