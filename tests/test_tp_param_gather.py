"""Tests for tensor-parallel PARAMETER gathering.

vLLM shards parameters across TP ranks (e.g. ``lm_head``/``embed_tokens`` are vocab-sharded by
``VocabParallelEmbedding``). Without gathering, intervention code that reads a parameter inside a
trace sees only the local shard, so a vocab-indexed read like ``lm_head.weight[token_id]`` lands on
the WRONG global row on a rank that does not own that token (it indexes into that rank's shard). This
is the parameter analogue of the activation gather in ``VLLMBatcher`` — see
``Envoy.__getattr__``/``Batcher.gather_param``.

Run with:  pytest tests/test_tp_param_gather.py --tp 2 -v
"""
import pytest
import torch

try:
    from nnsight.modeling.vllm import VLLM
except Exception as e:
    pytest.skip(f"Skipping VLLM tests: \n{e}", allow_module_level=True)

import nnsight

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The Eiffel Tower is in the city of"


@pytest.fixture(scope="module")
def tp(request):
    tp = request.config.getoption("--tp")
    if tp > torch.cuda.device_count() or tp < 1:
        pytest.exit("--tp can't be higher than the number of available GPUs.")
    return tp


@pytest.fixture(scope="module")
def model(tp: int):
    return VLLM(
        MODEL,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.3,
        dispatch=True,
        dtype=torch.float16,
    )


class TestTPParamGather:
    @torch.no_grad()
    def test_lm_head_weight_is_full_vocab(self, tp, model):
        """Reading ``lm_head.weight`` inside a trace yields the full ``[vocab, hidden]`` tensor on
        every rank. At tp=1 it always did; under TP>1 it must be gathered, not the per-rank shard."""
        vocab = model.model.config.vocab_size
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(PROMPT):
                rows = nnsight.save(model.lm_head.weight.shape[0])
        assert int(rows) == vocab, (
            f"lm_head.weight has {int(rows)} rows under tp={tp}, expected full vocab {vocab}; "
            f"a sharded ~{vocab // max(tp, 1)} means the TP parameter gather did not fire."
        )

    @torch.no_grad()
    def test_upper_shard_token_row_accessible(self, tp, model):
        """A token id in the UPPER vocab shard (>= vocab/tp) must be indexable and return a real
        unembed row. On the raw per-rank shard that index is out of range / a different token, so this
        only passes once the parameter is gathered to its full logical shape."""
        if tp < 2:
            pytest.skip("Upper-shard indexing is only meaningful with TP>1")
        vocab = model.model.config.vocab_size
        tid = vocab - 5  # firmly inside the last rank's shard
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(PROMPT):
                row = nnsight.save(model.lm_head.weight[tid].float())
        assert torch.isfinite(row).all()
        assert row.norm() > 0

    @torch.no_grad()
    def test_row_and_column_parallel_weights_gathered(self, tp, model):
        """EVERY TP-sharded weight type gathers to its full [output_size, input_size], not just the
        vocab-parallel lm_head. RowParallelLinear (o_proj) shards the INPUT dim (1); ColumnParallelLinear
        (qkv_proj) shards the OUTPUT dim (0). Regression for keying the gather dim off the parameter's
        output_dim/input_dim attrs (vLLM sets BOTH on every linear weight), which gathered row-parallel
        weights on the wrong dim — e.g. an o_proj shard [896,448] became [1792,448] instead of [896,896]."""
        if tp < 2:
            pytest.skip("Sharding only occurs with TP>1")
        L = model.model.config.num_hidden_layers // 2
        o = model.model.layers[L].self_attn.o_proj._module        # RowParallelLinear
        qkv = model.model.layers[L].self_attn.qkv_proj._module    # ColumnParallelLinear (QKV)
        exp_o = (o.output_size, o.input_size)        # full logical shape (LinearBase stores both)
        exp_qkv = (qkv.output_size, qkv.input_size)
        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(PROMPT):
                ow = model.model.layers[L].self_attn.o_proj.weight
                o_shape = nnsight.save([ow.shape[0], ow.shape[1]])
                qw = model.model.layers[L].self_attn.qkv_proj.weight
                qkv_shape = nnsight.save([qw.shape[0], qw.shape[1]])
        assert tuple(int(x) for x in o_shape) == exp_o, (tuple(int(x) for x in o_shape), exp_o)
        assert tuple(int(x) for x in qkv_shape) == exp_qkv, (tuple(int(x) for x in qkv_shape), exp_qkv)
