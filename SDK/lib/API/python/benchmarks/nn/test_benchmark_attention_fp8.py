# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Benchmarks for FP8 dynamic AttentionWithRope layer (chansey workload).

Models the chansey multi-turn workload with Llama-3.1-405B FP8 dynamic
at TP=8 per-shard dimensions: 16 Q heads, 1 KV head, hidden=16384.

Shapes mirror chansey traffic:
- First-turn prefill: ~10K tokens
- Subsequent-turn prefills: ~125 tokens at growing context (10K-15K)
- Decode: seq_len=1 at various context depths

Uses --benchmark-mode to switch between smoke (CI) and full (offline):

    --benchmark-mode=smoke   1 shape, 1 iteration
    --benchmark-mode=full    all shapes, 50 iterations each
"""

from __future__ import annotations

from typing import Any, NamedTuple, cast

import numpy as np
import pytest
import torch
from benchmark_utils import (
    compute_stats,
    measure_gpu_latency,
    measure_gpu_latency_cuda_graph,
)
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.type import Shape
from max.graph.weights import WeightData
from max.interfaces import RequestID, TextGenerationContext, TokenBuffer
from max.kv_cache import PagedKVCacheManager
from max.nn import AttentionWithRope, RotaryEmbedding
from max.nn.kv_cache import KVCacheParams, unflatten_ragged_attention_inputs
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.core import TextContext

# --------------------------------------------------------------------------- #
# Llama3.1-405B per-GPU shard config (TP=8, FP8 dynamic)
#
# Chansey runs Llama-3.1-405B-Instruct-FP8-dynamic with TP=8.
# Per shard: 128/8=16 Q heads, 8/8=1 KV head, hidden=16384.
# The attention layer input is the full hidden_size (pre-sharding),
# but Q/K/V projections are sharded: q_dim=16*128=2048, kv_dim=1*128=128.
# --------------------------------------------------------------------------- #

HIDDEN_SIZE = 16384
N_HEADS = 16  # 128 total / TP=8
N_KV_HEADS = 1  # 8 total / TP=8
HEAD_DIM = 128
MAX_SEQ_LEN = 131072
ROPE_THETA = 500000.0

Q_DIM = N_HEADS * HEAD_DIM  # 2048
KV_DIM = N_KV_HEADS * HEAD_DIM  # 128

_PAGE_SIZE = 16  # matches chansey config

# Chansey-inspired shapes:
#   First turn: bs=1, seq=10000, ctx=0 (large prefill)
#   Subsequent turns: bs=N, seq=125, ctx=10000 (small prefill, deep context)
#   Decode: bs=N, seq=1, ctx=10000-15000
SMOKE_SHAPES = [
    (1, 1, 10000),
    (1, 10000, 0),
]

TOTAL_NUM_PAGES = 32768  # page_size=16 → ~500K tokens capacity


def _build_full_shapes() -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    # First-turn prefill
    shapes.append((1, 10000, 0))
    # Subsequent-turn prefills at growing context
    for ctx in [10000, 12000, 15000]:
        for bs in [1, 8, 16, 32]:
            pages = bs * ((ctx + 125 + _PAGE_SIZE - 1) // _PAGE_SIZE)
            if pages <= TOTAL_NUM_PAGES:
                shapes.append((bs, 125, ctx))
    # Decode at various context depths
    for ctx in [10000, 12000, 15000]:
        for bs in [1, 8, 16, 32]:
            pages = bs * ((ctx + 1 + _PAGE_SIZE - 1) // _PAGE_SIZE)
            if pages <= TOTAL_NUM_PAGES:
                shapes.append((bs, 1, ctx))
    return shapes


FULL_SHAPES = _build_full_shapes()

_DEFAULT_ITERATIONS = {"smoke": 1, "full": 50}
_DEFAULT_WARMUP = {"smoke": 1, "full": 5}


# --------------------------------------------------------------------------- #
# Dynamic parametrization based on --benchmark-mode
# --------------------------------------------------------------------------- #


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "batch_size" in metafunc.fixturenames:
        mode = metafunc.config.getoption("--benchmark-mode")
        shapes = SMOKE_SHAPES if mode == "smoke" else FULL_SHAPES
        metafunc.parametrize(
            "batch_size,seq_len,context_len",
            shapes,
            ids=[f"b{b}_q{s}_ctx{c}" for b, s, c in shapes],
        )


# --------------------------------------------------------------------------- #
# FP8 dynamic QuantConfig (fbgemm_fp8)
# --------------------------------------------------------------------------- #


def _make_fp8_quant_config() -> QuantConfig:
    """Build QuantConfig for FP8 dynamic (fbgemm_fp8) quantization.

    Mirrors RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic:
    column-wise input quantization, per-row weight scales.
    """
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.COLWISE,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float8_e4m3fn,
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.ROWWISE,
        dtype=DType.float32,
    )
    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        format=QuantFormat.FBGEMM_FP8,
        mlp_quantized_layers={0},
        attn_quantized_layers={0},
        embedding_output_dtype=DType.bfloat16,
    )


def _wrap_fp8(tensor: torch.Tensor, name: str) -> WeightData:
    """Wrap an FP8 tensor as WeightData for dlpack compatibility."""
    return WeightData(
        Buffer.from_dlpack(tensor.cpu().view(torch.uint8)).view(
            DType.float8_e4m3fn
        ),
        name,
        DType.float8_e4m3fn,
        Shape(tensor.shape),
    )


# --------------------------------------------------------------------------- #
# Graph compilation (session-scoped)
# --------------------------------------------------------------------------- #


class CompiledAttentionBundle(NamedTuple):
    compiled_model: Any
    kv_manager: PagedKVCacheManager
    kv_params: KVCacheParams
    device: Accelerator
    hidden_size: int


def _build_compiled_attention() -> CompiledAttentionBundle:
    device = Accelerator()
    session = InferenceSession(devices=[device])

    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        devices=[DeviceRef.GPU()],
    )

    quant_config = _make_fp8_quant_config()
    rope = RotaryEmbedding(
        HIDDEN_SIZE, N_HEADS, ROPE_THETA, MAX_SEQ_LEN, head_dim=HEAD_DIM
    )

    # Separate Q/K/V projections — graph-level concat via wqkv property.
    layer = AttentionWithRope(
        rope=rope,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        hidden_size=HIDDEN_SIZE,
        kv_params=kv_params,
        stacked_qkv=False,
        has_bias=False,
        dtype=DType.float8_e4m3fn,
        devices=[DeviceRef.GPU()],
        quant_config=quant_config,
    )

    # Generate synthetic FP8 weights for separate Q/K/V projections.
    # FP8 dynamic: no input_scale in checkpoint (computed at runtime).
    def _make_fp8_weights(
        name: str, out_dim: int, in_dim: int
    ) -> dict[str, Any]:
        weight = torch.randn(out_dim, in_dim, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        return {
            f"{name}.weight": _wrap_fp8(weight, f"{name}.weight"),
            f"{name}.weight_scale": torch.rand(
                (out_dim, 1), dtype=torch.float32
            )
            * 1e-3,
        }

    state: dict[str, Any] = {}
    state.update(_make_fp8_weights("q_proj", Q_DIM, HIDDEN_SIZE))
    state.update(_make_fp8_weights("k_proj", KV_DIM, HIDDEN_SIZE))
    state.update(_make_fp8_weights("v_proj", KV_DIM, HIDDEN_SIZE))
    state.update(_make_fp8_weights("o_proj", HIDDEN_SIZE, Q_DIM))

    layer.load_state_dict(state)

    device_ref = DeviceRef.GPU()
    input_type = TensorType(
        dtype=DType.bfloat16,
        shape=["total_seq_len", HIDDEN_SIZE],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    flattened_kv_types = kv_params.get_symbolic_inputs().flatten()

    with Graph(
        "AttentionWithRopeFP8",
        input_types=(input_type, input_row_offsets_type, *flattened_kv_types),
    ) as graph:
        inputs, input_row_offsets, *kv_cache = graph.inputs
        kv_collection = unflatten_ragged_attention_inputs(
            kv_cache, n_devices=1
        )[0]
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        graph.output(
            layer(
                layer_idx,
                inputs.tensor,
                kv_collection,
                freqs_cis=rope.freqs_cis,
                input_row_offsets=input_row_offsets.tensor,
            )
        )

    compiled = session.load(graph, weights_registry=layer.state_dict())

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=TOTAL_NUM_PAGES,
        session=session,
        max_batch_size=128,
    )

    return CompiledAttentionBundle(
        compiled_model=compiled,
        kv_manager=kv_manager,
        kv_params=kv_params,
        device=device,
        hidden_size=HIDDEN_SIZE,
    )


@pytest.fixture(scope="session")
def compiled_attention() -> CompiledAttentionBundle:
    return _build_compiled_attention()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _prepare_inputs(
    bundle: CompiledAttentionBundle,
    batch_size: int,
    q_seq_len: int,
    context_len: int,
) -> tuple[list[TextGenerationContext], list[Any]]:
    device = bundle.device
    total_len = context_len + q_seq_len

    batch: list[TextGenerationContext] = []
    for _ in range(batch_size):
        ctx = TextContext(
            request_id=RequestID(),
            max_length=max(total_len, MAX_SEQ_LEN),
            tokens=TokenBuffer(np.empty(total_len, dtype=np.int64)),
        )
        bundle.kv_manager.claim(ctx.request_id, replica_idx=0)
        bundle.kv_manager.alloc(ctx, replica_idx=0)
        if context_len > 0:
            ctx.tokens.skip_processing(context_len)
        batch.append(ctx)

    kv_runtime = bundle.kv_manager.runtime_inputs(
        cast(list[list[TextGenerationContext]], [batch])
    ).inputs[0]
    assert kv_runtime.attention_dispatch_metadata is not None

    total_tokens = batch_size * q_seq_len
    input_tensor = Buffer.from_dlpack(
        torch.randn(total_tokens, bundle.hidden_size, dtype=torch.bfloat16)
    ).to(device)
    row_offsets = Buffer.from_numpy(
        np.array(
            [i * q_seq_len for i in range(batch_size + 1)], dtype=np.uint32
        )
    ).to(device)

    execute_args = [
        input_tensor,
        row_offsets,
        kv_runtime.blocks.to(device),
        kv_runtime.cache_lengths.to(device),
        kv_runtime.lookup_table.to(device),
        kv_runtime.max_lengths,
        kv_runtime.attention_dispatch_metadata,
    ]

    return batch, execute_args


def _release_batch(
    bundle: CompiledAttentionBundle,
    batch: list[TextGenerationContext],
) -> None:
    for ctx in batch:
        bundle.kv_manager.release(ctx.request_id, replica_idx=0)


# --------------------------------------------------------------------------- #
# Test entry point
# --------------------------------------------------------------------------- #

BENCHMARK_TABLE_TITLE = (
    "AttentionWithRope FP8 dynamic [llama3.1-405b TP=8 shard, chansey]"
)


def _run_benchmark(
    bundle: CompiledAttentionBundle,
    request: pytest.FixtureRequest,
    benchmark_tables: dict[str, list[tuple[str, dict[str, float]]]],
    table_title: str,
    batch_size: int,
    seq_len: int,
    context_len: int,
) -> None:
    mode = request.config.getoption("--benchmark-mode")
    num_iterations = (
        request.config.getoption("--benchmark-iterations")
        or _DEFAULT_ITERATIONS[mode]
    )
    num_warmup = (
        request.config.getoption("--benchmark-warmup") or _DEFAULT_WARMUP[mode]
    )

    batch, execute_args = _prepare_inputs(
        bundle, batch_size, seq_len, context_len
    )
    bundle.device.synchronize()

    is_decode = seq_len == 1

    if is_decode:
        graph_key = batch_size * 100000 + context_len
        latencies = measure_gpu_latency_cuda_graph(
            bundle.compiled_model,
            execute_args,
            bundle.device,
            graph_key=graph_key,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )
    else:
        latencies = measure_gpu_latency(
            lambda: bundle.compiled_model.execute(*execute_args),
            bundle.device,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )

    _release_batch(bundle, batch)

    stats = compute_stats(latencies)
    assert stats["mean_ms"] > 0.0

    if mode == "full":
        table = benchmark_tables.setdefault(table_title, [])
        table.append((f"b{batch_size}_q{seq_len}_ctx{context_len}", stats))


def test_benchmark_attention_fp8(
    compiled_attention: CompiledAttentionBundle,
    benchmark_tables: dict[str, list[tuple[str, dict[str, float]]]],
    request: pytest.FixtureRequest,
    batch_size: int,
    seq_len: int,
    context_len: int,
) -> None:
    _run_benchmark(
        compiled_attention,
        request,
        benchmark_tables,
        BENCHMARK_TABLE_TITLE,
        batch_size,
        seq_len,
        context_len,
    )
