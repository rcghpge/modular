# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Benchmarks for NVFP4-quantized AttentionWithRope layer.

Uses --benchmark-mode to switch between smoke (CI) and full (offline):

    --benchmark-mode=smoke   1 shape, 1 iteration
    --benchmark-mode=full    all shapes, 50 iterations each
"""

from __future__ import annotations

import math
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
# Llama3-8B-scale config
# --------------------------------------------------------------------------- #

HIDDEN_SIZE = 4096
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
MAX_SEQ_LEN = 131072
ROPE_THETA = 500000.0

Q_DIM = N_HEADS * HEAD_DIM
KV_DIM = N_KV_HEADS * HEAD_DIM

_TOTAL_PREFILL_TOKENS = 8192
_PAGE_SIZE = 128

SMOKE_SHAPES = [
    (1, 8192, 8192),
    (1, 1, 8192),
]

TOTAL_NUM_PAGES = 4096


def _build_full_shapes() -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    for ctx in [8192, 16384, 32768]:
        for bs in [1, 8, 16, 32]:
            pages = bs * ((ctx + 1 + _PAGE_SIZE - 1) // _PAGE_SIZE)
            if pages <= TOTAL_NUM_PAGES:
                shapes.append((bs, 1, ctx))
            seq = _TOTAL_PREFILL_TOKENS // bs
            pages = bs * ((ctx + seq + _PAGE_SIZE - 1) // _PAGE_SIZE)
            if pages <= TOTAL_NUM_PAGES:
                shapes.append((bs, seq, ctx))
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
# NVFP4 QuantConfig
# --------------------------------------------------------------------------- #


def _make_nvfp4_quant_config() -> QuantConfig:
    """Build QuantConfig for NVFP4 block-scaled quantization."""
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, 16),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, 16 // 2),
    )
    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        format=QuantFormat.NVFP4,
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

    quant_config = _make_nvfp4_quant_config()
    rope = RotaryEmbedding(HIDDEN_SIZE, N_HEADS, ROPE_THETA, MAX_SEQ_LEN)

    # stacked_qkv=False: FP4 models ship with separate Q/K/V weights.
    # The concat → rope_split_store path still fuses rope+split+store.
    layer = AttentionWithRope(
        rope=rope,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        hidden_size=HIDDEN_SIZE,
        kv_params=kv_params,
        stacked_qkv=False,
        has_bias=False,
        dtype=DType.uint8,
        devices=[DeviceRef.GPU()],
        quant_config=quant_config,
    )

    # Generate synthetic NVFP4 weights for separate projections.
    def _make_fp4_weights(
        name: str, out_dim: int, in_dim: int
    ) -> dict[str, Any]:
        packed_k = in_dim // 2
        ws_cols = math.ceil(packed_k / 8)
        weight = torch.randint(0, 255, (out_dim, packed_k), dtype=torch.uint8)
        ws = (torch.rand(out_dim, ws_cols, dtype=torch.float32) * 100 + 50).to(
            torch.float8_e4m3fn
        )
        return {
            f"{name}.weight": weight,
            f"{name}.weight_scale": _wrap_fp8(ws, f"{name}.weight_scale"),
            f"{name}.weight_scale_2": torch.rand((), dtype=torch.float32)
            * 1e-4,
            f"{name}.input_scale": torch.rand((), dtype=torch.float32) * 1e-3,
        }

    state: dict[str, Any] = {}
    state.update(_make_fp4_weights("q_proj", Q_DIM, HIDDEN_SIZE))
    state.update(_make_fp4_weights("k_proj", KV_DIM, HIDDEN_SIZE))
    state.update(_make_fp4_weights("v_proj", KV_DIM, HIDDEN_SIZE))
    state.update(_make_fp4_weights("o_proj", HIDDEN_SIZE, Q_DIM))

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
        "AttentionWithRopeFP4",
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

BENCHMARK_TABLE_TITLE = "AttentionWithRope NVFP4 [llama3-8b]"


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


def test_benchmark_attention_fp4(
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
