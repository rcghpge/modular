# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Benchmarks for GGUFQAttentionWithRope layer.

Uses GGUF Q4_K quantized weights. Measures the unfused GGUF QKV matmul +
fused rope + flash attention pipeline.

NOTE: GGUF quantized weights require specific block-packed byte layouts that
cannot be easily synthesized. This benchmark requires real GGUF model weights.
Run with a real model path via --gguf-model-path or skip.

Uses --benchmark-mode to switch between smoke (CI) and full (offline):

    --benchmark-mode=smoke   1 shape, 1 iteration
    --benchmark-mode=full    all shapes, 50 iterations each
"""

from __future__ import annotations

from pathlib import Path
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
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import load_weights
from max.interfaces import RequestID, TextGenerationContext, TokenBuffer
from max.kv_cache import PagedKVCacheManager
from max.nn import RotaryEmbedding
from max.nn.attention.attention_with_rope import GGUFQAttentionWithRope
from max.nn.kv_cache import KVCacheParams, unflatten_ragged_attention_inputs
from max.pipelines.core import TextContext

# --------------------------------------------------------------------------- #
# Llama3-8B-scale config
# --------------------------------------------------------------------------- #

HIDDEN_SIZE = 4096
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
MAX_SEQ_LEN = 65536
ROPE_THETA = 500000.0

Q_DIM = N_HEADS * HEAD_DIM  # 4096
KV_DIM = N_KV_HEADS * HEAD_DIM  # 1024

SMOKE_SHAPES = [
    (1, 8192, 8192),
    (1, 1, 8192),
]
FULL_SHAPES = [
    (1, 1, 8192),
    (8, 1, 8192),
    (1, 1, 32768),
    (8, 1, 32768),
    (1, 8192, 8192),
    (1, 8192, 16384),
    (1, 8192, 32768),
    (1, 8192, 49152),
]

_DEFAULT_ITERATIONS = {"smoke": 1, "full": 50}
_DEFAULT_WARMUP = {"smoke": 1, "full": 5}

TOTAL_NUM_PAGES = 4096


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
# Graph compilation (session-scoped)
# --------------------------------------------------------------------------- #


class CompiledAttentionBundle(NamedTuple):
    compiled_model: Any
    kv_manager: PagedKVCacheManager
    kv_params: KVCacheParams
    device: Accelerator
    hidden_size: int


def _build_compiled_attention(
    gguf_model_path: str,
) -> CompiledAttentionBundle:
    device = Accelerator()
    session = InferenceSession(devices=[device])

    kv_params = KVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_layers=1,
        devices=[DeviceRef.GPU()],
    )

    rope = RotaryEmbedding(HIDDEN_SIZE, N_HEADS, ROPE_THETA, MAX_SEQ_LEN)

    encoding = QuantizationEncoding.Q4_K

    layer = GGUFQAttentionWithRope(
        rope=rope,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        hidden_size=HIDDEN_SIZE,
        kv_params=kv_params,
        dtype=DType.uint8,
        quantization_encoding=encoding,
        devices=[DeviceRef.GPU()],
    )

    # Load weights from a real GGUF file (layer 0 attention weights).
    weights = load_weights([Path(gguf_model_path)])
    # Filter to just the attention layer weights we need.
    attn_prefix = "model.layers.0.self_attn."
    attn_weights: dict[str, Any] = {}
    for key, val in weights.items():
        if key.startswith(attn_prefix):
            short_key = key[len(attn_prefix) :]
            attn_weights[short_key] = val
    layer.load_state_dict(attn_weights)

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
        "GGUFQAttentionWithRope",
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
def compiled_attention(
    request: pytest.FixtureRequest,
) -> CompiledAttentionBundle:
    gguf_path = request.config.getoption("--gguf-model-path", default=None)
    if not gguf_path:
        pytest.skip(
            "GGUF benchmark requires --gguf-model-path pointing to a "
            "Llama3-8B GGUF file (e.g. Q4_K_M quantization)"
        )
    return _build_compiled_attention(gguf_path)


def pytest_addoption(parser: pytest.Parser) -> None:
    # Only add if not already added by conftest.py
    try:
        parser.addoption(
            "--gguf-model-path",
            default=None,
            help="Path to a GGUF model file for GGUF attention benchmarks",
        )
    except ValueError:
        pass


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

BENCHMARK_TABLE_TITLE = "GGUFQAttentionWithRope [Q4_K, llama3-8b]"


def test_benchmark_attention_gguf(
    compiled_attention: CompiledAttentionBundle,
    benchmark_tables: dict[str, list[tuple[str, dict[str, float]]]],
    request: pytest.FixtureRequest,
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

    bundle = compiled_attention

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
        table = benchmark_tables.setdefault(BENCHMARK_TABLE_TITLE, [])
        table.append((f"b{batch_size}_q{seq_len}_ctx{context_len}", stats))
