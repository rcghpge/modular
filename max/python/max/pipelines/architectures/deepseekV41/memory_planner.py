# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Memory planner for DeepSeek-V4.1-Flash.

Estimates device weight bytes from checkpoint headers and one forward step's
transient memory, so the device memory that remains can go to the paged KV
cache. KV storage is sized afterward from what remains, so the activation
estimate excludes it to avoid counting it twice.
"""

from __future__ import annotations

import functools
import json
import logging
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.support.human_readable_formatter import to_human_readable_bytes

from . import kv_leaves
from .layers.indexer import INDEXER_SCORE_TILE_BYTES, indexer_score_row_tile
from .model_config import (
    DeepseekV41Config,
    forward_row_bound,
    head_bf16_enabled,
    nvfp4_sf_preinterleave_enabled,
    woa_mxfp8_enabled,
)
from .weight_adapters import (
    DSPARK_CHECKPOINT_PREFIX,
    FLOAT32_WEIGHT_SUFFIXES,
    FUSED_QUERY_PROJECTIONS,
    HEAD_WEIGHT,
    MXFP8_BLOCK_SIZE,
    is_host_dequantized,
    is_skipped_checkpoint_tensor,
)

logger = logging.getLogger("max.pipelines")

# Indexer scores are float32.
_SCORE_BYTES_PER_ENTRY = 4


@dataclass(frozen=True)
class KvLeafPlan:
    """Page geometry for one KV-cache leaf, reduced to plain ints."""

    slots_per_page: int
    bytes_per_slot: int
    """Bytes per physical cache entry: ``kv_dim * head_dim * dtype_bytes``."""


@dataclass(frozen=True)
class KvMemoryPlanConfig:
    """Static inputs to :func:`plan_kv_memory`."""

    leaves: Mapping[str, KvLeafPlan]
    max_seq_len: int
    score_ratio: int
    """Smallest (widest-row) compress ratio among layers that own KV storage.

    Layers run sequentially and reuse the score scratch, so it is sized for
    the widest row rather than summed.
    """


def plan_kv_memory(
    config: KvMemoryPlanConfig,
    slots_by_leaf: Mapping[str, int],
    score_budget_bytes: int,
    live_rows: int,
) -> tuple[dict[str, int], int, int]:
    """Plans KV-cache leaf and indexer score-scratch bytes.

    Each leaf's page-padded storage is summed independently. Score rows are
    chunked against ``score_budget_bytes``; ``score_bytes`` is one reusable
    chunk buffer, since chunks run sequentially.

    Args:
        config: Per-leaf page geometry plus the score-width inputs.
        slots_by_leaf: Physical (already ratio-compressed) cache entries per
            leaf, keyed as ``config.leaves``.
        score_budget_bytes: Byte ceiling for the score scratch buffer.
        live_rows: Query rows ``T`` the score buffer must size for.

    Returns:
        ``(arena_bytes_by_leaf, score_bytes, total_bytes)``.

    Raises:
        ValueError: If ``live_rows > 0`` and one score row exceeds
            ``score_budget_bytes``.
    """
    arena_bytes_by_leaf = {
        key: _leaf_arena_bytes(config.leaves[key], slots)
        for key, slots in slots_by_leaf.items()
    }

    score_bytes = 0
    if live_rows > 0:
        row_bytes = _score_row_bytes(config.max_seq_len, config.score_ratio)
        if row_bytes > score_budget_bytes:
            raise ValueError(
                f"one indexer score row needs {row_bytes} B "
                f"(max_seq_len={config.max_seq_len}, "
                f"ratio={config.score_ratio}) but score_budget_bytes is "
                f"only {score_budget_bytes} B; rejecting rather than "
                "truncating the row width or allocating over budget"
            )
        rows_per_chunk = score_budget_bytes // row_bytes
        score_bytes = min(live_rows, rows_per_chunk) * row_bytes

    total_bytes = sum(arena_bytes_by_leaf.values()) + score_bytes
    return arena_bytes_by_leaf, score_bytes, total_bytes


# The forward graph's planned arena, fitted to measured device memory at
# S = 40000: ~2 MiB of activations per query row, five float32 and two bool
# ``[T, S]`` score tensors, and a row-independent part. It barely depends on
# the tensor-parallel degree (MoE intermediates are ``T * top_k`` rows on
# every rank).
_ARENA_ROW_BYTES = 2 << 20
_ARENA_SCORE_F32_SLOTS = 5
_ARENA_SCORE_MASK_SLOTS = 2
_ARENA_FIXED_BYTES = 5 << 28  # 1.25 GiB
# The device MemoryManager allocates whole 256 MiB pages.
_MM_PAGE_BYTES = 256 << 20


def _mm_pages(nbytes: int) -> int:
    """``nbytes`` rounded up to whole MemoryManager pages."""
    return ceil(nbytes / _MM_PAGE_BYTES) * _MM_PAGE_BYTES if nbytes > 0 else 0


def indexer_activation_bytes(
    *, live_rows: int, score_cols: int, local_heads: int, row_tile: int
) -> int:
    """Returns per-device bytes of the indexer row loop's scratch.

    Counts, outside the graph arena, the ``[T, S]`` float32 output plus one
    tile's ``[rows, S]`` result and ``[rows * local_heads, S]`` product,
    ``rows = min(T, row_tile)``, each in whole MemoryManager pages.

    Args:
        live_rows: Query rows ``T`` in one forward.
        score_cols: Score columns ``S`` (one per raw position).
        local_heads: Indexer heads on one device.
        row_tile: Rows per indexer score tile.
    """
    if live_rows <= 0 or score_cols <= 0:
        return 0
    rows = min(live_rows, row_tile)
    return (
        _mm_pages(live_rows * score_cols * _SCORE_BYTES_PER_ENTRY)
        + _mm_pages(rows * score_cols * _SCORE_BYTES_PER_ENTRY)
        + _mm_pages(rows * local_heads * score_cols * _SCORE_BYTES_PER_ENTRY)
    )


def graph_arena_bytes(*, live_rows: int, score_cols: int) -> int:
    """Returns per-device bytes of the forward graph's planned arena.

    A fixed part, a per-row part and five float32 and two bool ``[T, S]``
    score tensors, rounded up to whole pages.
    """
    if live_rows <= 0:
        return 0
    per_entry = (
        _ARENA_SCORE_F32_SLOTS * _SCORE_BYTES_PER_ENTRY
        + _ARENA_SCORE_MASK_SLOTS
    )
    return _mm_pages(
        _ARENA_FIXED_BYTES
        + live_rows * _ARENA_ROW_BYTES
        + live_rows * max(0, score_cols) * per_entry
    )


def forward_activation_bytes(
    *,
    live_rows: int,
    score_cols: int,
    local_heads: int,
    tile_bytes: int = INDEXER_SCORE_TILE_BYTES,
) -> int:
    """Returns one forward's per-device transient peak bytes.

    The graph arena plus the indexer scratch, with the row tile the model
    builds for ``tile_bytes``.
    """
    row_tile = indexer_score_row_tile(local_heads, score_cols, tile_bytes)
    return graph_arena_bytes(
        live_rows=live_rows, score_cols=score_cols
    ) + indexer_activation_bytes(
        live_rows=live_rows,
        score_cols=score_cols,
        local_heads=local_heads,
        row_tile=row_tile,
    )


def _leaf_arena_bytes(leaf: KvLeafPlan, slots: int) -> int:
    """Page-padded bytes for one leaf's ``slots`` physical cache entries."""
    if slots <= 0:
        return 0
    pages = ceil(slots / leaf.slots_per_page)
    return pages * leaf.slots_per_page * leaf.bytes_per_slot


def _score_row_bytes(max_seq_len: int, ratio: int) -> int:
    """One dense indexer score row's bytes: ``2*ceil(max_seq_len/ratio)*4``.

    The factor of two conservatively budgets a fresh and a retained half.
    """
    width = 2 * ceil(max_seq_len / ratio)
    return width * _SCORE_BYTES_PER_ENTRY


# ---------------------------------------------------------------------------
# Weight sizing from safetensors headers. Checkpoint file sizes are wrong for
# this model: Engram's hash tables live in pinned host memory, DSpark and
# vision tensors are not loaded by the trunk, and replicated tensors are held
# once per device.
# ---------------------------------------------------------------------------

# Engram's hash tables (weight and e8m0 scale) live in pinned host memory.
_HOST_RESIDENT_MARKER = ".engram.embed."

# Split across the devices (each holds ``1 / num_devices``): the routed
# experts (expert parallel), the head-sharded attention and indexer
# projections, and Engram's column-parallel ``wkv`` (weight and scale).
# Everything else is replicated, including ``wo_b`` (held whole on each
# device).
_SHARDED_MARKERS = (
    ".ffn.experts.",
    ".attn.wq_b.",
    ".attn.wo_a.",
    ".attn.attn_sink",
    ".attn.indexer.wq_b.",
    ".attn.indexer.weights_proj.",
    ".engram.wkv.",
)
# The fused query projection's fp8 weights are dequantized (2x) and their
# e8m0 scales folded in.
_FUSED_BF16_PROJECTIONS = tuple(
    f".attn.{projection}" for projection in FUSED_QUERY_PROJECTIONS
)
# The router gate's bf16 weight is held in float32.
_FLOAT32_GATE_SUFFIX = ".ffn.gate.weight"
# Routed-expert NVFP4 block scales. Unless pre-interleaved on the host, the
# compiler hoists the in-graph interleave into a second resident copy (2x).
_ROUTED_EXPERT_SCALE_MARKER = ".ffn.experts."
_ROUTED_EXPERT_SCALE_SUFFIX = ".weight_scale"


@dataclass(frozen=True)
class CheckpointTensor:
    """One safetensors header entry, reduced to what weight sizing needs."""

    name: str
    dtype: str
    nbytes: int


def read_safetensors_headers(paths: Iterable[Path]) -> list[CheckpointTensor]:
    """Reads every tensor's name, dtype and byte size from safetensors headers.

    Reads only each file's length prefix and JSON header.
    """
    tensors: list[CheckpointTensor] = []
    for path in paths:
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            begin, end = entry["data_offsets"]
            tensors.append(CheckpointTensor(name, entry["dtype"], end - begin))
    return tensors


@dataclass(frozen=True)
class DeviceWeightLayout:
    """The load-time switches that change how many bytes a tensor takes."""

    head_bf16: bool
    host_dequantized: Callable[[str], bool]
    """Whether a projection base (no ``.weight``) is dequantized to bf16."""
    expert_scales_preinterleaved: bool

    @classmethod
    def current(cls) -> DeviceWeightLayout:
        """The layout the weight adapter uses under the current env switches."""
        return cls(
            head_bf16=head_bf16_enabled(),
            host_dequantized=functools.partial(
                is_host_dequantized, woa_mxfp8=woa_mxfp8_enabled()
            ),
            # Assumes every expert scale shape tiles the interleave tile
            # exactly, so the env switch alone decides.
            expert_scales_preinterleaved=nvfp4_sf_preinterleave_enabled(),
        )


def _device_bytes_factor(
    tensor: CheckpointTensor, layout: DeviceWeightLayout
) -> float:
    """How many bytes one checkpoint byte becomes once loaded on a device.

    bf16 weights widened to float32 double. fp8 weights dequantized to bf16
    on the host double, and their e8m0 scales never reach a device. Every
    other MXFP8 weight's compact ``[N / block, K / block]`` e8m0 scale is
    row-expanded to ``[N, K / block]``.
    """
    name, dtype = tensor.name, tensor.dtype
    if name == HEAD_WEIGHT:
        return 1.0 if layout.head_bf16 else 2.0
    if dtype == "BF16" and name.endswith(
        (*FLOAT32_WEIGHT_SUFFIXES, _FLOAT32_GATE_SUFFIX)
    ):
        return 2.0
    if dtype == "F8_E4M3" and name.endswith(".weight"):
        base = name[: -len(".weight")]
        if _ROUTED_EXPERT_SCALE_MARKER not in name and (
            layout.host_dequantized(base)
            or base.endswith(_FUSED_BF16_PROJECTIONS)
        ):
            return 2.0
        return 1.0
    if dtype == "F8_E8M0" and name.endswith(".scale"):
        base = name[: -len(".scale")]
        if layout.host_dequantized(base) or base.endswith(
            _FUSED_BF16_PROJECTIONS
        ):
            return 0.0
        return float(MXFP8_BLOCK_SIZE)
    if (
        dtype == "F8_E4M3"
        and _ROUTED_EXPERT_SCALE_MARKER in name
        and name.endswith(_ROUTED_EXPERT_SCALE_SUFFIX)
        and not layout.expert_scales_preinterleaved
    ):
        return 2.0
    return 1.0


def device_weight_bytes(
    tensors: Iterable[CheckpointTensor],
    num_devices: int,
    layout: DeviceWeightLayout | None = None,
) -> tuple[int, int]:
    """Returns per-device weight bytes, plus the bytes kept in host memory.

    Args:
        tensors: The checkpoint's tensors (:func:`read_safetensors_headers`).
        num_devices: Devices the trunk is split across (TP = EP degree).
        layout: The load-time switches; defaults to
            :meth:`DeviceWeightLayout.current`.

    Returns:
        ``(per_device_bytes, host_bytes)``: what each device holds after
        load, and the pinned host-resident Engram table bytes.
    """
    num_devices = max(1, num_devices)
    if layout is None:
        layout = DeviceWeightLayout.current()
    per_device = 0.0
    host = 0
    for tensor in tensors:
        name = tensor.name
        # DSpark stages and vision tensors are not loaded with the trunk.
        if name.startswith(
            DSPARK_CHECKPOINT_PREFIX
        ) or is_skipped_checkpoint_tensor(name):
            continue
        if _HOST_RESIDENT_MARKER in name:
            host += tensor.nbytes
            continue
        nbytes = tensor.nbytes * _device_bytes_factor(tensor, layout)
        if any(marker in name for marker in _SHARDED_MARKERS):
            nbytes /= num_devices
        per_device += nbytes
    return ceil(per_device), host


def check_planned_max_length(
    *, planned: int | None, requested: int, forward_rows: int
) -> None:
    """Refuses a memory plan whose KV pool cannot hold one prefill chunk.

    Without this check the plan would succeed with ``max_length`` clamped to
    a few tokens. The error is worded like other out-of-memory planning
    errors so callers that retry with smaller configs recognize it.

    Args:
        planned: The plan's ``planned_max_length``.
        requested: The ``max_length`` the config asked for.
        forward_rows: Query rows one forward carries (the prefill chunk).

    Raises:
        RuntimeError: If ``planned < min(requested, forward_rows)``.
    """
    if not all(isinstance(v, int) for v in (planned, requested, forward_rows)):
        return  # No plan (virtual devices) or a partial test double.
    assert planned is not None
    if planned >= min(requested, forward_rows):
        return
    raise RuntimeError(
        f"DeepSeek-V4.1-Flash: the KV pool left after the weights and the "
        f"forward's {forward_rows}-row activation reserve holds only "
        f"{planned} tokens (max_length {requested} requested), so the "
        "model, activations, and signal buffers don't leave room for KV "
        "cache. Lower max_length or the prefill chunk "
        "(max_batch_input_tokens), or raise device_memory_utilization."
    )


class DeepseekV41MemoryPlanner(PagedMemoryPlanner):
    """Reserves the forward's transient memory out of the KV budget.

    The graph arena and indexer scratch (:func:`forward_activation_bytes`)
    can reach tens of GiB per device, so they must be subtracted before the
    KV leaves are sized.
    """

    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """Estimates device weight bytes, summed over devices.

        Free memory is compared summed over devices, so this returns the
        per-device size times the device count. Falls back to the
        checkpoint's file sizes when the headers cannot be read (e.g. a
        remote repo).
        """
        model_config = pipeline_config.model
        num_devices = max(1, len(model_config.device_specs))
        try:
            paths = model_config.resolved_weight_paths()
            tensors = read_safetensors_headers(
                p for p in paths if str(p).endswith(".safetensors")
            )
        except (OSError, ValueError) as e:
            logger.warning(
                "DeepSeek-V4.1-Flash weight sizing fell back to file sizes "
                "(could not read safetensors headers: %s)",
                e,
            )
            return model_config.weights_size()
        if not tensors:
            return model_config.weights_size()
        per_device, host = device_weight_bytes(tensors, num_devices)
        logger.info(
            "Estimated DeepSeek-V4.1-Flash weights: %s per device x %d "
            "devices (%s of Engram tables kept in host memory)",
            to_human_readable_bytes(per_device),
            num_devices,
            to_human_readable_bytes(host),
        )
        return per_device * num_devices

    def estimate_activation_memory(
        self, pipeline_config: Any, huggingface_config: Any
    ) -> int:
        """Estimates one forward's per-device transient peak bytes.

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration (the
                top-level config, with text fields under ``text_config``).

        Returns:
            :func:`graph_arena_bytes` plus :func:`indexer_activation_bytes`
            per device, summed over devices (free memory is summed the same
            way). Excludes KV leaf storage (see module docstring).
        """
        text_config = DeepseekV41Config.get_text_config(huggingface_config)
        specs = kv_leaves.kv_leaf_specs(text_config)

        if not any(spec.kind == "comp" for spec in specs):
            # No compressed layer, so no indexer scoring.
            return 0

        runtime = pipeline_config.runtime
        max_length = pipeline_config.model.max_length
        # Decode also scores a dense [T, S] row, so this is not zeroed for
        # decode-only roles.
        live_rows = forward_row_bound(
            runtime.max_num_input_tokens,
            runtime.max_batch_input_tokens,
            max_length,
        )
        num_devices = max(1, len(pipeline_config.model.device_specs))
        local_heads = ceil(text_config.index_n_heads / num_devices)

        row_tile = indexer_score_row_tile(
            local_heads, max_length, INDEXER_SCORE_TILE_BYTES
        )
        arena = graph_arena_bytes(live_rows=live_rows, score_cols=max_length)
        indexer = indexer_activation_bytes(
            live_rows=live_rows,
            score_cols=max_length,
            local_heads=local_heads,
            row_tile=row_tile,
        )
        per_device = arena + indexer

        logger.info(
            "Estimated DeepSeek-V4.1-Flash forward activation memory: %s per "
            "device x %d devices (graph arena %s + indexer scoring %s; %d "
            "live rows x %d columns, %d local heads, %d-row tiles)",
            to_human_readable_bytes(per_device),
            num_devices,
            to_human_readable_bytes(arena),
            to_human_readable_bytes(indexer),
            live_rows,
            max_length,
            local_heads,
            row_tile,
        )
        return per_device * num_devices
