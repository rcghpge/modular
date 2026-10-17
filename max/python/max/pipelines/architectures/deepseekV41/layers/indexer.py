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
"""Sparse-attention indexer for DeepSeek-V4.1-Flash.

The indexer cheaply scores every past KV position for each query, so that
attention can be restricted to the highest-scoring positions.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Sequence
from math import ceil
from typing import overload

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    ShardingStrategy,
    StaticDim,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn.comm import Signals
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm

from ...deepseekV4.layers.rope import apply_rope_tail

INDEXER_SCORE_ROW_TILE = 4096
"""Default query rows per :meth:`DeepseekV41Indexer.score` tile.

Caps the ``[rows * n_heads, S]`` float32 product. Used only when ``S`` is
symbolic; otherwise tiles are sized from :data:`INDEXER_SCORE_TILE_BYTES`."""

_SCORE_TILE_MIB_ENV = "MAX_DSV41_INDEXER_TILE_MIB"
_DEFAULT_SCORE_TILE_MIB = 512


def _score_tile_bytes_from_env() -> int:
    raw = os.environ.get(_SCORE_TILE_MIB_ENV)
    mib = _DEFAULT_SCORE_TILE_MIB if raw in (None, "") else int(raw)
    if mib <= 0:
        raise ValueError(f"{_SCORE_TILE_MIB_ENV} must be positive, got {raw}")
    return mib << 20


INDEXER_SCORE_TILE_BYTES = _score_tile_bytes_from_env()
"""Byte budget for one tile's ``[rows * local_heads, S]`` float32 product.

Defaults to ``_DEFAULT_SCORE_TILE_MIB`` MiB; override with
``MAX_DSV41_INDEXER_TILE_MIB``. Rows are scored independently, so the tile
size never changes a score."""


def indexer_score_row_tile(
    local_heads: int,
    score_cols: int,
    tile_bytes: int = INDEXER_SCORE_TILE_BYTES,
) -> int:
    """Returns how many query rows one score tile holds under ``tile_bytes``.

    The most rows whose ``[rows * local_heads, score_cols]`` float32 product
    fits, and at least 1.
    """
    per_row = max(1, local_heads) * max(1, score_cols) * 4
    return max(1, tile_bytes // per_row)


# Makes each tiled `score` call's per-tile row dims distinct graph symbols.
_score_call_ids = itertools.count()
# The same, for each tiled `reduce_scores` call.
_reduce_call_ids = itertools.count()

# Pessimistic SIMD width the all-reduce assumes when sizing its per-device
# scratch; must agree with the kernel for the size limit below to hold.
_ALLREDUCE_SCRATCH_VEC = 32


def allreduce_max_elements(num_devices: int, dtype: DType) -> int:
    """Returns the most elements one ``ops.allreduce.sum`` accepts.

    Each device stages ``ceil(ceil(n / 32) / num_devices) * 32 * dtype_size``
    bytes in its ``Signals._SCRATCH_BYTES`` signal-buffer scratch; a larger
    input fails at execution.
    """
    vecs_per_device = Signals._SCRATCH_BYTES // (
        _ALLREDUCE_SCRATCH_VEC * dtype.size_in_bytes
    )
    return vecs_per_device * num_devices * _ALLREDUCE_SCRATCH_VEC


def map_row_tiles(
    xs: Sequence[TensorValue],
    fn: Callable[[list[TensorValue]], list[TensorValue]],
    *,
    max_rows: int,
    row_tile: int,
) -> list[TensorValue]:
    """Applies cross-device, row-wise ``fn`` to ``xs`` one row tile at a time.

    Tile ``i`` covers rows ``[i * row_tile, (i + 1) * row_tile)`` clamped to
    ``T``; the last of ``ceil(max_rows / row_tile)`` tiles runs to ``T``.
    With ``max_rows <= row_tile`` this is ``fn(xs)``.

    Args:
        xs: One ``[T, S]`` tensor per device, sharing ``T``.
        fn: Maps one tile per device to one output tile per device.
        max_rows: The most rows a call is built for.
        row_tile: Rows per tile.

    Returns:
        Each device's tiles concatenated back to ``[T, S]``.
    """
    if max_rows <= row_tile:
        return fn(list(xs))
    t_len = ops.reshape(ops.shape_to_tensor([xs[0].shape[0]]), [])
    num_tiles = ceil(max_rows / row_tile)
    call_id = next(_reduce_call_ids)
    per_device: list[list[TensorValue]] = [[] for _ in xs]
    for i in range(num_tiles):
        start = ops.min(
            ops.constant(i * row_tile, DType.int64, DeviceRef.CPU()), t_len
        )
        stop = (
            t_len
            if i == num_tiles - 1
            else ops.min(
                ops.constant((i + 1) * row_tile, DType.int64, DeviceRef.CPU()),
                t_len,
            )
        )
        rows = (slice(start, stop), f"indexer_reduce_rows_{call_id}_tile_{i}")
        outs = fn([x[rows, :] for x in xs])
        for tiles, out in zip(per_device, outs, strict=True):
            tiles.append(out)
    return [
        ops.rebind(
            ops.concat(tiles, axis=0),
            x.shape,
            message="indexer reduce tiles must cover every query row",
        )
        for x, tiles in zip(xs, per_device, strict=True)
    ]


class DeepseekV41Indexer(Module):
    """Projects queries and keys, then scores KV positions for each query.

    .. code-block:: text

        score[t, s] = sum_h relu(q[t, h, :] . k[s, :]) * weight[t, h]
        weight[t, h] = weights_proj(x)[t, h] * head_dim ** -0.5

    RoPE is interleaved over each head's trailing ``rope_head_dim`` dims.
    Keys are projected from a compressed pre-RoPE KV latent, not from ``x``.
    ``wk``/``k_norm`` exist only when ``owns_k``, since a layer may reuse
    another layer's index keys. Scores are dense and unmasked; top-k
    selection is left to the caller.

    Under tensor parallelism ``wq_b``/``weights_proj`` are split by head and
    the partial head-sums are all-reduced (:meth:`reduce_scores`) before
    top-k, so every device ranks identical scores. ``wk``/``k_norm`` are
    replicated because keys have no head axis.
    """

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        main_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        rms_norm_eps: float,
        dtype: DType,
        device: DeviceRef,
        *,
        owns_k: bool,
        score_row_tile: int = INDEXER_SCORE_ROW_TILE,
        score_max_rows: int | None = None,
        single_request: bool = False,
        score_tile_bytes: int | None = None,
    ) -> None:
        """Initializes the indexer.

        Args:
            owns_k: Whether this layer produces its own index keys (creates
                ``wk``/``k_norm``).
            score_row_tile: Query rows per :meth:`score` tile when ``S`` is
                symbolic or ``score_tile_bytes`` is unset.
            score_max_rows: The most rows one :meth:`score` call carries;
                defaults to a static ``S``. If it fits one tile, no row loop
                is built.
            single_request: Builds only the single-request path of
                :meth:`score_by_request`, which then fails at run time on a
                batch of several requests. Use it only when every call
                carries one request.
            score_tile_bytes: If set, sizes the row tile from this byte
                budget for a static ``S`` (:func:`indexer_score_row_tile`).

        Raises:
            ValueError: If a tile size is not positive or ``rope_head_dim``
                exceeds ``index_head_dim``.
        """
        super().__init__()
        if score_row_tile <= 0:
            raise ValueError(
                f"score_row_tile must be positive, got {score_row_tile}"
            )
        if score_tile_bytes is not None and score_tile_bytes <= 0:
            raise ValueError(
                f"score_tile_bytes must be positive, got {score_tile_bytes}"
            )
        self.score_row_tile = score_row_tile
        self.score_tile_bytes = score_tile_bytes
        self.score_max_rows = score_max_rows
        self.single_request = single_request
        if rope_head_dim > index_head_dim:
            raise ValueError(
                f"rope_head_dim ({rope_head_dim}) cannot exceed "
                f"index_head_dim ({index_head_dim})"
            )
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        # Scaled by `head_dim` only; there is no `n_heads**-0.5` factor.
        self.softmax_scale = self.head_dim**-0.5
        self.owns_k = owns_k

        self.wq_b = Linear(
            q_lora_rank, self.n_heads * self.head_dim, dtype, device
        )
        # Uses the model `dtype` rather than a fixed bf16.
        self.weights_proj = Linear(hidden_size, self.n_heads, dtype, device)

        if owns_k:
            self.wk = Linear(main_head_dim, self.head_dim, dtype, device)
            self.k_norm = RMSNorm(self.head_dim, DType.float32, rms_norm_eps)

    def shard_heads(self, num_devices: int) -> None:
        """Row-splits ``wq_b`` and ``weights_proj`` by head.

        Raises:
            ValueError: If the heads do not divide ``num_devices``.
        """
        if self.n_heads % num_devices:
            raise ValueError(
                f"{self.n_heads} indexer heads do not split across "
                f"{num_devices} devices"
            )
        self.wq_b.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        self.weights_proj.sharding_strategy = ShardingStrategy.rowwise(
            num_devices
        )

    def keep_local_heads(self, num_devices: int) -> None:
        """Keeps only this device's share of the :meth:`shard_heads` split."""
        self.n_heads //= num_devices

    @staticmethod
    def reduce_scores(
        scores: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        *,
        max_rows: int | None = None,
    ) -> list[TensorValue]:
        """All-reduces per-device head-sums into complete ``[T, S]`` scores.

        Reduced in row tiles (:func:`map_row_tiles`) of the most rows one
        all-reduce's signal-buffer scratch fits
        (:func:`allreduce_max_elements`). The sum is elementwise, so tiling
        does not change any result.

        Args:
            scores: One ``[T, S]`` float32 partial score per device.
            signal_buffers: The all-reduce signal buffers.
            max_rows: The most rows one call is built for; defaults to a
                static ``T``. Without either, the reduce is not tiled.

        Returns:
            One identical ``[T, S]`` score per device.
        """
        scores = list(scores)
        t, s = scores[0].shape
        if max_rows is None and isinstance(t, StaticDim):
            max_rows = int(t)
        if max_rows is None or not isinstance(s, StaticDim):
            return ops.allreduce.sum(scores, signal_buffers)
        row_tile = max(
            1, allreduce_max_elements(len(scores), scores[0].dtype) // int(s)
        )
        return map_row_tiles(
            scores,
            lambda tiles: ops.allreduce.sum(tiles, signal_buffers),
            max_rows=max_rows,
            row_tile=row_tile,
        )

    def produce_q(self, qr: TensorValue, freqs_cis: TensorValue) -> TensorValue:
        """Returns ``[T, n_heads, head_dim]`` queries with RoPE applied.

        RoPE covers each head's trailing ``rope_head_dim`` dims.

        Args:
            qr: ``[1, T, q_lora_rank]`` normalized low-rank query.
            freqs_cis: ``[T, rope_head_dim // 2, 2]`` at the query positions.
        """
        t = qr.shape[-2]
        q = ops.reshape(self.wq_b(qr), [1, t, self.n_heads, self.head_dim])
        q = apply_rope_tail(q, freqs_cis, self.rope_head_dim)
        return ops.reshape(q, [t, self.n_heads, self.head_dim])

    def produce_weights(self, x: TensorValue) -> TensorValue:
        """Returns ``[T, n_heads]`` float32 per-head weights.

        Pre-multiplied by ``softmax_scale``.

        Args:
            x: The layer's raw hidden state; any leading batch-of-1 axis is
                dropped.
        """
        t = x.shape[-2]
        weights = (
            ops.cast(self.weights_proj(x), DType.float32) * self.softmax_scale
        )
        return ops.reshape(weights, [t, self.n_heads])

    def produce_k(
        self, latent: TensorValue, freqs_cis: TensorValue
    ) -> TensorValue:
        """Returns ``[S, head_dim]`` index keys: ``wk``, RMSNorm, then RoPE.

        Args:
            latent: ``[..., S, main_head_dim]`` compressed KV latent, before
                RoPE.
            freqs_cis: ``[S, rope_head_dim // 2, 2]`` at each key's position.

        Raises:
            ValueError: If ``owns_k`` is not set.
        """
        if not self.owns_k:
            raise ValueError(
                "produce_k requires owns_k=True; this indexer was "
                "constructed to reuse another layer's index keys instead "
                "of producing its own."
            )
        s = latent.shape[-2]
        k = ops.reshape(self.wk(latent), [1, s, 1, self.head_dim])
        k = self.k_norm(k)
        k = apply_rope_tail(k, freqs_cis, self.rope_head_dim)
        return ops.reshape(k, [s, self.head_dim])

    def score(
        self, q: TensorValue, weights: TensorValue, keys: TensorValue
    ) -> TensorValue:
        """Returns unmasked ``[T, S]`` float32 scores over the local heads.

        Rows are scored in tiles of one :func:`~max.graph.ops.while_loop`
        (:meth:`_score_row_loop`), so only one ``[rows * n_heads, S]``
        product is live at a time; the result is bit-identical to untiled.

        Args:
            q: ``[T, n_heads, head_dim]`` from :meth:`produce_q`.
            weights: ``[T, n_heads]`` from :meth:`produce_weights`.
            keys: ``[S, head_dim]`` from :meth:`produce_k` (possibly another
                layer's), shared by every query row.
        """
        t = q.shape[0]
        s = keys.shape[0]
        k32_t = ops.transpose(ops.cast(keys, DType.float32), 0, 1)
        max_rows = self._max_rows(s)
        tile = self._row_tile(s)
        if max_rows is not None and max_rows <= tile:
            return self._score_rows(q, weights, k32_t)

        # Cast before slicing (exact, and what `_score_rows` does anyway): a
        # cast of a sliced tile inside a region fails GPU compilation
        # ("'mo.cast' op requires the same shape for all operands").
        q32 = ops.cast(q, DType.float32)
        weights32 = ops.cast(weights, DType.float32)
        out = ops.buffer_create(BufferType(DType.float32, [t, s], q.device))
        self._score_row_loop(
            out,
            [q32, weights32],
            tile,
            ops.constant(0, DType.int64, DeviceRef.CPU()),
            ops.reshape(ops.shape_to_tensor([t]), []),
            lambda q_i, w_i: self._score_rows(q_i, w_i, k32_t),
        )
        return ops.buffer_load(out)

    def score_by_request(
        self,
        q: TensorValue,
        weights: TensorValue,
        keys: TensorValue,
        request_ids: TensorValue,
    ) -> TensorValue:
        """:meth:`score` for a ragged batch, each row against its own keys.

        ``B == 1`` runs :meth:`score`'s shared-key path on ``keys[0]``;
        ``B > 1`` runs the per-row path (:meth:`_score_rows_own_table`).
        Both loops are built and the shape-derived batch size gives every
        row to one of them, without a device read-back. With
        ``single_request`` only the shared path is built.

        Args:
            q: ``[T, n_heads, head_dim]`` queries.
            weights: ``[T, n_heads]`` head weights.
            keys: ``[B, S, head_dim]``, one position-indexed key table per
                request.
            request_ids: ``[T]`` int32, each row's request.

        Returns:
            ``[T, S]`` float32 scores.
        """
        t = q.shape[0]
        s = keys.shape[1]
        if self.single_request:
            keys = ops.rebind(
                keys,
                [1, s, keys.shape[2]],
                message="single_request indexer got several requests",
            )
            return self.score(q, weights, keys[0])
        k32_t = ops.transpose(ops.cast(keys[0], DType.float32), 0, 1)
        keys32_t = ops.transpose(ops.cast(keys, DType.float32), 1, 2)
        q32 = ops.cast(q, DType.float32)
        weights32 = ops.cast(weights, DType.float32)
        zero = ops.constant(0, DType.int64, DeviceRef.CPU())
        t_len = ops.reshape(ops.shape_to_tensor([t]), [])
        single = ops.reshape(ops.shape_to_tensor([keys.shape[0]]), []) == 1
        shared_rows = ops.where(single, t_len, zero)
        out = ops.buffer_create(BufferType(DType.float32, [t, s], q.device))
        tile = self._row_tile(s)
        self._score_row_loop(
            out,
            [q32, weights32],
            tile,
            zero,
            shared_rows,
            lambda q_i, w_i: self._score_rows(q_i, w_i, k32_t),
        )
        # A per-row tile also holds its gathered keys, so shrink it to the
        # shared path's `rows * n_heads * S` footprint.
        per_row_tile = max(
            1,
            tile * self.n_heads // (self.head_dim + self.n_heads),
        )
        self._score_row_loop(
            out,
            [q32, weights32, request_ids],
            per_row_tile,
            shared_rows,
            t_len,
            lambda q_i, w_i, ids_i: self._score_rows_own_table(
                q_i, w_i, keys32_t, ids_i
            ),
        )
        return ops.buffer_load(out)

    def _row_tile(self, s: Dim) -> int:
        """Returns the rows per score tile.

        Sized from ``score_tile_bytes`` when set and ``S`` is static, else
        ``score_row_tile``.
        """
        if self.score_tile_bytes is not None and isinstance(s, StaticDim):
            return indexer_score_row_tile(
                self.n_heads, int(s), self.score_tile_bytes
            )
        return self.score_row_tile

    def _max_rows(self, s: Dim) -> int | None:
        """The most rows a :meth:`score` call is built for."""
        if self.score_max_rows is None and isinstance(s, StaticDim):
            return int(s)
        return self.score_max_rows

    @staticmethod
    def _score_row_loop(
        out: BufferValue,
        inputs: Sequence[TensorValue],
        tile: int,
        row_start: TensorValue,
        row_end: TensorValue,
        rows_fn: Callable[..., TensorValue],
    ) -> None:
        """Stores ``rows_fn`` of rows ``[row_start, row_end)`` into ``out``.

        One :func:`~max.graph.ops.while_loop` body, ``tile`` rows per
        iteration, so the graph holds one copy of ``rows_fn``'s kernels
        regardless of the tile count. No iteration runs when ``row_start >=
        row_end``.

        Args:
            out: ``[T, S]`` float32 output buffer.
            inputs: Tensors sharing the leading row dim.
            tile: Rows per iteration.
            row_start: Host int64 scalar, first row.
            row_end: Host int64 scalar, end row (exclusive).
            rows_fn: Maps one tile of each input to ``[rows, S]`` scores.
        """
        rows_dim = f"indexer_score_rows_{next(_score_call_ids)}"
        tile_rows = ops.constant(tile, DType.int64, DeviceRef.CPU())

        def body(start: TensorValue) -> TensorValue:
            stop = ops.min(start + tile_rows, row_end)
            # Every input, and the stored slice, shares the tile's row dim.
            rows = (slice(start, stop), rows_dim)
            sliced = [
                x[(rows,) + (slice(None),) * (x.rank - 1)] for x in inputs
            ]
            ops.buffer_store_slice(out, rows_fn(*sliced), [rows, slice(None)])
            return stop

        ops.while_loop(row_start, lambda start: start < row_end, body)

    def _score_rows_own_table(
        self,
        q32: TensorValue,
        weights32: TensorValue,
        keys32_t: TensorValue,
        request_ids: TensorValue,
    ) -> TensorValue:
        """Scores rows like :meth:`_score_rows`, each against its own keys.

        ``keys32_t`` holds the ``[B, head_dim, S]`` float32 transposed key
        tables; ``request_ids`` picks each row's table.
        """
        k_t = ops.gather(keys32_t, request_ids, axis=0)  # [rows, head_dim, S]
        dots = ops.relu(ops.matmul(q32, k_t))  # [rows, n_heads, S]
        return ops.squeeze(
            ops.sum(dots * ops.unsqueeze(weights32, -1), axis=1), axis=1
        )

    def _score_rows(
        self, q: TensorValue, weights: TensorValue, k32_t: TensorValue
    ) -> TensorValue:
        """Applies :meth:`score`'s formula to ``q``'s rows.

        ``k32_t`` is the ``[head_dim, S]`` float32 transposed keys.
        """
        t = q.shape[0]
        s = k32_t.shape[1]
        q32 = ops.reshape(
            ops.cast(q, DType.float32), [t * self.n_heads, self.head_dim]
        )
        dots = ops.matmul(q32, k32_t)
        dots = ops.reshape(ops.relu(dots), [t, self.n_heads, s])
        return ops.squeeze(
            ops.sum(dots * ops.unsqueeze(weights, -1), axis=1), axis=1
        )

    def __call__(
        self,
        x: TensorValue,
        qr: TensorValue,
        latent: TensorValue,
        q_freqs_cis: TensorValue,
        k_freqs_cis: TensorValue,
    ) -> TensorValue:
        """Projects and scores on one device; requires ``owns_k``.

        Sharded or reused-key layers call :meth:`produce_q`,
        :meth:`produce_weights`, :meth:`score` and :meth:`reduce_scores`
        directly.
        """
        q = self.produce_q(qr, q_freqs_cis)
        weights = self.produce_weights(x)
        k = self.produce_k(latent, k_freqs_cis)
        return self.score(q, weights, k)


# ---------------------------------------------------------------------------
# Block-level candidate selection. Score columns are grouped into fixed-size
# blocks of KV positions and only each query's best-scoring blocks are kept,
# bounding the positions a later per-token top-k has to search.
# ---------------------------------------------------------------------------

# `ops.max`'s float32 axis-reduction identity is the most negative finite
# value, not `-inf`, so an all-`-inf` block reduces to this sentinel.
_FLOAT32_MAX_REDUCTION_NEG_INF_SENTINEL = float(np.finfo(np.float32).min)


def _reinstate_true_neg_inf(reduced_max: TensorValue) -> TensorValue:
    """Maps ``ops.max``'s float32 reduction sentinel back to ``-inf``."""
    return ops.where(
        reduced_max <= _FLOAT32_MAX_REDUCTION_NEG_INF_SENTINEL,
        float("-inf"),
        reduced_max,
    )


def candidate_pool(
    scores: TensorValueLike,
    row_end: TensorValueLike,
    *,
    block_size: int,
    topk_blocks: int,
    row_repeat: int = 1,
    row_start: TensorValueLike | None = None,
) -> TensorValue:
    """Selects the top ``topk_blocks`` indexer-score blocks for each row.

    Each row's blocks are numbered from its own ``start``: block ``b`` covers
    columns ``[start + b * block_size, start + (b + 1) * block_size)``,
    clipped by ``end``. A block scores the NaN-propagating max of its
    in-bound columns; the newest block of a nonempty ``[start, end)`` is
    forced to ``+inf`` so it is always selected. A selected ``-inf`` block
    stores as ``-1``; a selected NaN block keeps its id. Order among NaN or
    tied ``+inf`` blocks is unspecified. Uses only static-shape graph ops (no
    scatter, no host sync), so it is graph-capturable.

    Args:
        scores: ``[T, W]`` float32, column ``w`` is packed KV position ``w``.
        row_end: ``[ceil(T / row_repeat)]`` int32, causal-interval end
            (exclusive).
        block_size: Columns per block. Must be positive.
        topk_blocks: Output candidate width. Must be non-negative.
        row_repeat: Consecutive rows sharing one ``(start, end)`` pair.
        row_start: ``[ceil(T / row_repeat)]`` int32, causal-interval start
            (inclusive); ``None`` means ``0``.

    Returns:
        ``[T, topk_blocks]`` int32 local block ids, ``-1``-padded past the
        live count. An empty interval yields an all-``-1`` row.

    Raises:
        ValueError: If ``T`` or ``W`` is not static, ``row_end`` or
            ``row_start`` has the wrong group count, or a size argument is
            out of range.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive: {block_size=}")
    if topk_blocks < 0:
        raise ValueError(f"topk_blocks must be non-negative: {topk_blocks=}")
    if row_repeat <= 0:
        raise ValueError(f"row_repeat must be positive: {row_repeat=}")

    scores_v = TensorValue(scores)
    t_dim, w_dim = scores_v.shape[-2], scores_v.shape[-1]
    if not isinstance(t_dim, StaticDim) or not isinstance(w_dim, StaticDim):
        raise ValueError(f"candidate_pool requires static T, W: {scores_v=}")
    t, w = int(t_dim), int(w_dim)
    device = scores_v.device

    if w == 0 or t == 0:
        # The graph compiler rejects a zero-sized leading dim in the
        # gather/top_k chain below.
        return ops.broadcast_to(
            ops.constant(-1, DType.int32, device), (t, topk_blocks)
        )

    nblocks = -(-w // block_size)  # ceil division; w, block_size are static.
    expected_groups = -(-t // row_repeat)

    def _per_row_groups(x: TensorValueLike, name: str) -> TensorValue:
        xv = ops.cast(TensorValue(x), DType.int32)
        group_dim = xv.shape[0]
        if isinstance(group_dim, StaticDim) and int(group_dim) != (
            expected_groups
        ):
            raise ValueError(
                f"{name} must have {expected_groups} groups for {t=},"
                f" {row_repeat=}: {xv=}"
            )
        group_of_row = ops.constant(
            [r // row_repeat for r in range(t)], DType.int32, device
        )
        return ops.gather(xv, group_of_row, axis=0)  # [T]

    end_per_row = _per_row_groups(row_end, "row_end")
    start_per_row = (
        ops.constant([0] * t, DType.int32, device)
        if row_start is None
        else _per_row_groups(row_start, "row_start")
    )

    col_iota = ops.constant(
        list(range(nblocks * block_size)), DType.int32, device
    )
    abs_col = ops.unsqueeze(start_per_row, -1) + ops.unsqueeze(
        col_iota, 0
    )  # [T, nblocks*block_size]
    end_col = ops.unsqueeze(end_per_row, -1)  # [T, 1]
    in_bound = abs_col < end_col

    # Clamp gather indices into [0, W); `in_bound` masks the excess to -inf.
    gather_col = ops.min(abs_col, w - 1)
    gathered = ops.gather_nd(
        scores_v, ops.unsqueeze(gather_col, -1), batch_dims=1
    )
    masked = ops.where(in_bound, gathered, float("-inf"))
    masked_blocks = ops.reshape(masked, (t, nblocks, block_size))

    # Max over a NaN is backend-defined, so propagate NaN explicitly.
    any_nan = (
        ops.max(ops.cast(ops.is_nan(masked_blocks), DType.float32), axis=-1) > 0
    )
    block_max = _reinstate_true_neg_inf(ops.max(masked_blocks, axis=-1))
    block_scores = ops.squeeze(
        ops.where(any_nan, float("nan"), block_max), -1
    )  # [T, nblocks]

    start_col = ops.unsqueeze(start_per_row, -1)
    nonempty = end_col > start_col  # [T, 1]
    newest_idx = ops.floor_div(end_per_row - start_per_row - 1, block_size)
    block_iota = ops.constant(list(range(nblocks)), DType.int32, device)
    force_mask = (
        ops.unsqueeze(newest_idx, -1) == ops.unsqueeze(block_iota, 0)
    ) & nonempty
    # The force overwrites even a NaN block max.
    forced_scores = ops.where(force_mask, float("inf"), block_scores)

    k = min(topk_blocks, nblocks)
    top_vals, top_idx = ops.top_k(forced_scores, k=k, axis=-1)
    top_idx = ops.cast(top_idx, DType.int32)
    # Only `-inf` re-pads to -1; a selected NaN keeps its id.
    ids = ops.where(top_vals == float("-inf"), -1, top_idx)

    if k < topk_blocks:
        pad = ops.broadcast_to(
            ops.constant(-1, DType.int32, device), (t, topk_blocks - k)
        )
        ids = ops.concat([ids, pad], axis=-1)
    return ids


def candidate_blocks_active(
    width: int, compress_ratio: int, *, block_size: int, topk_blocks: int
) -> bool:
    """Returns whether :func:`candidate_block_keep` can drop a reachable block.

    False when ``ceil(width / (block_size * compress_ratio)) <=
    topk_blocks`` (every block is kept, so the pass can be skipped) or when
    any size is not positive (the mechanism is off).
    """
    if block_size <= 0 or topk_blocks <= 0 or compress_ratio <= 0:
        return False
    group = block_size * compress_ratio
    return -(-width // group) > topk_blocks


def candidate_block_keep(
    scores: TensorValueLike,
    positions: TensorValueLike,
    *,
    compress_ratio: int,
    block_size: int,
    topk_blocks: int,
) -> TensorValue:
    """Returns a mask keeping each row's ``topk_blocks`` best-scoring blocks.

    Scores are laid out by raw token: compressed entry ``j`` sits in column
    ``j * ratio + ratio - 1``, so a block of ``block_size`` compressed
    positions spans ``block_size * ratio`` raw columns. Per row:

    1. Score each block by its max column, ``-inf``-padding the last block.
    2. Force the block of the newest compressed position, ``(compress_len -
       1) // block_size`` with ``compress_len = (pos + 1) // ratio``, to
       ``+inf`` (none when ``compress_len == 0``).
    3. Keep the ``topk_blocks`` best blocks, dropping ``-inf`` picks.

    Step 3 avoids a scatter: blocks above the ``k``-th best are kept, and
    blocks tied with it fill the remaining picks lowest index first, so
    exactly ``min(k, reachable)`` are kept regardless of ``top_k``'s tie
    order. NaN columns reduce as the backend's ``max`` does (not propagated).

    Args:
        scores: ``[T, W]`` float32 indexer scores, complete rather than one
            device's partial, with every column the row cannot see already
            ``-inf``.
        positions: ``[T]`` int32 absolute query positions.
        compress_ratio: The layer's compression ratio (1 or 2).
        block_size: Compressed positions per block.
        topk_blocks: Blocks kept per row.

    Returns:
        ``[T, ceil(W / g)]`` bool keep mask, block ``b`` covering raw
        columns ``[b * g, (b + 1) * g)`` with ``g = block_size *
        compress_ratio``. Expand it with :func:`expand_candidate_blocks`.

    Raises:
        ValueError: If a size is not positive or ``W`` is not static.
    """
    if block_size <= 0 or topk_blocks <= 0 or compress_ratio <= 0:
        raise ValueError(
            f"candidate blocks need positive sizes: {block_size=},"
            f" {topk_blocks=}, {compress_ratio=}"
        )
    scores_v = TensorValue(scores)
    t_dim, w_dim = scores_v.shape[-2], scores_v.shape[-1]
    if not isinstance(w_dim, StaticDim):
        raise ValueError(f"candidate_block_keep requires static W: {scores_v=}")
    w = int(w_dim)
    device = scores_v.device
    group = block_size * compress_ratio
    nblocks = -(-w // group)
    pad = nblocks * group - w
    if pad:
        scores_v = ops.concat(
            [
                scores_v,
                ops.broadcast_to(
                    ops.constant(float("-inf"), DType.float32, device),
                    [t_dim, pad],
                ),
            ],
            axis=-1,
        )
    block_scores = ops.squeeze(
        _reinstate_true_neg_inf(
            ops.max(ops.reshape(scores_v, [t_dim, nblocks, group]), axis=-1)
        ),
        -1,
    )  # [T, nblocks]

    pos = ops.cast(TensorValue(positions), DType.int32)
    compress_len = ops.floor_div(pos + 1, compress_ratio)
    last = ops.floor_div(compress_len - 1, block_size)  # [T]
    block_iota = ops.unsqueeze(
        ops.range(0, nblocks, dtype=DType.int32, device=device), 0
    )  # [1, nblocks]
    forced = ops.where(
        ops.unsqueeze(last, -1) == block_iota, float("inf"), block_scores
    )

    k = min(topk_blocks, nblocks)
    top_vals, _ = ops.top_k(forced, k=k, axis=-1)
    kth = top_vals[:, k - 1 : k]  # [T, 1]
    above = forced > kth
    tied = forced == kth
    # Picks left for the blocks tied with the k-th best: >= 1 by definition.
    need = k - ops.sum(ops.cast(above, DType.int32), axis=-1)  # [T, 1]
    tied_ids = ops.where(tied, block_iota, nblocks)
    # Ascending tied block ids (negated for `top_k`'s largest-first order).
    lowest_neg, _ = ops.top_k(-tied_ids, k=k, axis=-1)
    last_tied = -ops.gather_nd(
        lowest_neg, ops.unsqueeze(need - 1, -1), batch_dims=1
    )  # [T, 1]
    keep = above | (tied & (block_iota <= last_tied))
    return keep & (forced > float("-inf"))


def expand_candidate_blocks(
    keep: TensorValueLike, width: int, *, compress_ratio: int, block_size: int
) -> TensorValue:
    """Expands a ``[T, nblocks]`` block keep mask to ``[T, width]`` columns.

    Each block spans ``block_size * compress_ratio`` raw columns; columns
    past ``width`` are dropped.
    """
    keep_v = TensorValue(keep)
    t_dim, nb_dim = keep_v.shape[-2], keep_v.shape[-1]
    nblocks = int(nb_dim)
    group = block_size * compress_ratio
    expanded = ops.reshape(
        ops.broadcast_to(ops.unsqueeze(keep_v, -1), [t_dim, nblocks, group]),
        [t_dim, nblocks * group],
    )
    if nblocks * group != width:
        expanded = expanded[:, :width]
    return expanded


# ---------------------------------------------------------------------------
# Position-ordered candidate rows. Each query's picked KV positions are
# stored in a ``-1``-padded row of fixed width, sorted by ascending position
# so they can be walked in sequence rather than in score order.
# ---------------------------------------------------------------------------

# Sorts padding after every legal position; int32-safe under negation.
_POSITION_SENTINEL = (1 << 31) - 1


@overload
def position_order(
    indices: TensorValueLike, *, scores: None = None, invalid: int = -1
) -> TensorValue: ...


@overload
def position_order(
    indices: TensorValueLike, *, scores: TensorValueLike, invalid: int = -1
) -> tuple[TensorValue, TensorValue]: ...


def position_order(
    indices: TensorValueLike,
    *,
    scores: TensorValueLike | None = None,
    invalid: int = -1,
) -> TensorValue | tuple[TensorValue, TensorValue]:
    """Reorders a candidate index row by ascending position.

    Padding entries (``invalid``) stay trailing.

    Args:
        indices: ``[..., width]`` integer position ids, ``invalid``-padded.
        scores: Optional ``[..., width]`` values parallel to ``indices``,
            reordered the same way.
        invalid: The padding sentinel.

    Returns:
        The reordered int32 indices, or ``(indices, scores)`` if ``scores``
        was given.

    Raises:
        ValueError: If ``width`` is not static.
    """
    idx = ops.cast(TensorValue(indices), DType.int32)
    width_dim = idx.shape[-1]
    if not isinstance(width_dim, StaticDim):
        raise ValueError(f"position_order requires a static width: {idx=}")
    width = int(width_dim)
    if width == 0:
        return (idx, TensorValue(scores)) if scores is not None else idx

    is_valid = idx != invalid
    sentinel = ops.constant(_POSITION_SENTINEL, DType.int32, idx.device)
    sort_key = ops.where(is_valid, idx, sentinel)
    # Negate so `top_k`'s largest-first order is ascending position.
    _, order = ops.top_k(-sort_key, k=width, axis=-1)

    batch_dims = idx.rank - 1
    gather_idx = ops.unsqueeze(order, -1)
    reordered_idx = ops.gather_nd(idx, gather_idx, batch_dims=batch_dims)

    if scores is None:
        return reordered_idx

    reordered_scores = ops.gather_nd(
        TensorValue(scores), gather_idx, batch_dims=batch_dims
    )
    return reordered_idx, reordered_scores


def bounded_position_order(
    scores: TensorValueLike,
    valid: TensorValueLike,
    width: int,
) -> tuple[TensorValue, TensorValue]:
    """Selects the top ``width`` visible candidates, ordered by position.

    Masks ``not valid`` to ``-inf`` before top-k and stores a ``-inf`` pick
    as ``-1``. Column ``s`` is KV position ``s``, so ``top_k``'s indices are
    the position ids.

    Args:
        scores: ``[..., S]`` dense float32 scores, one column per position.
        valid: ``[..., S]`` bool; ``False`` where a query cannot see a
            candidate.
        width: Output candidate width.

    Returns:
        ``(positions, picked_scores)``, both ``[..., min(width, S)]``
        (``[..., width]`` when ``S`` is dynamic): positions
        ascending and ``-1``-padded, with each position's score. Padded
        scores are meaningless; gate on ``positions != -1``.
    """
    scores_v = TensorValue(scores)
    s_dim = scores_v.shape[-1]
    k = min(width, int(s_dim)) if isinstance(s_dim, StaticDim) else width

    masked = ops.where(TensorValue(valid), scores_v, float("-inf"))
    top_scores, top_idx = ops.top_k(masked, k=k, axis=-1)
    top_idx = ops.cast(top_idx, DType.int32)
    positions = ops.where(top_scores > float("-inf"), top_idx, -1)

    ordered_positions, ordered_scores = position_order(
        positions, scores=top_scores
    )
    return ordered_positions, ordered_scores
