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
"""CSA2 sparse attention over quantized, paged window and compressed caches.

A layer keeps two byte caches of ``[pages, page_bytes]`` uint8. The window
cache holds a 528-byte MXFP8 record per token: 512 E4M3 values, then 16
UE8M0 scales of 32 channels each. The compressed cache holds a 288-byte
NVFP4 record per group of ``ratio`` tokens: 256 bytes of E2M1 pairs, then 32
E4M3 scales of 16 channels each. Both rotate the last 64 channels with
interleaved RoPE before quantizing. A page holds ``page_size`` records, all
of their values and then all of their scales; slot ``s`` is offset
``s % page_size`` of page ``s // page_size``.

Each query token names up to 128 window slots and up to 512 compressed
slots in a ``[T, 640]`` int32 list, ``-1`` for none, and its heads attend
over those records through one softmax, with a per-head sink logit that
joins the denominator only. Each record is both key and value.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    ops,
)

HEAD_DIM = 512
"""Channels of a query head, a latent and a dequantized record."""
WINDOW_RECORD_BYTES = 528
"""Bytes of an MXFP8 window record."""
COMPRESSED_RECORD_BYTES = 288
"""Bytes of an NVFP4 compressed record."""
INDEX_WIDTH = 640
"""Entries of a token's index list: 128 window slots, then 512 compressed."""
MAX_COMPRESSED = 512
"""The most compressed slots a token's list names."""
DECODE_SPLIT_ROWS = 296
"""Partial rows (``splits * tokens``) a decode workspace holds by default,
two waves of a B200's CTAs; a decode splits a token's work only while its
partials fit."""


def cache_page_bytes(page_size: int, record_bytes: int, data_bytes: int) -> int:
    """Returns the bytes of a cache page of ``page_size`` records.

    The attention reads a multi-record page's values in rows of
    ``data_bytes``, so such a page rounds up to a multiple of it.

    Args:
        page_size: Records a page holds.
        record_bytes: Bytes of a record.
        data_bytes: Bytes of a record's values.
    """
    raw = page_size * record_bytes
    if page_size == 1:
        return -(-raw // 16) * 16
    return -(-raw // data_bytes) * data_bytes


def _host_f32(value: float) -> TensorValue:
    return ops.constant([value], DType.float32, DeviceRef.CPU())


def store_window_kv(
    cache: BufferValue,
    latent: TensorValue,
    positions: TensorValue,
    rope_table: TensorValue,
    slots: TensorValue,
    *,
    page_size: int,
) -> None:
    """RoPEs, quantizes to MXFP8 and stores each row's window record.

    Args:
        cache: The window cache, ``[pages, page_bytes]`` uint8.
        latent: ``[T, 512]`` bf16 KV latents before RoPE.
        positions: ``[T]`` token positions.
        rope_table: ``[rows, 64]`` float32 cosines then sines per position.
        slots: ``[T]`` int64 slots, negative to skip a row.
        page_size: Records a cache page holds.
    """
    ops.inplace_custom(
        "mo.csa.window_kv_store",
        device=latent.device,
        values=[
            latent,
            ops.cast(positions, DType.int64),
            rope_table,
            ops.cast(slots, DType.int64),
            cache,
        ],
        parameters={"page_size": page_size},
    )


def store_compressed_kv(
    cache: BufferValue,
    latent: TensorValue,
    positions: TensorValue,
    rope_table: TensorValue,
    slots: TensorValue,
    *,
    page_size: int,
    ratio: int,
) -> None:
    """RoPEs, quantizes to NVFP4 and stores each group-closing row's record.

    The trailing 64 channels rotate at the group's first position.

    Args:
        cache: The compressed cache, ``[pages, page_bytes]`` uint8.
        latent: ``[T, 512]`` bf16 pooled latents before RoPE.
        positions: ``[T]`` token positions.
        rope_table: ``[rows, 64]`` float32 cosines then sines per position.
        slots: ``[T]`` int64 slots, negative to skip a row.
        page_size: Records a cache page holds.
        ratio: Positions pooled into one record, 1 or 2.
    """
    ops.inplace_custom(
        "mo.csa.compressed_kv_store",
        device=latent.device,
        values=[
            latent,
            ops.cast(positions, DType.int64),
            rope_table,
            ops.cast(slots, DType.int64),
            cache,
        ],
        parameters={"page_size": page_size, "ratio": ratio},
    )


def csa2_attention(
    q: TensorValue,
    idx: TensorValue,
    sink: TensorValue,
    kv_window: BufferValue,
    kv_comp: BufferValue | None,
    *,
    scale: float,
    window_page: int,
    comp_page: int = 1,
    compressed_rows: int = MAX_COMPRESSED,
    prefill: bool = False,
    split_rows: int = DECODE_SPLIT_ROWS,
) -> TensorValue:
    """Returns ``[T, n_heads, 512]`` bf16 sparse attention outputs.

    Decode reads the records in place and suits up to 8 query tokens a
    sequence. Prefill first dequantizes every slot of both caches to bf16,
    so its scratch and time grow with the caches' size.

    Args:
        q: ``[T, n_heads, 512]`` bf16 queries.
        idx: ``[T, 640]`` int32 window then compressed slots, ``-1`` for
            none, the compressed ones packed first.
        sink: ``[n_heads]`` float32 sink logits.
        kv_window: The window cache.
        kv_comp: The compressed cache, ``None`` on a window-only layer.
        scale: Softmax scale.
        window_page: Records a window cache page holds.
        comp_page: Records a compressed cache page holds.
        compressed_rows: The most compressed entries any list names, at
            most 512; it sets the tiles each token walks.
        prefill: Whether to dequantize the caches first.
        split_rows: Partial rows a decode's split workspace holds.
    """
    t, n_heads = q.shape[0], int(q.shape[1])
    device = q.device
    window_only = kv_comp is None
    name = "mo.csa2." + ("prefill" if prefill else "decode")
    parameters: dict[str, bool | int | float | str | DType] = {
        "num_heads": n_heads,
        "window_page": window_page,
    }
    if window_only:
        name += ".window_only"
    else:
        parameters["compressed_rows"] = compressed_rows
        parameters["comp_page"] = comp_page
    if prefill:
        values = kv_window.shape[0] * window_page
        if kv_comp is not None:
            values = values + kv_comp.shape[0] * comp_page
        workspace = TensorType(DType.bfloat16, [values * HEAD_DIM], device)
    else:
        workspace = TensorType(
            DType.float32, [split_rows * n_heads * (HEAD_DIM + 1)], device
        )
    caches = [kv_window] if kv_comp is None else [kv_window, kv_comp]
    return ops.inplace_custom(
        name,
        device=device,
        values=[
            ops.cast(q, DType.bfloat16),
            *caches,
            ops.cast(idx, DType.int32),
            ops.cast(sink, DType.float32),
            _host_f32(scale),
        ],
        out_types=[
            TensorType(DType.bfloat16, [t, n_heads, HEAD_DIM], device),
            workspace,
        ],
        parameters=parameters,
    )[0].tensor
