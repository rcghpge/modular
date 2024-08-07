# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
import numpy as np
from max.graph import GraphValue, ops, ValueLike, TensorType, DType
from .mlp import Linear
from .rotary_embedding import RotaryEmbedding
import math


def rope(x: ValueLike, freqs_cis: ValueLike) -> GraphValue:
    v = GraphValue(x)

    complex = ops.as_interleaved_complex(v)
    freqs_cis_bcast = ops.unsqueeze(ops.unsqueeze(freqs_cis, 1), 0)

    x_re = complex[..., 0]
    x_im = complex[..., 1]

    freqs_re = freqs_cis_bcast[..., 0]
    freqs_im = freqs_cis_bcast[..., 1]

    # TODO: Replace with '-' and '+' below when supported.
    rope_re = ops.sub((x_re * freqs_re), (x_im * freqs_im))
    rope_im = ops.add((x_re * freqs_im), (x_im * freqs_re))

    # TODO: Use stack (or as_complex?) here.
    # TODO: rope_complex = ops.concat([rope_re[..., None], rope_im[..., None]], axis=-1) should also work, but does not
    rope_complex = ops.concat(
        [ops.unsqueeze(rope_re, -1), ops.unsqueeze(rope_im, -1)], axis=-1
    )

    return ops.reshape(rope_complex, v.tensor_type.shape)


def attention_mask(start_pos, seq_len, activation_dtype):
    mask_val = ops.cast(
        ops.broadcast_to(
            ops.scalar(float("-inf"), dtype=DType.float32),
            shape=[seq_len, seq_len],
        ),
        activation_dtype,
    )
    mask = ops.band_part(mask_val, -1, 0, exclude=True)
    return ops.concat(
        [
            ops.cast(
                ops.broadcast_to(
                    ops.scalar(0, dtype=DType.float32),
                    shape=[seq_len, start_pos],
                ),
                activation_dtype,
            ),
            mask,
        ],
        axis=1,
    )


@dataclass
class Attention:
    n_heads: int
    n_kv_heads: int
    head_dim: int
    dim: int

    wq: Linear
    wk: Linear
    wv: Linear
    wo: Linear

    rope: RotaryEmbedding

    def repeat_kv(self, kv: GraphValue):
        """Repeats key/value tensors to match the number of query heads."""
        batch = kv.tensor_type.shape[0]
        kv = ops.reshape(kv, [batch, -1, self.n_kv_heads, 1, self.head_dim])

        # TODO (MSDK-655): Should use a ops.tile() implementation here when available.
        num_tiles = [1, 1, 1, self.n_heads // self.n_kv_heads, 1]
        for i, n in enumerate(num_tiles):
            tiles = [kv for _ in range(n)]
            kv = ops.concat(tiles, axis=i)
        return ops.reshape(kv, [batch, -1, self.n_heads, self.head_dim])

    def attention(self, xq, xk, xv, k_cache, v_cache):
        prev_seq_len = k_cache.tensor_type.shape[0]
        seq_len = xq.tensor_type.shape[1]

        k_cache = ops.squeeze(k_cache, axis=1)
        v_cache = ops.squeeze(v_cache, axis=1)

        keys = ops.transpose(
            ops.concat([k_cache, ops.transpose(xk, 0, 1)]), 0, 1
        )
        values = ops.transpose(
            ops.concat([v_cache, ops.transpose(xv, 0, 1)]), 0, 1
        )

        keys = self.repeat_kv(keys)
        values = self.repeat_kv(values)

        xq = ops.transpose(xq, 1, 2)
        keys = ops.transpose(keys, 1, 2)
        values = ops.transpose(values, 1, 2)

        scores = (xq @ ops.transpose(keys, 2, 3)) * ops.scalar(
            math.sqrt(1.0 / self.head_dim), dtype=xq.tensor_type.dtype
        )

        scores = scores + attention_mask(
            prev_seq_len, seq_len, xq.tensor_type.dtype
        )

        return ops.softmax(scores) @ values

    def __call__(
        self,
        x: ValueLike,
        k_cache: ValueLike,
        v_cache: ValueLike,
    ):
        batch, seq_len = x.tensor_type.shape[0], x.tensor_type.shape[1]
        start_pos = k_cache.shape[0]
        # matmul weights
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = ops.reshape(xq, [batch, seq_len, self.n_heads, self.head_dim])
        xk = ops.reshape(xk, [batch, seq_len, self.n_heads, self.head_dim])
        xv = ops.reshape(xv, [batch, seq_len, self.n_heads, self.head_dim])

        xq = self.rope(xq, start_pos, seq_len)
        xk = self.rope(xq, start_pos, seq_len)

        output = self.attention(xq, xk, xv, k_cache, v_cache)

        return output
