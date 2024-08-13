# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""The attention mechanism used within the model."""

import math
from dataclasses import dataclass

import numpy as np
from max.graph import DimLike, DType, GraphValue, ValueLike, ops

from .mlp import Linear
from .rotary_embedding import RotaryEmbedding


def attention_mask(
    start_pos: DimLike, seq_len: DimLike, activation_dtype: DType
) -> GraphValue:
    """Computes Attention mask."""
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

    def attention(
        self,
        xq: ValueLike,
        xk: ValueLike,
        xv: ValueLike,
        k_cache: ValueLike,
        v_cache: ValueLike,
    ) -> GraphValue:
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
    ) -> GraphValue:
        """Computes attention on x, reusing the KV cache.

        Args:
            x: Activations with shape (batch, seq_len, dim).
            k_cache: Previously computed keys with shape
                (prev_seq_len, 1, batch, n_kv_heads, head_dim).
            v_cache: Previously computed values with shape
                (prev_seq_len, 1, batch, n_kv_heads, head_dim).

        Returns the result of multi-headed self attention on the input.
        """
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
        xk = self.rope(xk, start_pos, seq_len)

        output = (
            self.attention(xq, xk, xv, k_cache, v_cache)
            .transpose(1, 2)
            .reshape([batch, seq_len, -1])
        )

        return self.wo(output), xk, xv
