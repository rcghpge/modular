# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""KV cache for the Transformer."""

import numpy as np


class KVCache:
    keys: np.array
    values: np.array
    sequence_length: int

    def __init__(
        self,
        max_length: int,
        batch_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        self.keys = np.zeros(
            shape=(max_length, n_layers, batch_size, n_kv_heads, head_dim)
        ).astype(np.float32)
        self.values = np.zeros(
            shape=(max_length, n_layers, batch_size, n_kv_heads, head_dim)
        ).astype(np.float32)
        self.sequence_length = 0

    def update(self, new_keys: np.array, new_values: np.array):
        # TODO: Write new KV cache entries at appropriate location in main cache.
        pass

    def keys_view(self) -> np.array:
        return self.keys

    def values_view(self) -> np.array:
        return self.values
