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
        """Insert the updated key and value cache elements in the main cache."""
        key_length = new_keys.shape[0]
        new_sequence_length = self.sequence_length + key_length
        self.keys[self.sequence_length : new_sequence_length, ...] = new_keys
        self.values[
            self.sequence_length : new_sequence_length, ...
        ] = new_values
        self.sequence_length = new_sequence_length

    def keys_view(self) -> np.array:
        """A view into the main key cache."""
        return self.keys[0 : self.sequence_length, ...]

    def values_view(self) -> np.array:
        """A view into the main value cache."""
        return self.values[0 : self.sequence_length, ...]
