# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Llama3 model hyperparameters."""

from dataclasses import dataclass


@dataclass
class Hyperparameters:
    seq_len: int = 2048
    """Maximum length of the token sequence that can be processed by this model."""

    n_layers: int = 32
    """Number of MultiHeadAttention layers to use in this model."""

    n_heads: int = 32
    """Number of heads for the query to use in the MultiHeadAttention layers."""

    n_kv_heads: int = 8
    """Number of key and value heads to use in the MultiHeadAttention layers."""

    vocab_size: int = 128256
    """Number of tokens in the vocabulary."""

    hidden_dim: int = 4096
    """Hidden dimension of embedded tokens."""

    rope_theta: float = 500000.0
    """Rotary period hyperparameter for rope embeddings."""

    layer_norm_rms_epsilon: float = 1e-5
    """Epsilon value for layer norm calculation."""

    @property
    def head_dim(self):
        """Dimension of each head."""
        return self.hidden_dim // self.n_heads

    @property
    def kv_weight_dim(self):
        """Dimension of the key and value attention weights."""
        return self.head_dim * self.n_kv_heads
