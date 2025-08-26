# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Transformer components for XCodec2 (stub implementation)."""

from __future__ import annotations

import max.nn as nn
from max.graph import TensorValue

from .config import XCodec2Config


class TransformerBlock(nn.Module):
    """Transformer block for XCodec2.

    This is a stub implementation that will be replaced with the full
    TransformerBlock implementation (with rotary embeddings) in a future PR.
    """

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim  # 1024
        self.num_heads = config.heads  # 16

        # TODO: Implement full transformer block in future PR:
        # - Multi-head attention with rotary embeddings
        # - Feed-forward network
        # - Layer normalization
        # - Residual connections

        # For now, just a simple pass-through layer
        self.dummy_layer = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            dtype=config.dtype,
            device=config.device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass of TransformerBlock (stub implementation).

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        # TODO: Replace with real transformer processing
        # For now, just apply a simple linear transformation
        return self.dummy_layer(x)
