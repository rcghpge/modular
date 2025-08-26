# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""VocosBackbone implementation for XCodec2 (stub implementation)."""

from __future__ import annotations

import max.nn as nn
from max.graph import TensorValue

from .config import XCodec2Config


class VocosBackbone(nn.Module):
    """Vocos backbone module for XCodec2.

    This is a stub implementation that will be replaced with the full
    VocosBackbone implementation (ResNet blocks + Transformers) in a future PR.
    """

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim  # 1024

        # TODO: Implement full backbone in future PR:
        # - Conv1d embedding layer
        # - ResNet blocks (prior_net)
        # - Transformer blocks with rotary embeddings
        # - Final layer norm
        # - ResNet blocks (post_net)

        # For now, just a simple pass-through layer
        self.dummy_layer = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            dtype=config.dtype,
            device=config.device,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass of VocosBackbone (stub implementation).

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        # TODO: Replace with real backbone processing
        # For now, just apply a simple linear transformation
        return self.dummy_layer(x)
