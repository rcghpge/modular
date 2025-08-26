# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""ResidualFSQ implementation for XCodec2 (stub implementation)."""

from __future__ import annotations

import max.nn as nn
from max.dtype import DType
from max.graph import TensorValue, ops

from .config import XCodec2Config


class ResidualFSQ(nn.Module):
    """Residual Finite Scalar Quantization for XCodec2.

    This is a stub implementation that will be replaced with the full
    ResidualFSQ implementation in a future PR."""

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.vq_dim = config.vq_dim  # 2048
        self.levels = [4, 4, 4, 4, 4, 4, 4, 4]  # 8 levels of 4 each
        self.num_quantizers = 1

        # TODO: Implement actual ResidualFSQ in future PR
        # For now, create a simple embedding lookup table
        self.codebook_size = (
            16384  # From original CodecDecoderVocos constructor
        )
        self.embedding = nn.Embedding(
            self.codebook_size,
            self.vq_dim,
            dtype=config.dtype,
            device=config.device,
        )

    def __call__(self, x: TensorValue) -> tuple[TensorValue, TensorValue]:
        """Forward pass of ResidualFSQ (stub implementation).

        Args:
            x: Input tensor of shape (batch, seq_len, vq_dim)

        Returns:
            Tuple of (quantized_embeddings, indices)
            - quantized_embeddings: shape (batch, seq_len, vq_dim)
            - indices: shape (batch, seq_len)
        """
        batch_size, seq_len, dim = x.shape

        # TODO: Replace with real quantization logic
        # For now, just return dummy indices and lookup embeddings
        dummy_indices = ops.constant(
            0, dtype=DType.int32, device=self.config.device
        ).broadcast_to((batch_size, seq_len))
        quantized = self.embedding(dummy_indices)

        return quantized, dummy_indices

    def get_output_from_indices(self, indices: TensorValue) -> TensorValue:
        """Get quantized embeddings from indices.

        Args:
            indices: Token indices of shape (batch, seq_len)

        Returns:
            Quantized embeddings of shape (batch, seq_len, vq_dim)
        """
        # Clamp indices to valid range to avoid errors
        from max import nn

        clamped_indices = nn.clamp(indices, 0, self.codebook_size - 1)
        return self.embedding(clamped_indices)
