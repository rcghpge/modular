# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""ResidualFSQ implementation for XCodec2."""

from __future__ import annotations

from functools import cached_property

import max.nn as nn
import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from .config import XCodec2Config


class FSQ:
    """Finite Scalar Quantization implementation.

    Based on "Finite Scalar Quantization: VQ-VAE Made Simple" - https://arxiv.org/abs/2309.15505
    """

    def __init__(
        self,
        levels: list[int],
        dim: int,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        self.levels = levels
        self.codebook_dim = len(levels)
        self.dim = dim
        self.device = device

        # Calculate basis: cumulative product [1, levels[0], levels[0]*levels[1], ...]
        # For levels [4,4,4,4,4,4,4,4]: basis = [1, 4, 16, 64, 256, 1024, 4096, 16384]
        self.basis_list = [1]
        for i in range(len(levels) - 1):
            self.basis_list.append(self.basis_list[-1] * levels[i])

        # Codebook size: product of all levels = 4^8 = 65536
        self.codebook_size = 1
        for level in levels:
            self.codebook_size *= level

    @cached_property
    def levels_tensor(self) -> TensorValue:
        """Get levels tensor, creating it if needed."""
        return ops.constant(
            np.array(self.levels), dtype=DType.int32, device=self.device
        )

    @cached_property
    def basis_tensor(self) -> TensorValue:
        """Get basis tensor, creating it if needed."""
        return ops.constant(
            np.array(self.basis_list), dtype=DType.int32, device=self.device
        )

    def indices_to_level_indices(self, indices: TensorValue) -> TensorValue:
        """Convert single indices to level indices for each dimension.

        Args:
            indices: Token indices of shape (batch, seq_len)

        Returns:
            Level indices of shape (batch, seq_len, codebook_dim)
        """
        # Add dimension for broadcasting: (batch, seq_len) → (batch, seq_len, 1)
        indices = ops.unsqueeze(indices, -1)

        # Convert to level indices: (indices // basis) % levels
        # This decomposes the single index into 8 level indices (0-3 each)
        level_indices = (indices // self.basis_tensor) % self.levels_tensor

        return level_indices

    def _scale_and_shift_inverse(
        self, level_indices: TensorValue
    ) -> TensorValue:
        """Convert level indices to normalized codes.

        Args:
            level_indices: Level indices of shape (batch, seq_len, codebook_dim)

        Returns:
            Normalized codes of shape (batch, seq_len, codebook_dim)
        """
        # Convert level indices (0,1,2,3) to normalized range
        # Formula: (level_indices - half_width) / half_width
        # For levels [4,4,4,4,4,4,4,4]: half_width = [2,2,2,2,2,2,2,2]
        half_width = self.levels_tensor // 2  # [2,2,2,2,2,2,2,2]
        half_width = half_width.cast(DType.float32)
        level_indices = level_indices.cast(DType.float32)

        codes = (level_indices - half_width) / half_width
        return codes

    def _indices_to_codes(self, indices: TensorValue) -> TensorValue:
        """Convert indices to normalized codes.

        Args:
            indices: Token indices of shape (batch, seq_len)

        Returns:
            Normalized codes of shape (batch, seq_len, codebook_dim)
        """
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes


class ResidualFSQ(nn.Module):
    """Residual Finite Scalar Quantization for XCodec2.

    Based on "Finite Scalar Quantization: VQ-VAE Made Simple" with residual quantization.
    Implements the decoder path: token indices → embeddings via learned projections.
    """

    def __init__(self, config: XCodec2Config) -> None:
        super().__init__()
        self.config = config
        self.dim = config.vq_dim  # 2048 (input/output dimension)
        self.levels = [4, 4, 4, 4, 4, 4, 4, 4]  # 8 levels of 4 each
        self.num_quantizers = 1
        self.codebook_dim = len(self.levels)  # 8

        # Projection layers (matching HuggingFace weights)
        # project_in: 2048 → 8 (compress to quantization space)
        self.project_in = nn.Linear(
            self.dim,
            self.codebook_dim,
            has_bias=True,
            dtype=config.dtype,
            device=config.device,
        )

        # project_out: 8 → 2048 (expand back to embedding space)
        self.project_out = nn.Linear(
            self.codebook_dim,
            self.dim,
            has_bias=True,
            dtype=config.dtype,
            device=config.device,
        )

        # FSQ layer for handling the quantization logic
        self.fsq = FSQ(
            levels=self.levels,
            dim=self.codebook_dim,
            dtype=config.dtype,
            device=config.device,
        )

        # Scales will be created lazily via cached_property

    @cached_property
    def scales(self) -> TensorValue:
        """Get scales tensor, creating it if needed."""
        # For single quantizer (num_quantizers=1), scales = [1, 1, 1, 1, 1, 1, 1, 1]
        # Since (levels - 1) ** 0 = 1 for all levels
        ones = np.ones(self.codebook_dim, dtype=np.float32)
        return ops.constant(
            ones, dtype=DType.float32, device=self.config.device
        )

    def get_codes_from_indices(self, indices: TensorValue) -> TensorValue:
        """Convert indices to codes using FSQ codebook.

        Args:
            indices: Token indices of shape (batch, seq_len)

        Returns:
            Codes of shape (batch, seq_len, codebook_dim)
        """
        # Basic validation
        if indices.rank != 2:
            raise ValueError(
                f"Expected indices to have rank 2, got {indices.rank}"
            )

        # Get codes from indices using FSQ
        codes = self.fsq._indices_to_codes(indices)  # (batch, seq_len, 8)

        # For single quantizer with scales=1, no scaling needed
        return codes

    def get_output_from_indices(self, indices: TensorValue) -> TensorValue:
        """Get quantized embeddings from indices (main method for Llasa).

        Args:
            indices: Token indices of shape (batch, seq_len)

        Returns:
            Quantized embeddings of shape (batch, seq_len, vq_dim)
        """
        # Convert indices to codes
        codes = self.get_codes_from_indices(indices)  # (batch, seq_len, 8)

        # Project back to embedding space (8D → 2048D)
        embeddings = self.project_out(codes)  # (batch, seq_len, 2048)

        return embeddings

    def __call__(self, x: TensorValue) -> tuple[TensorValue, TensorValue]:
        """Forward pass of ResidualFSQ (not needed for Llasa, but kept for completeness).

        Args:
            x: Input tensor of shape (batch, seq_len, vq_dim)

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "ResidualFSQ should not be called directly in Llasa"
        )
