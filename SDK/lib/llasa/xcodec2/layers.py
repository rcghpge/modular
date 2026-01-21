# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""ResNet block implementation for XCodec2."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import legacy as nn


class ResnetBlock(nn.Module):
    """ResNet block for VocosBackbone."""

    def __init__(
        self,
        in_channels: int,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm1 = nn.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            device=device,
        )
        self.conv1 = nn.Conv1D(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=32,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            device=device,
        )
        self.conv2 = nn.Conv1D(
            kernel_size=3,
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass of ResnetBlock.

        Args:
            x: Input tensor of shape (batch, seq_len, in_channels)

        Returns:
            Output tensor of shape (batch, seq_len, in_channels)
        """
        h = x

        # First block: norm → swish → conv
        # GroupNorm expects (batch, channels, seq_len), so transpose
        h = self.norm1(h.transpose(1, 2)).transpose(
            1, 2
        )  # (batch, seq_len, channels)
        h = h * ops.sigmoid(h)
        h = self.conv1(h)

        # Second block: norm → swish → conv
        # GroupNorm expects (batch, channels, seq_len), so transpose
        h = self.norm2(h.transpose(1, 2)).transpose(
            1, 2
        )  # (batch, seq_len, channels)
        h = h * ops.sigmoid(h)
        h = self.conv2(h)

        # Residual connection
        return x + h
