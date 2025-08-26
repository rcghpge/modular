# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Configuration for XCodec2 model components."""

from __future__ import annotations

from dataclasses import dataclass, field

from max.dtype import DType
from max.graph import DeviceRef


@dataclass
class XCodec2Config:
    """Configuration for XCodec2 audio decoder components."""

    # Audio generation parameters (from CodecDecoderVocos)
    hidden_dim: int = 1024
    """Hidden dimension of the model."""

    depth: int = 12
    """Number of transformer layers."""

    heads: int = 16
    """Number of attention heads."""

    pos_emb_dim: int = 64
    """Positional embedding dimension."""

    hop_length: int = 320
    """Hop length for STFT processing."""

    vq_dim: int = 2048
    """Vector quantizer dimension."""

    # Audio parameters
    sample_rate: int = 16000
    """Output audio sample rate (16kHz as specified by xCodec2)."""

    # Model parameters
    dtype: DType = DType.float32
    """Data type for model weights and computations."""

    device: DeviceRef = field(default_factory=lambda: DeviceRef.GPU())
    """Device for model execution."""

    @property
    def n_fft(self) -> int:
        """FFT size for ISTFT (hop_length * 4)."""
        return self.hop_length * 4

    @property
    def codec_tokens_per_sec(self) -> int:
        """Number of codec tokens per second (xCodec2 specification)."""
        return 50
