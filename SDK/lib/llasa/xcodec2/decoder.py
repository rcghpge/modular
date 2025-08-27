# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""High-level XCodec2 decoder wrapper for Llasa integration."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from max.graph import TensorValue

from .config import XCodec2Config
from .weight_loader import load_xcodec2_weights
from .xcodec2 import XCodec2Model

logger = logging.getLogger(__name__)


class XCodec2Decoder:
    """XCodec2 decoder wrapper for Llasa TTS.

    Provides speech token to audio conversion using XCodec2's decoder components.
    Optimized for Llama-generated speech tokens.
    """

    def __init__(self, config: XCodec2Config, model_path: str | Path) -> None:
        """Initialize the XCodec2 decoder.

        Args:
            config: XCodec2 configuration
            model_path: HuggingFace model ID or path to model weights
        """
        self.config = config
        self.xcodec2_model = XCodec2Model(config)

        logger.info(f"Loading XCodec2 weights from: {model_path}")
        weights = load_xcodec2_weights(str(model_path))
        self.xcodec2_model.load_state_dict(
            weights,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )
        logger.info("Successfully loaded XCodec2 weights")

    def decode_code(self, vq_code: TensorValue) -> TensorValue:
        """Decode speech token codes to audio waveform.

        Args:
            vq_code: Speech token indices of shape (batch, seq_len)

        Returns:
            Audio waveform of shape (batch, audio_length)
        """
        return self.xcodec2_model.decode_code(vq_code)

    @property
    def sample_rate(self) -> int:
        """Audio sample rate (16kHz)."""
        return self.config.sample_rate

    @property
    def codec_tokens_per_sec(self) -> int:
        """Number of codec tokens per second (50)."""
        return self.config.codec_tokens_per_sec

    def decode_numpy(self, speech_tokens: np.ndarray) -> np.ndarray:
        """Decode speech tokens to audio waveform (NumPy interface).

        Converts between NumPy arrays (used by AudioDecoder) and TensorValues (used by XCodec2Model).

        Args:
            speech_tokens: NumPy array of speech token indices, shape (num_tokens,) or (batch, num_tokens)

        Returns:
            Audio waveform as NumPy array, shape (num_samples,) or (batch, num_samples)
        """
        # TODO: For now, return dummy audio matching expected shape
        # In a real implementation, this would need to execute the graph
        # and convert the results back to numpy

        num_tokens = speech_tokens.shape[
            -1
        ]  # Last dimension is sequence length
        duration_seconds = num_tokens * (
            1.0 / self.config.codec_tokens_per_sec
        )  # 50 tokens/sec
        num_samples = int(duration_seconds * self.config.sample_rate)  # 16kHz

        if speech_tokens.ndim == 1:
            dummy_audio = np.zeros(num_samples, dtype=np.float32)
        else:
            batch_size = speech_tokens.shape[0]
            dummy_audio = np.zeros((batch_size, num_samples), dtype=np.float32)

        return dummy_audio
