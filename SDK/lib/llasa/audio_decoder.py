# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""XCodec2 wrapper for Llasa TTS."""

from __future__ import annotations

import numpy as np

from .config import LlasaConfig
from .xcodec2.decoder import XCodec2Decoder


class AudioDecoder:
    """Audio decoder for Llasa TTS using XCodec2.

    Converts speech token sequences to audio waveforms using the XCodec2 model.
    """

    def __init__(self, config: LlasaConfig) -> None:
        """Initialize the audio decoder.

        Args:
            config: Llasa configuration
        """
        self.config = config
        self.xcodec2_decoder = XCodec2Decoder(config.xcodec2_config)

    @property
    def sample_rate(self) -> int:
        """Audio sample rate from XCodec2 config."""
        return self.xcodec2_decoder.sample_rate

    @property
    def codec_tokens_per_sec(self) -> int:
        """Number of codec tokens per second from XCodec2 config."""
        return self.xcodec2_decoder.codec_tokens_per_sec

    @property
    def duration_per_token(self) -> float:
        """Duration per token in seconds (1/50 = 0.02 seconds)."""
        return 1.0 / self.codec_tokens_per_sec

    def decode(self, speech_tokens: np.ndarray) -> np.ndarray:
        """Decode speech tokens to audio waveform using XCodec2.

        Args:
            speech_tokens: Array of speech token indices

        Returns:
            Audio waveform as numpy array (shape: [num_samples])
        """
        return self.xcodec2_decoder.decode_numpy(speech_tokens)
