# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""XCodec2 wrapper for Llasa TTS (stub implementation)."""

from __future__ import annotations

import numpy as np

from .config import AudioType, LlasaConfig


class AudioDecoder:
    """Audio decoder for Llasa TTS.

    This is a temporary implementation that generates white noise or sine waves
    instead of an actual audio decoding. It will be replaced with the real
    XCodec2 model in a future PR.
    """

    def __init__(self, config: LlasaConfig) -> None:
        """Initialize the audio decoder.

        Args:
            config: Llasa configuration
        """
        self.config = config
        self.sample_rate = 16000  # From `modeling_xcodec2.py` reference on HF
        self.codec_tokens_per_sec = 50  # From HF model card
        self.duration_per_token = (
            1.0 / self.codec_tokens_per_sec
        )  # 0.02 seconds per token

    def decode(
        self,
        speech_tokens: np.ndarray,
        output_type: AudioType = AudioType.WHITE_NOISE,
    ) -> np.ndarray:
        """Decode speech tokens to audio waveform (stub implementation).

        Args:
            speech_tokens: Array of speech token indices
            output_type: Type of stub audio to generate

        Returns:
            Audio waveform as numpy array (shape: [num_samples])
        """
        num_tokens = len(speech_tokens)
        duration_seconds = num_tokens * self.duration_per_token
        num_samples = int(duration_seconds * self.sample_rate)

        if output_type == AudioType.SINE_WAVE:
            # Generate a simple sine wave (440 Hz - A4 note)
            t = np.linspace(0, duration_seconds, num_samples, False)
            frequency = 440.0  # Hz
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        else:
            # Generate white noise (default)
            audio = 0.1 * np.random.normal(0, 1, num_samples)

        return audio.astype(np.float32)
