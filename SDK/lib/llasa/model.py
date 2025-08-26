# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Llasa TTS pipeline orchestration."""

from __future__ import annotations

import numpy as np
from max import driver

from .audio_decoder import AudioDecoder
from .config import LlasaConfig
from .speech_lm import SpeechLM


class LlasaTTS:
    """Llasa Text-to-Speech pipeline.

    Orchestrates the two-stage TTS process:
    1. Text → Speech tokens (via SpeechLM/Llama)
    2. Speech tokens → Audio (via XCodec2)
    """

    def __init__(
        self,
        config: LlasaConfig,
        device_specs: list[driver.DeviceSpec] | None = None,
        encoding: str = "bfloat16",
    ) -> None:
        """Initialize the Llasa TTS pipeline.

        Args:
            config: Llasa configuration
            device_specs: Device specifications for MAX pipeline
            encoding: Model encoding (bfloat16, float32, etc.)
        """
        self.config = config
        self.speech_lm = SpeechLM(config, device_specs, encoding)
        self.audio_decoder = AudioDecoder(config)

    async def generate_audio(
        self,
        text: str,
        max_tokens: int | None = None,
    ) -> np.ndarray:
        """Generate audio from input text.

        Args:
            text: Input text to convert to speech
            max_tokens: Maximum number of speech tokens to generate

        Returns:
            Audio waveform as numpy array
        """
        speech_tokens = await self.speech_lm.generate_speech_tokens(
            text, max_tokens
        )
        audio = self.audio_decoder.decode(speech_tokens)
        return audio

    def save_audio(self, audio: np.ndarray, output_path: str) -> None:
        """Save audio to file.

        Args:
            audio: Audio waveform
            output_path: Path to save the audio file
        """
        try:
            import soundfile as sf

            sf.write(output_path, audio, self.audio_decoder.sample_rate)
        except ImportError:
            # Fallback: save as numpy array if soundfile not available
            import numpy as np

            np.save(output_path.replace(".wav", ".npy"), audio)
            print(
                f"Warning: soundfile not available, saved as {output_path.replace('.wav', '.npy')}"
            )

    async def generate_and_save(
        self,
        text: str,
        output_path: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate audio from text and save to file.

        Args:
            text: Input text to convert to speech
            output_path: Path to save the generated audio
            max_tokens: Maximum number of speech tokens to generate

        Returns:
            Path to the saved audio file
        """
        audio = await self.generate_audio(text, max_tokens)
        self.save_audio(audio, output_path)
        return output_path
