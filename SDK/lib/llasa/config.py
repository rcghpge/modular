# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass


@dataclass(frozen=True)
class LlasaConfig:
    """Configuration for the Llasa TTS model."""

    llama_model_path: str = "HKUSTAudio/Llasa-8B"
    xcodec2_model_path: str = "HKUSTAudio/xcodec2"

    output_file: str = "generated_audio.wav"

    # Speech token format (from HF model card)
    speech_token_pattern: str = "<|s_{}|>"
    text_understanding_start: str = "<|TEXT_UNDERSTANDING_START|>"
    text_understanding_end: str = "<|TEXT_UNDERSTANDING_END|>"
    speech_generation_start: str = "<|SPEECH_GENERATION_START|>"
    speech_generation_end: str = "<|SPEECH_GENERATION_END|>"

    # Generation parameters (from HF model card)
    max_length: int = 2048
    max_new_tokens: int = 200  # Shorter to see natural patterns
    temperature: float = 1.0
    top_p: float = 1.0
