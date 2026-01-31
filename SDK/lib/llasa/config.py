# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass, field

from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from typing_extensions import Self

from .xcodec2.config import XCodec2Config


@dataclass(frozen=True)
class LlasaConfig(ArchConfig):
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
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_p: float = 1.0

    # XCodec2 configuration
    xcodec2_config: XCodec2Config = field(default_factory=XCodec2Config)

    @classmethod
    def initialize(cls, pipeline_config: PipelineConfig) -> Self:
        return cls()

    def get_max_seq_len(self) -> int:
        return self.max_length
