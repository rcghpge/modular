# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tokenizer for Llasa TTS pipeline."""

from __future__ import annotations

import numpy as np
from max.interfaces import (
    AudioGenerationRequest,
    PipelineTokenizer,
    TextGenerationRequest,
)
from max.pipelines import AudioGenerationConfig, PipelineConfig, TextTokenizer
from max.pipelines.core import TTSContext
from transformers import AutoTokenizer

from .config import LlasaConfig
from .prompt_template import create_llasa_prompt


class LlasaTokenizer(
    PipelineTokenizer[TTSContext, np.ndarray, AudioGenerationRequest]
):
    """Tokenizer for the Llasa TTS model.

    Wraps TextTokenizer and handles Llasa-specific prompt formatting.
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        text_tokenizer: TextTokenizer | None = None,
        model_path: str | None = None,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        chat_template: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Llasa tokenizer.

        Args:
            pipeline_config: Pipeline configuration
            text_tokenizer: Optional pre-initialized text tokenizer
            model_path: Path to the model (if text_tokenizer not provided)
            revision: Model revision
            max_length: Maximum sequence length
            max_new_tokens: Maximum new tokens to generate
            trust_remote_code: Whether to trust remote code
            enable_llama_whitespace_fix: Whether to enable Llama whitespace fix
            chat_template: Optional chat template
            **kwargs: Additional keyword arguments
        """
        if pipeline_config is None:
            raise ValueError("pipeline_config is required")
        assert isinstance(pipeline_config, AudioGenerationConfig)

        llasa_config_dict = pipeline_config.audio_decoder_config
        self.llasa_config = LlasaConfig(**llasa_config_dict)

        if text_tokenizer is None:
            assert model_path is not None
            self.text_tokenizer = TextTokenizer(
                model_path=model_path,
                pipeline_config=pipeline_config,
                revision=revision,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                trust_remote_code=trust_remote_code,
                enable_llama_whitespace_fix=enable_llama_whitespace_fix,
                chat_template=chat_template,
                **kwargs,
            )
        else:
            self.text_tokenizer = text_tokenizer

        # Get the speech generation end token ID for stopping generation
        speech_end_id = self.text_tokenizer.delegate.convert_tokens_to_ids(
            self.llasa_config.speech_generation_end
        )
        if speech_end_id is None:
            raise ValueError(
                f"Speech generation end token '{self.llasa_config.speech_generation_end}' "
                f"not found in tokenizer vocabulary. This token is required for proper "
                f"generation stopping behavior."
            )
        self.text_tokenizer._default_eos_token_ids.add(speech_end_id)

    @property
    def eos(self) -> int:
        """The end of sequence token for this tokenizer."""
        return self.text_tokenizer.eos

    @property
    def delegate(self) -> AutoTokenizer:
        """Get the underlying HuggingFace tokenizer."""
        return self.text_tokenizer.delegate

    @property
    def expects_content_wrapping(self) -> bool:
        """Whether the tokenizer expects content wrapping."""
        return self.text_tokenizer.expects_content_wrapping

    async def new_context(self, request: AudioGenerationRequest) -> TTSContext:
        """Creates a new context from an AudioGenerationRequest.

        Args:
            request: Incoming audio generation request

        Returns:
            Initialized TTSContext
        """
        if isinstance(request.prompt, str):
            input_ids, _ = create_llasa_prompt(
                request.prompt, self.llasa_config
            )
            prompt_ids = input_ids.tolist()
        elif isinstance(request.prompt, list):
            prompt_ids = request.prompt
        elif request.input is not None:
            input_ids, _ = create_llasa_prompt(request.input, self.llasa_config)
            prompt_ids = input_ids.tolist()
        else:
            raise ValueError(
                "either input or prompt must be provided in AudioGenerationRequest"
            )

        text_request = TextGenerationRequest(
            request_id=request.request_id,
            model_name=request.model,
            prompt=prompt_ids,
            sampling_params=request.sampling_params,
        )

        text_context = await self.text_tokenizer.new_context(text_request)

        ctx = TTSContext(
            model_name=request.model,
            request_id=request.request_id,
            max_length=text_context.max_length,
            tokens=text_context.tokens,
            json_schema=text_context.json_schema,
            eos_token_ids=text_context.eos_token_ids,
            eos_sequences=text_context.eos_sequences,
            audio_prompt_tokens=np.array(
                request.audio_prompt_tokens, dtype=np.int32
            ),
            sampling_params=request.sampling_params,
            streaming=request.streaming,
        )
        return ctx

    async def encode(
        self, prompt: str, add_special_tokens: bool = True
    ) -> np.ndarray:
        """Encodes text prompts as tokens.

        Args:
            prompt: Un-encoded prompt text
            add_special_tokens: Whether to add special tokens

        Returns:
            Encoded tokens as numpy array
        """
        return await self.text_tokenizer.encode(prompt, add_special_tokens)

    async def decode(self, encoded: np.ndarray, **kwargs) -> str:
        """Decodes response tokens to text.

        Args:
            encoded: Encoded response tokens
            **kwargs: Additional keyword arguments

        Returns:
            Un-encoded response text
        """
        return await self.text_tokenizer.decode(encoded, **kwargs)
