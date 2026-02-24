# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Speech language model wrapper for Llasa TTS."""

from __future__ import annotations

import re

import numpy as np
from max import driver, pipelines
from max.interfaces import (
    PipelineTask,
    RequestID,
    SamplingParams,
    TextGenerationInputs,
    TextGenerationRequest,
)

from .config import LlasaConfig
from .prompt_template import create_llasa_prompt


class SpeechLM:
    """Wrapper for Llama model to generate speech tokens."""

    def __init__(
        self,
        config: LlasaConfig,
        device_specs: list[driver.DeviceSpec] | None = None,
        encoding: pipelines.SupportedEncoding = "bfloat16",
    ) -> None:
        """Initialize the speech language model.

        Args:
            config: Llasa configuration
            device_specs: Device specifications for MAX pipeline
            encoding: Model encoding (bfloat16, float32, etc.)
        """
        self.config = config
        self.encoding = encoding

        if device_specs is None:
            device_specs = [driver.DeviceSpec(id=0, device_type="gpu")]

        pipeline_config = pipelines.PipelineConfig(
            model=pipelines.MAXModelConfig(
                device_specs=device_specs,
                quantization_encoding=encoding,
                model_path=config.llama_model_path,
                trust_remote_code=True,
                max_length=config.max_length,
            ),
        )

        self.tokenizer, self.pipeline = pipelines.PIPELINE_REGISTRY.retrieve(
            pipeline_config, task=PipelineTask.TEXT_GENERATION
        )

        assert isinstance(
            self.pipeline, pipelines.TextGenerationPipelineInterface
        )
        self.text_pipeline = self.pipeline  # Type hint for linter

        # Get the KV cache manager for resource management
        self.kv_cache_manager = self.text_pipeline.kv_manager

    async def generate_speech_tokens(
        self, text: str, max_tokens: int | None = None
    ) -> np.ndarray:
        """Generate speech tokens from input text.

        Args:
            text: Input text to convert to speech
            max_tokens: Maximum number of tokens to generate

        Returns:
            numpy array of speech token indices
        """
        generated_text = await self._generate_raw_text(
            text, max_tokens, RequestID("llasa_speech_generation")
        )
        speech_tokens = self._extract_speech_tokens(generated_text)
        return np.array(speech_tokens, dtype=np.int32)

    def _extract_speech_tokens(self, generated_text: str) -> list[int]:
        """Extract speech token indices from generated text.

        Args:
            generated_text: Generated text containing speech tokens

        Returns:
            List of speech token indices
        """
        # Find all speech tokens using the pattern from the config
        # Given `<|s_12345|>`, we want to extract `12345`
        pattern = re.escape(self.config.speech_token_pattern).replace(
            re.escape("{}"), r"(\d+)"
        )
        matches = re.findall(pattern, generated_text)

        # Convert matched strings to integers, filtering out empty matches
        speech_tokens = [int(match) for match in matches if match.strip()]

        return speech_tokens

    async def _generate_raw_text(
        self,
        text: str,
        max_tokens: int | None = None,
        request_id: RequestID = RequestID("llasa_generation"),
    ) -> str:
        """Generate raw text from the model.

        Args:
            text: Input text to convert to speech
            max_tokens: Maximum number of tokens to generate
            request_id: Unique identifier for the request

        Returns:
            Generated text output
        """
        if max_tokens is None:
            max_tokens = self.config.max_new_tokens

        input_ids, hf_tokenizer = create_llasa_prompt(text, self.config)
        input_token_list = input_ids.tolist()

        stop_token_id = hf_tokenizer.convert_tokens_to_ids(
            self.config.speech_generation_end
        )

        sampling_params = SamplingParams(
            max_new_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            ignore_eos=False,  # Allow EOS token to stop generation
            stop_token_ids=[stop_token_id],
        )

        request = TextGenerationRequest(
            request_id=request_id,
            model_name=self.config.llama_model_path,
            prompt=input_token_list,
            sampling_params=sampling_params,
        )

        context = await self.tokenizer.new_context(request)

        # Claim the request ID in the KV cache manager
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.claim(context.request_id, replica_idx=0)

        all_tokens = []
        batch = [context]

        while not context.status.is_done:
            # Allocate KV cache blocks for this step
            if self.kv_cache_manager is not None:
                self.kv_cache_manager.alloc(context, replica_idx=0, num_steps=1)

            inputs: TextGenerationInputs = TextGenerationInputs(
                batches=[batch], num_steps=1
            )
            output = self.text_pipeline.execute(inputs)
            response = output[request_id]
            all_tokens.extend(response.tokens)
            if response.is_done:
                break

        generated_text = await self.tokenizer.decode(all_tokens)
        return generated_text

    async def generate_text_for_debugging(
        self, text: str, max_tokens: int | None = None
    ) -> str:
        """Generate raw text output for debugging purposes.

        Args:
            text: Input text to convert to speech
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text output
        """
        return await self._generate_raw_text(
            text, max_tokens, RequestID("llasa_debug_generation")
        )
