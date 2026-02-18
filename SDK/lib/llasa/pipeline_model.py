# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Llasa TTS pipeline model for MAX pipelines."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from max.driver import Buffer, Device
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.interfaces import AudioGenerationOutput, GenerationStatus, PipelineTask
from max.interfaces.request import RequestID
from max.nn.legacy import ReturnLogits
from max.nn.legacy.kv_cache import KVCacheInputs
from max.pipelines import (
    PIPELINE_REGISTRY,
    AudioGenerationConfig,
    PipelineConfig,
    SpeechTokenGenerationPipeline,
    TextTokenizer,
    TTSContext,
)
from max.pipelines.lib import (
    KVCacheConfig,
    LoRAConfig,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    SupportedEncoding,
)
from max.profiler import Tracer, traced
from transformers.models.auto.configuration_auto import AutoConfig

from .config import LlasaConfig
from .decoder import load_xcodec2_decoder

logger = logging.getLogger(__name__)


class SpeechTokenExtractor:
    """Extracts speech token indices from generated text using regex pattern matching.

    Llasa generates text containing speech tokens in the format `<|s_12345|>`,
    which need to be extracted and converted to integer indices for the audio decoder.
    """

    def __init__(self, config: LlasaConfig) -> None:
        """Initialize the speech token extractor.

        Args:
            config: Llasa configuration containing the speech token pattern
        """
        self.config = config
        self.pattern = re.escape(config.speech_token_pattern).replace(
            re.escape("{}"), r"(\d+)"
        )

    def __call__(self, generated_text: str) -> np.ndarray:
        """Extract speech token indices from generated text.

        Args:
            generated_text: Generated text containing speech tokens in format `<|s_12345|>`

        Returns:
            NumPy array of speech token indices as int32
        """
        matches = re.findall(self.pattern, generated_text)

        speech_tokens = [int(match) for match in matches if match.strip()]

        return np.array(speech_tokens, dtype=np.int32)


class LlasaPipelineModel(PipelineModel[TTSContext]):
    """Pipeline model for Llasa text-to-speech generation."""

    pipeline_config: AudioGenerationConfig
    speech_tokenizer: TextTokenizer
    speech_lm_pipeline: SpeechTokenGenerationPipeline
    session: InferenceSession
    devices: list[Device]
    audio_decoder: Model
    speech_token_extractor: SpeechTokenExtractor
    llasa_config: LlasaConfig
    _prev_num_steps: int

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
    ) -> None:
        """Initialize the Llasa pipeline model.

        Args:
            pipeline_config: Pipeline configuration
            session: Inference session
            huggingface_config: HuggingFace model configuration
            encoding: Model encoding
            devices: List of devices
            kv_cache_config: KV cache configuration
            weights: Model weights
            adapter: Optional weights adapter
            return_logits: Whether to return logits
        """
        # Handle LoRA config (similar to TTSPipelineModel)
        lora_config: LoRAConfig | None = None
        if pipeline_config.lora:
            lora_config = pipeline_config.lora
            pipeline_config.lora = None

        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        assert isinstance(pipeline_config, AudioGenerationConfig)
        self.pipeline_config = pipeline_config

        llasa_config_dict = pipeline_config.audio_decoder_config
        self.llasa_config = LlasaConfig(**llasa_config_dict)

        # Re-initialize LoRA config if needed
        if lora_config:
            pipeline_config.lora = lora_config

        speech_tokenizer, speech_lm_pipeline = PIPELINE_REGISTRY.retrieve(
            pipeline_config, task=PipelineTask.SPEECH_TOKEN_GENERATION
        )
        assert isinstance(speech_lm_pipeline, SpeechTokenGenerationPipeline)
        assert isinstance(speech_tokenizer, TextTokenizer)
        self.speech_tokenizer = speech_tokenizer
        self.speech_lm_pipeline = speech_lm_pipeline
        self.kv_manager = self.speech_lm_pipeline._pipeline_model.kv_manager

        self.session = self.speech_lm_pipeline.session
        self.devices = devices

        audio_decoder_weights: Path | str | None = None
        if pipeline_config.audio_decoder_weights:
            audio_decoder_weights = pipeline_config.audio_decoder_weights
        else:
            assert self.pipeline_config._run_model_test_mode, (
                "Must provide audio_decoder_weights when running the TTS model."
            )

        xcodec2_config = self.llasa_config.xcodec2_config
        self.audio_decoder = load_xcodec2_decoder(
            self.devices,
            self.session,
            audio_decoder_weights,
            xcodec2_config,
        )

        self.speech_token_extractor = SpeechTokenExtractor(self.llasa_config)

        self.speech_end_id = (
            self.speech_tokenizer.delegate.convert_tokens_to_ids(
                self.llasa_config.speech_generation_end
            )
        )
        if self.speech_end_id is None:
            raise ValueError(
                f"Speech generation end token '{self.llasa_config.speech_generation_end}' "
                f"not found in tokenizer vocabulary."
            )
        self.speech_lm_pipeline._eos_token_id.add(self.speech_end_id)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        """Calculate the maximum sequence length for the pipeline model."""
        assert isinstance(pipeline_config, AudioGenerationConfig)
        llasa_config = LlasaConfig(**pipeline_config.audio_decoder_config)
        if pipeline_config.model.max_length:
            return min(
                llasa_config.max_length, pipeline_config.model.max_length
            )
        else:
            return llasa_config.max_length

    @classmethod
    def estimate_weights_size(cls, pipeline_config: PipelineConfig) -> int:
        """Estimate the size of the weights for computing total memory usage."""
        assert isinstance(pipeline_config, AudioGenerationConfig)
        speech_lm_size = pipeline_config.model.weights_size()
        if pipeline_config.audio_decoder_weights:
            audio_decoder_size = (
                Path(pipeline_config.audio_decoder_weights).stat().st_size
                if Path(pipeline_config.audio_decoder_weights).exists()
                else 0
            )
        else:
            llasa_config = LlasaConfig(**pipeline_config.audio_decoder_config)
            xcodec2_config = llasa_config.xcodec2_config
            # Rough estimate: hidden_dim * depth * pos_emb_dim * 4 bytes
            audio_decoder_size = (
                xcodec2_config.hidden_dim
                * xcodec2_config.depth
                * xcodec2_config.pos_emb_dim
                * 4
            )
        return speech_lm_size + audio_decoder_size

    def generate_speech_tokens(
        self,
        batch: dict[RequestID, TTSContext],
        tokens_to_generate: dict[RequestID, int],
    ) -> dict[RequestID, int]:
        """Generate the next chunk of speech tokens.

        The context is updated with the new speech tokens, and the
        `status` field is updated as well.

        Args:
            batch: A dictionary mapping request IDs to `TTSContext` objects.
            tokens_to_generate: A dictionary mapping request IDs to the number
                of tokens to generate for each request.

        Returns:
            Dictionary mapping request IDs to the number of speech tokens generated.
        """
        tracer: Tracer = Tracer("generate_speech_tokens")

        speech_lm_requests: dict[RequestID, TTSContext] = {}
        for batch_id, context in batch.items():
            if context.status != GenerationStatus.ACTIVE:
                continue
            speech_lm_requests[batch_id] = context

        if not speech_lm_requests:
            return {}

        current_lengths = {
            req_id: len(context.tokens)
            for req_id, context in speech_lm_requests.items()
        }

        max_length = self.speech_lm_pipeline._pipeline_config.model.max_length
        assert max_length
        num_tokens = max(tokens_to_generate.values())
        if current_lengths:
            num_tokens = min(
                num_tokens, max_length - max(current_lengths.values())
            )
        else:
            num_tokens = min(num_tokens, max_length)

        self._prev_num_steps = num_tokens

        speech_lm_responses = self.speech_lm_pipeline.next_speech_token(
            batch=speech_lm_requests,
            num_steps=num_tokens,
            tokens_to_generate=tokens_to_generate,
        )

        generated_token_counts = {}
        tracer.push("extract_speech_tokens")

        for request_id, context in speech_lm_requests.items():
            if request_id not in speech_lm_responses:
                continue

            token_index = current_lengths[request_id]
            generated_ids = context.tokens.all[token_index:]

            if len(generated_ids) > 0:
                generated_text = self.speech_tokenizer.delegate.decode(
                    generated_ids, skip_special_tokens=False
                )

                speech_tokens = self.speech_token_extractor(generated_text)

                if len(speech_tokens) > 0:
                    generated_token_counts[request_id] = speech_tokens.shape[0]
                    context.update_speech_tokens(speech_tokens)
                else:
                    generated_token_counts[request_id] = 0
            else:
                generated_token_counts[request_id] = 0

            # Update status based on speech LM pipeline's EOS detection
            # The pipeline already handles EOS detection via context.update() which
            # sets status to END_OF_SEQUENCE when EOS tokens are detected
            text_output = speech_lm_responses[request_id]
            context.status = (
                GenerationStatus.END_OF_SEQUENCE
                if text_output.is_done
                else GenerationStatus.ACTIVE
            )

        tracer.pop()
        return generated_token_counts

    @traced
    def next_chunk(
        self, batch: dict[RequestID, TTSContext]
    ) -> dict[RequestID, AudioGenerationOutput]:
        """Process the next chunk of audio generation.

        Args:
            batch: Dictionary mapping request IDs to TTSContext objects

        Returns:
            Dictionary mapping request IDs to AudioGenerationOutput
        """
        tracer: Tracer = Tracer("next_chunk")

        res: dict[RequestID, AudioGenerationOutput] = {}

        first_req = next(iter(batch.values()))
        needs_ce = first_req.tokens.generated_length == 0
        for context in batch.values():
            if (context.tokens.generated_length == 0) != needs_ce:
                raise ValueError(
                    "Mixing CE and TG requests in the same batch is not supported"
                )

        if needs_ce:
            tracer.push("handle_ce_requests")
            tokens_to_generate = {req_id: 1 for req_id in batch}
            generated_token_counts = self.generate_speech_tokens(
                batch, tokens_to_generate
            )
            for request_id, context in batch.items():
                res[request_id] = AudioGenerationOutput(
                    final_status=context.status,
                    steps_executed=generated_token_counts.get(request_id, 0),
                )
            tracer.pop()
            return res

        # Track how many tokens need to be generated for each request
        tokens_to_generate = {}
        for batch_id, context in batch.items():
            if self.pipeline_config.chunk_size:
                block_index = min(
                    context.block_counter,
                    len(self.pipeline_config.chunk_size) - 1,
                )
                tokens_to_generate[batch_id] = self.pipeline_config.chunk_size[
                    block_index
                ]
            else:
                tokens_to_generate[batch_id] = (
                    self.pipeline_config.max_num_steps or 100
                )

        generated_token_counts = self.generate_speech_tokens(
            batch, tokens_to_generate
        )

        tracer.push("decode_audio")
        decoder_inputs = {}
        decoded_tokens = {}
        max_speech_token_length = 0

        for request_id, context in batch.items():
            if (
                context.status != GenerationStatus.ACTIVE
                and not context.status.is_done
            ):
                res[request_id] = AudioGenerationOutput(
                    final_status=context.status,
                    steps_executed=generated_token_counts.get(request_id, 0),
                )
                continue

            if context.status.is_done:
                speech_tokens, _ = context.next_speech_tokens(buffer=0)
            else:
                speech_tokens, _ = context.next_speech_tokens(
                    audio_chunk_size=tokens_to_generate[request_id], buffer=0
                )

            if len(speech_tokens) == 0:
                res[request_id] = AudioGenerationOutput(
                    final_status=context.status,
                    steps_executed=generated_token_counts.get(request_id, 0),
                )
                continue

            decoded_tokens[request_id] = len(speech_tokens)
            decoder_inputs[request_id] = speech_tokens
            max_speech_token_length = max(
                max_speech_token_length, len(speech_tokens)
            )

        if len(decoder_inputs) > 0:
            padded_inputs = np.full(
                (len(decoder_inputs), max_speech_token_length),
                fill_value=0,  # Use 0 as padding for XCodec2
                dtype=np.int32,
            )
            for i, speech_tokens in enumerate(decoder_inputs.values()):
                padded_inputs[i, : len(speech_tokens)] = speech_tokens

            input_buffer = Buffer.from_numpy(padded_inputs).to(self.devices[0])
            audio_outputs = self.audio_decoder(input_buffer)[0]
            assert isinstance(audio_outputs, Buffer)
            audio_np = audio_outputs.to_numpy()

            # Calculate samples per token from XCodec2 config for audio trimming
            samples_per_token = self.llasa_config.xcodec2_config.hop_length

            for i, (request_id, _) in enumerate(decoder_inputs.items()):
                context = batch[request_id]
                status = context.status

                if status.is_done:
                    context.audio_generation_status = status

                # XCodec2 output shape is (batch, audio_length)
                # Trim audio to actual token count to remove padding artifacts
                # Token 0 (used for padding) produces actual audio, not silence
                actual_token_count = decoded_tokens[request_id]
                actual_audio_length = actual_token_count * samples_per_token
                audio_data = audio_np[i : i + 1, :actual_audio_length]

                context.decoded_index = (
                    context.decoded_index + decoded_tokens[request_id]
                )

                res[request_id] = AudioGenerationOutput(
                    final_status=status,
                    audio_data=audio_data,
                    steps_executed=generated_token_counts.get(request_id, 0),
                )
        else:
            for request_id, context in batch.items():
                res[request_id] = AudioGenerationOutput(
                    final_status=context.status,
                    steps_executed=generated_token_counts.get(request_id, 0),
                )

        tracer.pop()
        return res

    def release(self, request_id: RequestID) -> None:
        """Release all resources used by the context."""
        # Only release from the text generation pipeline's KV cache manager
        # if we've actually claimed/allocated blocks for it (which happens in generate_speech_tokens)
        # The scheduler manages its own KV cache manager separately
        if self.kv_manager.contains(request_id, replica_idx=0):
            self.kv_manager.release(request_id, replica_idx=0)

    # PipelineModel expects these methods but they don't fit with audio generation
    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        raise NotImplementedError(
            "Audio generation pipeline does not use execute"
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TTSContext]],
        kv_cache_inputs: KVCacheInputs | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        raise NotImplementedError(
            "Audio generation pipeline does not use prepare_initial_token_inputs"
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> ModelInputs:
        raise NotImplementedError(
            "Audio generation pipeline does not use prepare_next_token_inputs"
        )
