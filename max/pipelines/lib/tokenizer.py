# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# mypy: disable-error-code="import-not-found"
"""Implementations of provided tokenizers."""

from __future__ import annotations

import asyncio
import functools
import io
import json
import logging
from collections.abc import Sequence
from typing import Any, Optional, TypeVar, Union, cast

import numpy as np
from max.interfaces import (
    PipelineTokenizer,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CodeLlamaTokenizer,
    CodeLlamaTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger("max.pipelines")

TokenGeneratorContext = TypeVar("TokenGeneratorContext")


class IdentityPipelineTokenizer(
    PipelineTokenizer[TokenGeneratorContext, str, TextGenerationRequest],
):
    @property
    def eos(self) -> int:
        return 0

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> str:
        return prompt

    async def decode(
        self,
        encoded: str,
        **kwargs,
    ) -> str:
        if isinstance(encoded, str):
            return encoded
        return ""


class PreTrainedPipelineTokenizer(
    PipelineTokenizer[TokenGeneratorContext, np.ndarray, TextGenerationRequest],
):
    def __init__(
        self,
        delegate: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        assert isinstance(
            delegate, (PreTrainedTokenizer, PreTrainedTokenizerFast)
        )
        self.delegate = delegate

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        try:
            templated_message = self.delegate.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return cast(str, templated_message)
        except Exception:
            msg = (
                "apply_chat_template failed for"
                " PreTrainedTokenGeneratorTokenizer"
            )
            logger.warning(msg)
            return "\n".join([str(message["content"]) for message in messages])

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        return np.array(self.delegate.encode(prompt))

    async def decode(
        self,
        encoded: np.ndarray,
        **kwargs,
    ) -> str:
        return self.delegate.decode(encoded, **kwargs)


def max_tokens_to_generate(
    prompt_size: int,
    max_length: int | None,
    max_new_tokens: int | None = None,
) -> int | None:
    """Returns the max number of new tokens to generate."""
    if max_length is None:
        return max_new_tokens
    _difference_between_max_and_prompt = max(max_length - prompt_size, 0)
    if max_new_tokens is None:
        return _difference_between_max_and_prompt
    return min(max_new_tokens, _difference_between_max_and_prompt)


async def run_with_default_executor(fn, *args):  # noqa: ANN001
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


class TextTokenizer(
    PipelineTokenizer[TextContext, np.ndarray, TextGenerationRequest]
):
    """Encapsulates creation of TextContext and specific token encode/decode logic."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        try:
            self.delegate = AutoTokenizer.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                # If `max_length` is None, the max length will be taken
                # from the HuggingFace tokenizer_config.
                model_max_length=max_length,
            )
        except Exception as e:
            msg = (
                f"Failed to load tokenizer from {model_path}. "
                "This can happen if:\n"
                "- The model is not fully supported by the transformers python package\n"
                "- Required configuration files are missing\n"
                "- The model path is incorrect\n"
                "- '--trust_remote_code=True' is needed but not set\n"
            )
            raise ValueError(msg) from e

        # As we are adding special tokens during chat templating prior to tokenization,
        # when add_special_tokens=True, we duplicate BOS tokens specifically.
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )
        self.max_length = max_length or self.delegate.model_max_length

        # configure Llama whitespace fix if needed
        self._enable_llama_whitespace_fix = (
            enable_llama_whitespace_fix and self._is_llama_tokenizer
        )
        (
            self._llama_whitespace_fix_dummy_token_id,
            self._llama_whitespace_fix_dummy_token_len,
        ) = self._llama_whitespace_fix_dummy_token

        # cache tokenizer eos token ids
        self._default_eos_token_ids = set([self.eos])

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: Optional[list[TextGenerationRequestTool]],
        chat_template_options: Optional[dict[str, Any]] = None,
    ) -> str:
        chat_template_options = chat_template_options or {
            "add_generation_prompt": True
        }
        try:
            templated_message = self.delegate.apply_chat_template(
                messages, tokenize=False, tools=tools, **chat_template_options
            )
            return cast(str, templated_message)
        except Exception:
            msg = (
                "apply_chat_template failed for"
                f" TextTokenizer({self.model_path})"
            )
            logger.warning(msg)
            return "\n".join([str(message["content"]) for message in messages])

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    async def encode(
        self, prompt: Union[str, Sequence[int]], add_special_tokens: bool = True
    ) -> np.ndarray:
        """Transform the provided prompt into a token array."""

        encoded_prompt: np.ndarray
        if isinstance(prompt, str):
            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            if add_special_tokens:
                encoded_prompt = await run_with_default_executor(
                    self._encode_with_special_tokens, prompt
                )
            else:
                encoded_prompt = await run_with_default_executor(
                    self._encode_without_special_tokens, prompt
                )

            if self.max_length and len(encoded_prompt) > self.max_length:
                raise ValueError(
                    f"Input string is larger than tokenizer's max length ({len(encoded_prompt)} > {self.max_length})."
                )

            encoded_prompt = np.array(encoded_prompt)
        else:
            encoded_prompt = np.array(list(prompt))

        return encoded_prompt

    async def decode(self, encoded: np.ndarray, **kwargs) -> str:
        """Transformer a provided encoded token array, back into readable text."""
        # Sometimes, encoded comes in as an int so, make it np array
        if isinstance(encoded, int):
            encoded = np.array(encoded)

        # There is an issue where Llama tokenizer strips leading spaces
        # if a single token is decoded at a time. This is a temporary
        # fix until the issue resolved on the Tokenizers side.
        # More information:
        # https://github.com/huggingface/transformers/issues/31643
        # https://github.com/Lightning-AI/litgpt/pull/1559
        if self._enable_llama_whitespace_fix and encoded.size == 1:
            return self._decode_with_llama_whitespace_fix(encoded, **kwargs)

        return self.delegate.decode(encoded, **kwargs)

    async def _generate_prompt_and_token_ids(
        self,
        prompt: Optional[Union[Sequence[int], str]],
        messages: Optional[list[TextGenerationRequestMessage]],
        tools: Optional[list[TextGenerationRequestTool]] = None,
        chat_template_options: Optional[dict[str, Any]] = None,
    ) -> tuple[Union[str, list[int]], np.ndarray]:
        if prompt and messages:
            raise ValueError("both prompt and messages cannot be provided.")

        if isinstance(prompt, str):
            return prompt, await self.encode(prompt, add_special_tokens=True)
        elif isinstance(prompt, list):
            return prompt, await self.encode(prompt, add_special_tokens=True)
        elif isinstance(messages, list):
            prompt = self.apply_chat_template(
                messages, tools, chat_template_options
            )
            return prompt, await self.encode(prompt, add_special_tokens=False)
        else:
            raise ValueError(
                "either prompt must be provided as a list[int] or str, or messages must be provided as a list[TextGenerationRequestMessage]"
            )

    async def _get_eos_variables(
        self,
        ignore_eos: bool,
        stop_token_ids: Optional[list[int]],
        stop: Optional[list[str]],
    ) -> tuple[set[int], list[list[int]]]:
        eos_token_ids = self._default_eos_token_ids
        eos_sequences = list()

        if ignore_eos:
            eos_token_ids = set()
        elif stop_token_ids:
            eos_token_ids.update(stop_token_ids)
        elif stop:
            eos_sequences = await self._encode_stop_criteria(stop)

        return eos_token_ids, eos_sequences

    async def new_context(self, request: TextGenerationRequest) -> TextContext:
        """Create a new TextContext object, leveraging necessary information like
        cache_seq_id and prompt from TextGenerationRequest."""
        # Encode Prompt / Messages
        prompt, token_ids = await self._generate_prompt_and_token_ids(
            prompt=request.prompt,
            messages=request.messages,
            tools=request.tools,
            chat_template_options=request.chat_template_options,
        )

        json_schema = (
            json.dumps(request.response_format.get("json_schema", None))
            if request.response_format
            else None
        )

        eos_token_ids, eos_sequences = await self._get_eos_variables(
            request.sampling_params.ignore_eos,
            request.sampling_params.stop_token_ids,
            request.sampling_params.stop,
        )

        # Calculate Max Length
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            len(token_ids), self.max_length, max_new_tokens
        )

        context = TextContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            eos_sequences=eos_sequences,
            max_length=len(token_ids) + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            tokens=np.array(token_ids),
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json_schema,
            sampling_params=request.sampling_params,
            model_name=request.model_name,
            lora_name=request.lora_name,
        )
        context.assign_to_cache(request.index)
        return context

    @property
    def _is_llama_tokenizer(self) -> bool:
        tokenizers = (
            LlamaTokenizer,
            LlamaTokenizerFast,
            CodeLlamaTokenizer,
            CodeLlamaTokenizerFast,
        )
        return isinstance(self.delegate, tokenizers)

    @property
    def _llama_whitespace_fix_dummy_token(self) -> tuple[int, int]:
        dummy_token_id = 33  # \x1e
        dummy_token_decoded = self.delegate.decode([dummy_token_id])
        return dummy_token_id, len(dummy_token_decoded)

    def _decode_with_llama_whitespace_fix(
        self, encoded: np.ndarray, **kwargs
    ) -> str:
        if encoded.shape == ():
            # The np.insert below will replace the token instead of prepend it
            # if the array is actually a scalar.  Reshape to a 1-length rank-1
            # array in this case.  See MODELS-467 for symptom.
            encoded = encoded.reshape((1,))
        decoded = self.delegate.decode(
            np.insert(encoded, 0, self._llama_whitespace_fix_dummy_token_id),
            **kwargs,
        )
        return decoded[self._llama_whitespace_fix_dummy_token_len :]

    async def _encode_stop_criteria(self, stop: list[str]) -> list[list[int]]:
        """Encodes `stop` to be used as stop criteria during generation."""
        stop_tokenized: list[list[int]] = []
        for stop_crit in stop:
            tokenized: list[int] = (
                await self.encode(stop_crit, False)
            ).tolist()
            stop_tokenized.append(tokenized)

        return stop_tokenized


class TextAndVisionTokenizer(
    PipelineTokenizer[TextAndVisionContext, np.ndarray, TextGenerationRequest],
):
    """Encapsulates creation of TextContext and specific token encode/decode logic."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            # If `max_length` is None, the max length will be taken
            # from the HuggingFace tokenizer_config.
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        # As we are adding special tokens during chat templating prior to tokenization,
        # when add_special_tokens=True, we duplicate BOS tokens specifically.
        self._encode_with_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=True
        )
        self._encode_without_special_tokens = functools.partial(
            self.delegate.encode, add_special_tokens=False
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=revision, trust_remote_code=trust_remote_code
        )
        self._default_eos_token_ids = set([self.eos])

    def _wrap_str_message_content(
        self, messages: list[TextGenerationRequestMessage]
    ) -> list[TextGenerationRequestMessage]:
        # Wrap string type values of "content" key with "type": "text" and its
        # value. For example, if the message is {"content": "Hello, world!"},
        # it will be wrapped with {"type": "text", "text": "Hello, world!"}.
        # This is a workaround for LlamaVision's chat template:
        # https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct/blob/main/chat_template.json
        for message in messages:
            if isinstance(message["content"], str):
                message["content"] = [
                    {"type": "text", "text": message["content"]}
                ]
            elif isinstance(message["content"], list):
                for content in message["content"]:
                    if "content" in content and content["type"] == "text":
                        content["text"] = content.pop("content")
        return messages

    def apply_chat_template(
        self, messages: list[TextGenerationRequestMessage]
    ) -> str:
        # TODO: Refactor this.
        if self.model_path == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            messages = self._wrap_str_message_content(messages)
        try:
            templated_message = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return cast(str, templated_message)
        except Exception as e:
            msg = "apply_chat_template failed for TextAndVisionTokenizer"
            logger.warning(msg)
            logger.warning(str(e))
            prompt = []
            for message in messages:
                if isinstance(message["content"], str):
                    prompt.append(message["content"])
                elif isinstance(message["content"], list):
                    for content in message["content"]:
                        if content["type"] == "text":
                            if "text" in content:
                                prompt.append(content["text"])
                            else:
                                prompt.append(content["content"])
            return "\n".join(prompt)

    @property
    def eos(self) -> int:
        return self.delegate.eos_token_id

    @property
    def expects_content_wrapping(self) -> bool:
        return True

    async def encode(
        self, prompt: Union[str, Sequence[int]], add_special_tokens: bool = True
    ) -> np.ndarray:
        """Transform the provided prompt into a token array."""

        encoded_prompt: np.ndarray
        if isinstance(prompt, str):
            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            if add_special_tokens:
                encoded_prompt = await run_with_default_executor(
                    self._encode_with_special_tokens, prompt
                )
            else:
                encoded_prompt = await run_with_default_executor(
                    self._encode_without_special_tokens, prompt
                )

            max_length = self.max_length or self.delegate.model_max_length
            if max_length and len(encoded_prompt) > max_length:
                raise ValueError(
                    f"Input string is larger than tokenizer's max length ({len(encoded_prompt)} > {max_length})."
                )
        else:
            encoded_prompt = np.array(list(prompt))

        return encoded_prompt

    async def decode(self, encoded: np.ndarray, **kwargs) -> str:
        """Transformer a provided encoded token array, back into readable text."""
        return self.delegate.decode(encoded, **kwargs)

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext object, leveraging necessary information like
        cache_seq_id and prompt from TextGenerationRequest."""
        prompt: Union[str, Sequence[int]]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages is not None:
            prompt = self.apply_chat_template(request.messages)
            add_special_tokens = False
        else:
            msg = f"{request} does not provide messages or prompt."
            raise ValueError(msg)

        # Load images.
        images = (
            [
                _convert_image_mode(Image.open(io.BytesIO(image_data)), "RGB")
                for image_data in request.images
            ]
            if request.images
            else None
        )
        # LlamaVision & InternVL returns a python list
        processed_inputs = self.processor(
            text=prompt,
            images=images,
            add_special_tokens=add_special_tokens,
            return_tensors="np",
        )

        if "input_ids" not in processed_inputs:
            msg = "input_ids not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
            raise ValueError(msg)

        # TODO: This is a hack to support both LlamaVision, Pixtral and InternVL.
        if isinstance(processed_inputs["input_ids"][0], int):
            encoded_prompt = np.array(processed_inputs["input_ids"])
        else:
            encoded_prompt = np.array(processed_inputs["input_ids"][0])

        # TODO(zheng): We should probably just make max_new_tokens an optional
        # instead of -1.
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        extra_model_args = dict()

        if images is not None:
            if "pixel_values" not in processed_inputs:
                msg = "pixel_values not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
                raise ValueError(msg)
            pixel_values = processed_inputs["pixel_values"][0]
            if isinstance(pixel_values, list):
                pixel_values = tuple(pixel_values)
            elif isinstance(pixel_values, np.ndarray):
                pixel_values = (pixel_values,)
            else:
                raise ValueError(
                    f"pixel_values is not a numpy array but it is {type(pixel_values)}"
                )

            if "aspect_ratio_ids" in processed_inputs:
                extra_model_args["aspect_ratio_ids"] = (
                    processed_inputs.aspect_ratio_ids
                )
            if "aspect_ratio_mask" in processed_inputs:
                extra_model_args["aspect_ratio_mask"] = (
                    processed_inputs.aspect_ratio_mask
                )
        else:
            pixel_values = tuple()

        # Pass through image token indices if present
        if "image_token_indices" in processed_inputs:
            extra_model_args["image_token_indices"] = processed_inputs[
                "image_token_indices"
            ]

        json_schema = (
            json.dumps(request.response_format.get("json_schema", None))
            if request.response_format
            else None
        )

        if request.sampling_params.ignore_eos:
            eos_token_ids = set()
        else:
            eos_token_ids = self._default_eos_token_ids

        context = TextAndVisionContext(
            request_id=request.request_id,
            eos_token_ids=eos_token_ids,
            pixel_values=pixel_values,
            extra_model_args=extra_model_args,
            tokens=encoded_prompt,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            json_schema=json_schema,
        )
        context.assign_to_cache(request.index)
        return context


def _rgba_to_rgb(
    image: Image.Image,
    background_color=(255, 255, 255),  # noqa: ANN001
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def _convert_image_mode(image: Image.Image, to_mode: str):
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return _rgba_to_rgb(image)
    else:
        return image.convert(to_mode)
