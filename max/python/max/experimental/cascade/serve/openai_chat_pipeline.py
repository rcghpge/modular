# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Expose a generative pipeline over the OpenAI chat-completions schema.

Owns the translation in both directions: an OpenAI request becomes a
:class:`GenAIRequest`, and the wrapped pipeline's chunks become OpenAI response
bytes. Nothing here knows how any particular model formats reasoning or tool
calls -- the wrapped pipeline has already turned that into typed chunks -- and
nothing above here knows about OpenAI, so the HTTP route is left with only the
FastAPI boilerplate. Everything here is pydantic-to-pydantic: no HTTP types
cross into this module.

The chunk stream is forwarded to an :class:`OpenAIChatFormatter` worker rather
than serialized inline, so the per-chunk JSON encoding runs on the CPU worker
pool instead of the API event loop, and never round-trips the orchestrator.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Mapping, Sequence
from typing import Any

from max.experimental.cascade.core import pipeline_method
from max.experimental.cascade.interfaces.gen_ai import GenAIInterface
from max.experimental.cascade.interfaces.gen_ai_capnp import (
    GenAIRequest,
    TextGenOptions,
)
from max.experimental.cascade.interfaces.pipeline import CascadePipeline
from max.experimental.cascade.serve.openai_chat_formatter import (
    OpenAIChatFormatter,
)
from max.pipelines.lib.tool_parsing import maybe_name_from_tool
from max.serve.schemas.openai import CreateChatCompletionRequest

# TODO(GEX-2412): mint a unique id per request.
REQUEST_ID = "chatcmpl-cascade"


def _convert_stop(stop: str | Sequence[str] | None) -> list[str] | None:
    """Normalize the OpenAI ``stop`` field into a list of strings."""
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def _message_text(content: str | Sequence[Mapping[str, Any]]) -> str:
    """Flatten an OpenAI message content payload into plain text.

    Raises:
        ValueError: If the message carries a non-text content part. The route
            reports it as a ``400``.
    """
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    unsupported_types: list[str] = []
    for part in content:
        if part.get("type") == "text" and part.get("text") is not None:
            text_parts.append(part["text"])
        else:
            unsupported_types.append(part.get("type", "unknown"))

    if unsupported_types:
        raise ValueError(
            "Unsupported chat message content part types: "
            + ", ".join(sorted(set(unsupported_types)))
        )

    return "".join(text_parts)


def _fill_tools(
    req: GenAIRequest, request: CreateChatCompletionRequest
) -> None:
    """Write the request's tool definitions into ``req.tools``.

    Tools without a resolvable name are dropped, matching the MAX Serve route.
    """
    named = [
        (name, tool)
        for tool in request.tools or []
        if (name := maybe_name_from_tool(tool))
    ]
    for target, (name, tool) in zip(
        req.init("tools", len(named)), named, strict=True
    ):
        target.name = name
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        if isinstance(function.get("parameters"), dict):
            target.parametersJson = json.dumps(function["parameters"])
        if isinstance(function.get("description"), str):
            target.description = function["description"]


# Optional sampling fields forwarded verbatim; unset ones stay on the model
# default.
_OPTIONAL_SAMPLING_FIELDS = (
    ("topK", "top_k"),
    ("topP", "top_p"),
    ("minP", "min_p"),
    ("thinkingTemperature", "thinking_temperature"),
    ("seed", "seed"),
    ("frequencyPenalty", "frequency_penalty"),
    ("presencePenalty", "presence_penalty"),
    ("repetitionPenalty", "repetition_penalty"),
)


def _fill_text_options(
    options: TextGenOptions, request: CreateChatCompletionRequest
) -> None:
    """Write the request's sampling fields into *options*.

    Fields the client did not supply keep the schema defaults.
    """
    for field, request_field in _OPTIONAL_SAMPLING_FIELDS:
        value = getattr(request, request_field)
        if value is not None:
            getattr(options, field).value = value
    if request.ignore_eos is not None:
        options.ignoreEos = request.ignore_eos
    if stop := _convert_stop(request.stop):
        options.stop = stop
    if request.stop_token_ids:
        options.stopTokenIds = request.stop_token_ids
    # ``max_completion_tokens`` supersedes the legacy ``max_tokens``.
    max_new_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )
    if max_new_tokens is not None:
        options.numTokens = max_new_tokens
    if request.min_tokens is not None:
        options.minNewTokens = request.min_tokens
    if request.temperature is not None:
        options.temperature = request.temperature


def to_gen_ai_request(request: CreateChatCompletionRequest) -> GenAIRequest:
    """Convert an OpenAI chat-completion request into a cascade request.

    Raises:
        ValueError: If the request carries content this schema cannot express.
    """
    req = GenAIRequest.new_message()
    messages = req.init("messages", len(request.messages))
    for target, message in zip(messages, request.messages, strict=True):
        target.role = message.get("role", "")
        content = _message_text(message.get("content") or "")
        target.init("content", 1)[0].text = content
    _fill_tools(req, request)
    if request.tool_choice in ("none", "required"):
        req.toolChoice = request.tool_choice
    _fill_text_options(req.text, request)
    return req


class OpenAIChatCompletionPipeline(CascadePipeline):
    """Serve a generative pipeline over the OpenAI chat-completions schema."""

    def __init__(
        self, gen_ai: GenAIInterface, emit_reasoning_content: bool = False
    ) -> None:
        """Wrap *gen_ai*, adding a formatter worker for OpenAI chat responses.

        Args:
            gen_ai: An already-deployed pipeline to expose over OpenAI chat.
                :meth:`deploy` only brings up this wrapper's own formatter
                worker; ``gen_ai`` is deployed by whoever owns it.
            emit_reasoning_content: Server-wide reasoning field choice, fixed
                onto the formatter worker here.
        """
        self.gen_ai = gen_ai
        self.formatter = OpenAIChatFormatter(emit_reasoning_content)

    async def chat_completion_sse(
        self, request: CreateChatCompletionRequest
    ) -> AsyncIterator[bytes]:
        """Translate *request*, then open the stream of OpenAI SSE frames.

        Awaiting this does the translation and hands back a stream that has not
        started yet, so a request this schema cannot express still surfaces as a
        rejection rather than a truncated response body: by the time a caller
        starts forwarding frames, the response status is already committed.

        Splitting the translation out here is also what leaves room to move it
        onto a worker, once it grows work worth taking off the event loop (an
        ``image_url`` part that has to be fetched, say).

        Raises:
            ValueError: If the request carries content this schema cannot
                express.
        """
        req = to_gen_ai_request(request)
        return self._sse_frames(req, request.model)

    @pipeline_method
    async def _sse_frames(
        self, req: GenAIRequest, model: str
    ) -> AsyncIterator[bytes]:
        """Stream OpenAI chat SSE frames, serializing on the worker pool.

        The wrapped pipeline's chunk handle flows straight into the formatter
        worker, so both run worker-to-worker inside this one pipeline scope and
        the orchestrator only forwards the finished byte frames.
        """
        chunks = await self.gen_ai._generate_iterator(req)
        async for frame in await self.formatter.format_stream(
            chunks, model, REQUEST_ID, int(time.time())
        ):
            yield frame

    @pipeline_method
    async def chat_completion(
        self, request: CreateChatCompletionRequest
    ) -> Awaitable[bytes]:
        """Translate *request* and serialize a non-streaming OpenAI response.

        Mirrors :meth:`chat_completion_sse` for the non-streaming case: the
        formatter collects the chunk stream and serializes the finished JSON
        body, so the orchestrator only forwards the finished bytes. No split is
        needed here -- the whole body is produced before the caller sees
        anything, so a translation failure is still reportable.

        Raises:
            ValueError: If the request carries content this schema cannot
                express.
        """
        req = to_gen_ai_request(request)
        chunks = await self.gen_ai._generate_iterator(req)
        return await self.formatter.format_complete(
            chunks, request.model, REQUEST_ID, int(time.time())
        )
