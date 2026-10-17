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

"""DeepSeek-V4.1 tokenizer rendering prompts with the checkpoint's encoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from max.pipelines.lib.tokenizer import ReasoningTextTokenizer
from max.pipelines.modeling.types import (
    TextContentPart,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)

from .encoding_dsv41 import encode_messages
from .reasoning_effort import resolve_reasoning_effort

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


def _flatten_message(message: TextGenerationRequestMessage) -> dict[str, Any]:
    """Flattens a message to a role/content dict for the V4.1 encoder.

    Joins multi-part text with ``\\n\\n``, the separator the V4.1 prompt
    format uses between text parts.
    """
    if isinstance(message.content, str):
        content_str = message.content
    else:
        parts: list[str] = []
        for part in message.content:
            if isinstance(part, TextContentPart):
                parts.append(part.text)
            else:
                raise ValueError("only text content can be flattened.")
        content_str = "\n\n".join(parts)

    flattened: dict[str, Any] = {
        "role": str(message.role),
        "content": content_str,
    }
    if message.tool_calls is not None:
        flattened["tool_calls"] = message.tool_calls
    if message.tool_call_id is not None:
        flattened["tool_call_id"] = message.tool_call_id
    if message.reasoning_content is not None:
        flattened["reasoning_content"] = message.reasoning_content
    return flattened


def render_prompt(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[TextGenerationRequestTool] | None,
    chat_template_options: Mapping[str, Any],
) -> str:
    """Renders an OpenAI-style conversation into a DeepSeek-V4.1 prompt.

    Thinking is on by default, with the default reasoning-effort budget.

    Args:
        messages: Flattened messages (``role``, ``content`` and the optional
            ``tool_calls``, ``tool_call_id``, ``reasoning_content`` keys).
        tools: OpenAI-format tool definitions, attached to the leading system
            message (one is inserted when the conversation has none).
        chat_template_options: Request chat-template kwargs. Reads
            ``thinking``, ``enable_thinking``, ``reasoning_effort`` and
            ``drop_thinking``.

    Returns:
        The prompt string, BOS included.

    Raises:
        InputError: If ``reasoning_effort`` is not ``None``, ``"none"``, an
            int in [1, 100], or a named effort.
        ValueError: If the encoder rejects the conversation for any other
            reason.
    """
    thinking_mode, reasoning_effort = resolve_reasoning_effort(
        chat_template_options
    )

    conversation = [dict(m) for m in messages]
    if tools:
        if not conversation or conversation[0].get("role") != "system":
            conversation.insert(0, {"role": "system", "content": ""})
        conversation[0]["tools"] = list(tools)

    try:
        return encode_messages(
            conversation,
            thinking_mode=thinking_mode,
            drop_thinking=bool(
                chat_template_options.get("drop_thinking", True)
            ),
            reasoning_effort=reasoning_effort,
        )
    except (AssertionError, NotImplementedError) as e:
        # The encoder signals unsupported input with these.
        raise ValueError(f"DeepSeek-V4.1 prompt encoding failed: {e}") from e


class DeepseekV41Tokenizer(ReasoningTextTokenizer):
    """Tokenizer for DeepSeek-V4.1 that renders chat with the V4.1 encoder.

    A custom ``--chat-template``, when given, replaces the encoder.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        **kwargs,
    ) -> None:
        super().__init__(model_path, pipeline_config, **kwargs)

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        **chat_template_options: Any,
    ) -> str:
        """Renders messages with the DeepSeek-V4.1 encoder."""
        if self._custom_template_provided:
            return super().apply_chat_template(
                messages, tools, **chat_template_options
            )
        return render_prompt(
            [_flatten_message(message) for message in messages],
            tools,
            chat_template_options,
        )
