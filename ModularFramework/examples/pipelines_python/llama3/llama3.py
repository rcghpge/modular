# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeAlias, TypeVar

from .config import Llama3Config

Context = TypeVar("Context")


# TODO: Remove this re-implementation when max.serve is in monorepo.
class TokenGenerator(Generic[Context], Protocol):
    """Interface for LLM token-generator models."""

    async def new_context(self, prompt: str) -> Context:
        ...

    async def next_token(self, contexts: dict[str, Context]) -> dict[str, str]:
        ...


def get_max_tokens_to_generate(
    prompt_len: int, max_length: int, max_new_tokens: int
) -> int:
    """Returns the max number of tokens to generate (including the prompt)."""
    if max_new_tokens < 0:
        return max_length

    return min(max_new_tokens + prompt_len, max_length)


@dataclass
class Llama3Context:
    """The context for text generation using a Llama 3 model."""

    prompt_size: int
    max_tokens: int


class Llama3:
    config: Llama3Config

    def __init__(self, config: Llama3Config):
        self.config = config
        print(self.config)

    async def new_context(self, prompt: str) -> Llama3Context:
        # TODO: Encode prompt
        prompt_size = len(prompt)
        max_tokens = get_max_tokens_to_generate(
            prompt_size, self.config.max_length, self.config.max_new_tokens
        )
        # TODO: Create new empty KV cache
        # TODO: Get prompt tensor
        print("Initializing context for Llama 3...")
        return Llama3Context(prompt_size, max_tokens)

    async def next_token(
        self, contexts: dict[str, Llama3Context]
    ) -> dict[str, str]:
        # TODO: Provide previous tokens to compiled model
        # TODO: Provide previous KV cache to compiled model
        # TODO: Execute model
        # TODO: Get logits
        # TODO: Update KV cache
        # TODO: Sample the next token

        # Note: assuming a single request.
        request_id, context = next(iter(contexts.items()))

        decoded_token = request_id
        if decoded_token == "end":
            return {}
        return {request_id: decoded_token}
