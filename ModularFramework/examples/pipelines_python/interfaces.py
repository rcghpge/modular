# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Interfaces for different pipeline behaviors."""

from typing import Generic, Protocol, TypeVar

Context = TypeVar("Context")


class TokenGenerator(Generic[Context], Protocol):
    """Interface for LLM token-generator models."""

    async def new_context(self, prompt: str) -> Context:
        ...

    async def next_token(self, batch: dict[str, Context]) -> dict[str, str]:
        ...
