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
"""Helpers and the pipeline interface for cascade generative-AI types.

The request and chunk types themselves -- :class:`GenAIRequest`,
:class:`ChatMessage`, :class:`GenAIChunk`, ... -- are defined once in
``gen_ai.capnp`` and imported from the pycapnp-generated
``max.experimental.cascade.interfaces.gen_ai_capnp`` module. Build them with
``new_message`` (nested dicts initialize nested structs and lists in place)
and dispatch on a chunk's ``which()``::

    GenAIRequest.new_message(
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
        text={"numTokens": 64},
    )

Splitting a model's raw output into chunks is the pipeline's job, because the
pipeline is what knows the model's format; the serving layer only maps chunks
onto the schema its route speaks.

Kept in ``interfaces`` (below both ``workers`` and the concrete ``pipelines``)
so workers can consume the request types and ``serve`` can route on the
interface without either depending on concrete pipeline implementations.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from enum import Enum
from typing import Protocol, TypeVar

from max.experimental.cascade.core import pipeline_method
from max.experimental.cascade.interfaces.gen_ai_capnp import (
    GenAIChunk,
    GenAIRequest,
)


class Modality(str, Enum):
    """A kind of content a pipeline can consume or produce."""

    TEXT = "text"
    IMAGE = "image"


_T = TypeVar("_T")


class _Optional(Protocol[_T]):
    """An optional scalar: a union of ``unset`` and ``value``."""

    value: _T

    def which(self) -> str: ...


def optional(field: _Optional[_T]) -> _T | None:
    """Reads an optional scalar as its value, or ``None`` when unset."""
    return field.value if field.which() == "value" else None


def tools_enabled(req: GenAIRequest) -> bool:
    """Whether tool calls are possible for this request."""
    return len(req.tools) > 0 and req.toolChoice != "none"


def tool_schemas(req: GenAIRequest) -> dict[str, dict[str, object]]:
    """Map each offered tool's name to its parameter JSON Schema."""
    return {
        tool.name: json.loads(tool.parametersJson)
        for tool in req.tools
        if tool.parametersJson
    }


class GenAIInterface(ABC):
    """Standard interface for a generative-AI model of any modality.

    The composable primitive is :meth:`_generate_iterator`, which returns the
    chunk stream as a worker handle. A composing pipeline can forward that handle
    to another worker so post-processing runs worker-to-worker without routing
    every chunk through the orchestrator. :meth:`generate` is the standalone
    entry point, and just wraps the handle in a pipeline scope.
    """

    @abstractmethod
    def supported_input_modalities(self) -> set[Modality]:
        """Return the modalities this pipeline accepts in request messages."""
        ...

    @abstractmethod
    def supported_output_modalities(self) -> set[Modality]:
        """Return the modalities this pipeline can emit as chunks."""
        ...

    @abstractmethod
    async def _generate_iterator(
        self, req: GenAIRequest
    ) -> AsyncIterable[GenAIChunk]:
        """Open a streaming handle to the generated chunks.

        Returns a worker stream handle rather than a materialized iterator, so a
        composing pipeline can hand it to another worker. Must be called within
        an active pipeline scope; the handle stays valid for that scope.

        Prefixed to mark it as the composition primitive: direct callers want
        :meth:`generate`, which establishes a scope of its own. The pair exists
        because :func:`~max.experimental.cascade.core.pipeline_method` scopes do
        not nest -- an inner scope shadows the outer one and would tear down the
        stream early.
        """
        ...

    @pipeline_method
    async def generate(self, req: GenAIRequest) -> AsyncIterator[GenAIChunk]:
        """Generate a streaming response to a request."""
        async for chunk in await self._generate_iterator(req):
            yield chunk
