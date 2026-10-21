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
"""The call a spec-decode driver makes of one kind of target."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy.typing as npt
from max.graph import BufferType, TensorType, Value
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates

if TYPE_CHECKING:
    from ..block_driver import BlockProposer
    from ..config import SpeculativeConfig
    from . import VerifyFn


def _default_ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns no expert-parallel inputs."""
    del target
    return ()


def _default_trailing_input_types(target_config: Any) -> list[TensorType]:
    """Returns no inputs past the canonical tail."""
    del target_config
    return []


def _default_extra(
    target_config: Any, trailing: Sequence[Value[Any]]
) -> dict[str, Any]:
    """Returns nothing for a driver batch's ``extra``."""
    del target_config, trailing
    return {}


def _default_block_proposer(
    verify: VerifyFn,
    draft: Module,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any] | None:
    """Returns ``None``: the method's own proposer runs the draft."""
    del verify, draft, draft_config, speculative
    return None


def _default_trailing_arrays(
    target_config: Any, lengths: Sequence[int]
) -> list[npt.NDArray[Any]]:
    """Returns no host inputs past the canonical tail."""
    del target_config, lengths
    return []


@dataclass(frozen=True, kw_only=True)
class TargetAdapter:
    """How a spec-decode driver calls a target architecture's module.

    A field a target leaves out takes a default: one KV leaf, no
    expert-parallel inputs, nothing past the canonical tail, and the
    method's own proposer. ``verify`` has none.
    """

    primary_kv: str = ""
    """The KV leaf a driver dispatches the target over; ``""`` for a target
    with one leaf."""

    verify: Callable[[Module, Any, ReturnHiddenStates, Any], VerifyFn]
    """Returns how a built target module verifies under a driver."""

    ep_input_types: Callable[[Any], Sequence[TensorType | BufferType]] = (
        _default_ep_input_types
    )
    """Returns a built target module's expert-parallel inputs."""

    trailing_input_types: Callable[[Any], Sequence[TensorType | BufferType]] = (
        _default_trailing_input_types
    )
    """Returns the inputs the target's call takes past the canonical tail."""

    extra: Callable[[Any, Sequence[Value[Any]]], dict[str, Any]] = (
        _default_extra
    )
    """Returns what the target's call reads of the trailing graph inputs."""

    block_proposer: Callable[
        [VerifyFn, Module, Any, SpeculativeConfig],
        BlockProposer[Any, Any] | None,
    ] = _default_block_proposer
    """Returns the proposer of a draft that runs on the target's own state,
    or ``None`` for a draft its method's proposer runs."""

    trailing_arrays: Callable[..., Any] = _default_trailing_arrays
