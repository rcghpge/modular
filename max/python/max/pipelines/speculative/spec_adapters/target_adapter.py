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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from max.driver import Buffer, Device
from max.graph import BufferType, TensorType, Value
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.context import TextContext

if TYPE_CHECKING:
    from ..block_driver import BlockProposer
    from ..config import SpeculativeConfig
    from . import VerifyFn


def _default_ep_input_types(target: Any) -> Sequence[TensorType | BufferType]:
    """Returns no expert-parallel inputs."""
    del target
    return ()


def _default_tail_buffers(tail: Sequence[object]) -> list[Buffer]:
    """Returns nothing.

    The graph declares every KV child among its KV inputs.
    """
    del tail
    return []


def _default_check(
    speculative: SpeculativeConfig, structured_output: bool
) -> None:
    """Accepts every speculative config."""
    del speculative, structured_output


def _default_trailing_input_types(target_config: Any) -> list[TensorType]:
    """Returns no inputs past the canonical tail."""
    del target_config
    return []


def _default_trailing_buffers(
    target_config: Any,
    contexts: Sequence[TextContext],
    devices: Sequence[Device],
) -> list[Buffer]:
    """Returns no batch inputs past the canonical tail."""
    del target_config, contexts, devices
    return []


def _default_extra(
    target_config: Any, trailing: Sequence[Value[Any]]
) -> dict[str, Any]:
    """Returns nothing for a driver batch's ``extra``."""
    del target_config, trailing
    return {}


def _default_takes_vision_outputs(target_config: Any) -> bool:
    del target_config
    return False


def _default_block_proposer(
    verify: VerifyFn,
    draft: Module,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any] | None:
    """Returns ``None``: the method's own proposer runs the draft."""
    del verify, draft, draft_config, speculative
    return None


def verify_lengths(contexts: Sequence[TextContext]) -> list[int]:
    """Returns each request's verify-forward length.

    That is its tokens and the drafts it verifies, which the pipeline stages
    on the context first.
    """
    return [
        ctx.tokens.active_length
        + len(ctx.spec_decoding_state.draft_tokens_to_verify)
        for ctx in contexts
    ]


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

    tail_kv: str | None = None
    """The KV child the graph declares past the canonical tail, if any."""

    verify: Callable[[Module, Any, ReturnHiddenStates, Any], VerifyFn]
    """Returns how a built target module verifies under a driver."""

    ep_input_types: Callable[[Any], Sequence[TensorType | BufferType]] = (
        _default_ep_input_types
    )
    """Returns a built target module's expert-parallel inputs."""

    tail_buffers: Callable[[Sequence[Any]], list[Buffer]] = (
        _default_tail_buffers
    )
    """Returns the inputs of the ``tail_kv`` child, in declared order."""

    check: Callable[[SpeculativeConfig, bool], None] = _default_check
    """Raises if the target can't verify under the speculative config."""

    trailing_input_types: Callable[[Any], Sequence[TensorType | BufferType]] = (
        _default_trailing_input_types
    )
    """Returns the inputs the target's call takes past the canonical tail."""

    trailing_buffers: Callable[
        [Any, Sequence[TextContext], Sequence[Device]], list[Buffer]
    ] = _default_trailing_buffers
    """Returns a batch's trailing values that no KV child holds."""

    extra: Callable[[Any, Sequence[Value[Any]]], dict[str, Any]] = (
        _default_extra
    )
    """Returns what the target's call reads of the trailing graph inputs."""

    takes_vision_outputs: Callable[[Any], bool] = _default_takes_vision_outputs
    """Returns whether the graph takes a vision encoder's output that MAX
    doesn't compute, for an engine that runs the encoder itself."""

    block_proposer: Callable[
        [VerifyFn, Module, Any, SpeculativeConfig],
        BlockProposer[Any, Any] | None,
    ] = _default_block_proposer
    """Returns the proposer of a draft that runs on the target's own state,
    or ``None`` for a draft its method's proposer runs."""

    by_method: Mapping[str, TargetAdapter] = field(default_factory=dict)
    """The adapter to use under a speculative method, where it differs."""
