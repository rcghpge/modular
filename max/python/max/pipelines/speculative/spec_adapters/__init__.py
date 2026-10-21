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
"""How spec decoding calls each target architecture's module.

The modules MAX architectures build share no calling convention: each takes its
inputs in an order of its own and returns its outputs by position. A driver
can't call a target without knowing which architecture built it, so each
target's package holds a target adapter, its ``spec_adapters`` module, that
holds its call. ``_TARGET_ADAPTERS`` names each target architecture's adapter
module. A target adapter takes what it leaves out from Llama 3's.

A target whose checkpoint ships its draft names the draft's architecture by
method, as vLLM's ``hf_config_override`` names an MTP draft from its target's
model type.

A module is imported only when one of its targets runs, so spec decoding
imports no architecture it doesn't run.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy.typing as npt
from max.graph import BufferType, TensorType, Value
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates

from ..block_driver import BlockBatch, BlockProposer
from ..config import SpeculativeConfig
from ..driver import SequentialBatch
from ..spec_target import Verified
from .target_adapter import TargetAdapter

__all__ = [
    "VerifyFn",
    "block_proposer",
    "ep_input_types",
    "extra",
    "primary_kv",
    "trailing_arrays",
    "trailing_input_types",
    "verify_for",
]

VerifyFn = Callable[[SequentialBatch | BlockBatch], Verified[Any]]
"""Verifies a driver batch on the target."""

_TARGET_ADAPTERS: Mapping[str, str] = {
    "DeepseekV3ForCausalLM": ".deepseekV3.spec_adapters",
    "DeepseekV4ForCausalLM": ".deepseekV4.spec_adapters",
    "Gemma4ForConditionalGeneration": ".gemma4.spec_adapters",
    "Gemma4UnifiedForConditionalGeneration": ".gemma4.spec_adapters",
    "GraniteForCausalLM": ".llama3.spec_adapters",
    "LlamaForCausalLM": ".llama3.spec_adapters",
    "MiMoV2ForCausalLM": ".mimo_v2.spec_adapters",
}


def _target_adapter(target_arch: str) -> TargetAdapter:
    if target_arch not in _TARGET_ADAPTERS:
        raise ValueError(
            f"Spec decoding has no call for a {target_arch} target."
        )
    adapter = importlib.import_module(
        _TARGET_ADAPTERS[target_arch], "max.pipelines.architectures"
    ).ADAPTER
    assert isinstance(adapter, TargetAdapter)
    return adapter


def _adapter(target_arch: str, method: str | None = None) -> TargetAdapter:
    """Returns the target's adapter."""
    del method
    return _target_adapter(target_arch)


def verify_for(
    target_arch: str,
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns how a built target module verifies under a spec-decode driver.

    Args:
        target_arch: The target checkpoint's architecture name.
        target: The module the target's pipeline model built.
        target_config: The config it was built from.
        hidden_states: The hidden states it was built to return.
        draft_config: The config the draft was built from.
    """
    return _target_adapter(target_arch).verify(
        target, target_config, hidden_states, draft_config
    )


def primary_kv(target_arch: str) -> str:
    """Returns the name of the KV leaf a driver dispatches the target over.

    The target's other leaves ride in the driver batch's ``passthrough_kv``,
    under their names in the KV tree. A target with one leaf has the name
    ``""``.
    """
    return _target_adapter(target_arch).primary_kv


def ep_input_types(
    target_arch: str, target: Module
) -> Sequence[TensorType | BufferType]:
    """Returns the expert-parallel inputs of a built target module."""
    return _adapter(target_arch).ep_input_types(target)


def trailing_input_types(
    target_arch: str, target_config: Any
) -> Sequence[TensorType | BufferType]:
    """Returns the inputs the target's call takes past the canonical tail."""
    return _adapter(target_arch).trailing_input_types(target_config)


def trailing_arrays(
    target_arch: str, target_config: Any, lengths: Sequence[int]
) -> list[npt.NDArray[Any]]:
    """Returns the host values of :func:`trailing_input_types`.

    Args:
        target_arch: The target checkpoint's architecture name.
        target_config: The config the target was built from.
        lengths: Each request's verify-forward length: its tokens and the
            drafts it verifies.
    """
    return _adapter(target_arch).trailing_arrays(target_config, lengths)


def extra(
    target_arch: str, target_config: Any, trailing: Sequence[Value[Any]]
) -> dict[str, Any]:
    """Returns what the target's call reads of the trailing graph inputs.

    The driver hands it to the verify and the proposer as the batch's
    ``extra``.
    """
    return _adapter(target_arch).extra(target_config, trailing)


def block_proposer(
    target_arch: str,
    verify: VerifyFn,
    draft: Module,
    draft_config: Any,
    speculative: SpeculativeConfig,
) -> BlockProposer[Any, Any] | None:
    """Returns the proposer of a draft that runs on its target's own state.

    ``None`` for every other draft, which its method's proposer runs.
    """
    return _adapter(target_arch).block_proposer(
        verify, draft, draft_config, speculative
    )
