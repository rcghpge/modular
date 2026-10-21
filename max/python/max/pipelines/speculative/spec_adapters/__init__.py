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
module.

A module is imported only when one of its targets runs, so spec decoding
imports no architecture it doesn't run.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates

from ..block_driver import BlockBatch
from ..driver import SequentialBatch
from ..spec_target import Verified
from .target_adapter import TargetAdapter

__all__ = ["VerifyFn", "primary_kv", "verify_for"]

VerifyFn = Callable[[SequentialBatch | BlockBatch], Verified[Any]]
"""Verifies a driver batch on the target."""

_TARGET_ADAPTERS: Mapping[str, str] = {
    "Gemma4ForConditionalGeneration": ".gemma4.spec_adapters",
    "Gemma4UnifiedForConditionalGeneration": ".gemma4.spec_adapters",
    "GraniteForCausalLM": ".llama3.spec_adapters",
    "LlamaForCausalLM": ".llama3.spec_adapters",
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


def verify_for(
    target_arch: str,
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
) -> VerifyFn:
    """Returns how a built target module verifies under a spec-decode driver.

    Args:
        target_arch: The target checkpoint's architecture name.
        target: The module the target's pipeline model built.
        target_config: The config it was built from.
        hidden_states: The hidden states it was built to return.
    """
    return _target_adapter(target_arch).verify(
        target, target_config, hidden_states
    )


def primary_kv(target_arch: str) -> str:
    """Returns the name of the KV leaf a driver dispatches the target over.

    The target's other leaves ride in the driver batch's ``passthrough_kv``,
    under their names in the KV tree. A target with one leaf has the name
    ``""``.
    """
    return _target_adapter(target_arch).primary_kv
