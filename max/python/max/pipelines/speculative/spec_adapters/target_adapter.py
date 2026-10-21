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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates

if TYPE_CHECKING:
    from . import VerifyFn


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

    verify: Callable[[Module, Any, ReturnHiddenStates], VerifyFn]
    """Returns how a built target module verifies under a driver."""
