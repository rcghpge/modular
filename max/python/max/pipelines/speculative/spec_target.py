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
"""What a spec-decode driver's verify callable returns.

A target is the model being verified, so nothing here depends on how the draft
proposes. The drafting strategy shows up in one place only: which batch type
the driver hands its verify callable.
"""

from __future__ import annotations

from typing import Generic

from max import tree
from max.graph import TensorValue
from typing_extensions import TypeVar

__all__ = ["Verified"]

_TargetHiddenT = TypeVar("_TargetHiddenT")
"""The target's hidden-state payload, opaque to whichever driver carries it."""

_LogitsT = TypeVar("_LogitsT", default=TensorValue)
"""The logits' value type: a graph value, or an experimental ``Tensor``."""


@tree.dataclass(frozen=True)
class Verified(Generic[_TargetHiddenT, _LogitsT]):
    """What the target contributes: verification logits and hidden states.

    Naming the two results is the point. The target output tuple is
    ``[last_logits?][logits, offsets?][hidden per device...]``, and whether the
    leading entry exists depends on ``emit_last_token_logits`` -- which is why
    structurally identical code reads ``[1]`` and ``[3 : 3 + n]`` in one arch
    and ``[0]`` and ``[2 : 2 + n]`` in another. The adapter owns that layout.
    """

    logits: _LogitsT
    hidden: _TargetHiddenT
