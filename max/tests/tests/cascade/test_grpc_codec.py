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
"""Tests for the gRPC value-slot codec."""

from __future__ import annotations

import pytest
from max.experimental.cascade.core import Runtime
from max.experimental.cascade.grpc_runtime.codec import (
    decode_value_slot,
    encode_value_slot,
)
from max.experimental.cascade.interfaces import gen_ai_capnp


def _no_dial(target: str) -> Runtime:
    raise AssertionError(f"unexpected dial to {target!r}")


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.asyncio
async def test_capnp_list_round_trips_with_a_type_hint(count: int) -> None:
    messages = gen_ai_capnp.GenAIRequest.new_message().init("messages", count)
    for message in messages:
        message.role = "user"

    decoded = await decode_value_slot(
        encode_value_slot(messages),
        _no_dial,
        list[gen_ai_capnp.ChatMessage],
    )

    assert [message.role for message in decoded] == ["user"] * count
