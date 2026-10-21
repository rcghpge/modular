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
"""Gemma 4's language-model call."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.block_driver import BlockBatch
from max.pipelines.speculative.driver import SequentialBatch
from max.pipelines.speculative.spec_adapters.target_adapter import TargetAdapter
from max.pipelines.speculative.spec_target import Verified

if TYPE_CHECKING:
    from max.pipelines.speculative.spec_adapters import VerifyFn

PRIMARY_KV = "full_attention"

_SLIDING_KV = "sliding_attention"


def verify(
    target: Module,
    target_config: Any,
    hidden_states: ReturnHiddenStates,
    draft_config: Any,
) -> VerifyFn:
    """Returns the verify of a single-device Gemma 4 language model.

    Raises:
        ValueError: If the target wasn't built to return selected layers,
            which only block drafts read.
    """
    del draft_config
    if hidden_states != ReturnHiddenStates.SELECTED_LAYERS:
        raise ValueError(
            "A Gemma 4 target verifies only for block drafts, which read"
            f" selected layers. Got {hidden_states.value}."
        )
    return partial(
        _verify,
        target,
        target_config.text_config.hidden_size,
        target_config.unquantized_dtype,
    )


def _verify(
    target: Module,
    hidden_size: int,
    dtype: DType,
    batch: SequentialBatch | BlockBatch,
) -> Verified[list[TensorValue]]:
    # Gemma 4's embedding and head use collectives unconditionally, so the
    # entry point is the distributed one: every argument arrives as a
    # one-element list, beside the sliding-window cache and the vision
    # inputs a text-only forward still declares.
    assert isinstance(batch, BlockBatch)
    device = batch.device0
    # The vision merge scatters nothing on an empty index set, but the
    # signature still takes both operands.
    empty_embeds = ops.constant(
        np.zeros((0, hidden_size), dtype=np.float32),
        DType.float32,
        device=device,
    ).cast(dtype)
    empty_indices = ops.range(
        0, 0, 1, out_dim=0, dtype=DType.int32, device=device
    )
    outputs = target(
        batch.merged_tokens,
        batch.signal_buffers,
        batch.passthrough_kv[_SLIDING_KV],
        batch.kv_collections,
        batch.return_n_logits,
        batch.merged_offsets_per_dev,
        [empty_embeds],
        [empty_indices],
    )
    hidden = fuse_captured_hidden_states(captures_by_device(outputs[3:], 1))
    return Verified(logits=outputs[1], hidden=hidden)


ADAPTER = TargetAdapter(primary_kv=PRIMARY_KV, verify=verify)
