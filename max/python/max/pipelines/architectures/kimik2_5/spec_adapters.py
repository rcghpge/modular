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
"""Speculative-decoding adapters for a Kimi K2.5 target.

The target is a :class:`DeepseekV3`, but it differs from a bare DeepseekV3
target in that it may scatter vision embeddings into the merged sequence
before running its stack. Its MLA draft's proposer only declares its
names; its MHA draft runs through
:class:`~max.pipelines.speculative.mla.Eagle3MHAProposer`.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.graph import (
    BufferType,
    TensorType,
    TensorValue,
)
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.transformer import captures_by_device
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
from max.pipelines.speculative.driver import (
    CarryDimNames,
    SequentialBatch,
)
from max.pipelines.speculative.mla import MLAProposer
from max.pipelines.speculative.spec_target import Verified

from ..deepseekV3.deepseekV3 import DeepseekV3

__all__ = [
    "Eagle3KimiK25Proposer",
    "KimiK25Target",
]

_TargetHidden = list[list[TensorValue]]
"""Kimi's Eagle3 captures: one list of aux layers per device."""


class KimiK25Target:
    """The Kimi K2.5 target entry point, optionally scattering vision."""

    def __init__(self, target: DeepseekV3, *, enable_vision: bool) -> None:
        self.target = target
        self.enable_vision = enable_vision

    def verify(self, batch: SequentialBatch) -> Verified[_TargetHidden]:
        if self.enable_vision:
            # Embed, scatter the image embeddings at their merge positions,
            # then run the rest of the stack. Prefill merges zero draft tokens
            # per row, so positions stay valid without remapping; decode
            # introduces no new images.
            h_per_dev = self.target.embed_tokens(
                batch.merged_tokens, batch.signal_buffers
            )
            h_per_dev = [
                merge_multimodal_embeddings(
                    inputs_embeds=h_d,
                    multimodal_embeddings=img_emb_d,
                    image_token_indices=img_idx_d,
                )
                for h_d, img_emb_d, img_idx_d in zip(
                    h_per_dev,
                    batch.vision_embeddings,
                    batch.vision_scatter_indices,
                    strict=True,
                )
            ]
            outputs = self.target._process_hidden_states(
                h_per_dev,
                batch.signal_buffers,
                batch.kv_collections,
                batch.return_n_logits,
                list(batch.merged_offsets_per_dev),
                batch.dist.host_merged_offsets,
                batch.dist.data_parallel_splits,
                batch.batch_context_lengths,
                batch.ep_inputs,
            )
        else:
            outputs = self.target(
                batch.merged_tokens,
                batch.signal_buffers,
                batch.kv_collections,
                batch.return_n_logits,
                batch.merged_offsets_per_dev,
                batch.dist.host_merged_offsets,
                batch.dist.data_parallel_splits,
                batch.batch_context_lengths,
                batch.ep_inputs,
            )
        return Verified(
            logits=outputs[1],
            hidden=captures_by_device(outputs[3:], batch.n_devs),
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        if self.target.ep_manager is None:
            return ()
        return self.target.ep_manager.input_types()


class Eagle3KimiK25Proposer(MLAProposer[_TargetHidden]):
    """The MLA-draft Eagle3 proposer."""

    split_prefix = "eagle3"
    carry_dim_names = CarryDimNames(prefix="draft_step", per_device=True)
    uses_thinking_phase = True
    # In decode mode ALL-hs == LAST-hs, and ALL returns per-device hidden
    # states directly, avoiding the LAST path's allgather.
    step_hidden_mode = ReturnHiddenStates.ALL
