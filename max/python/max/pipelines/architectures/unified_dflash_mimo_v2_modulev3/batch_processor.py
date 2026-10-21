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
"""Inputs and batching for the ModuleV3 fused MiMo-V2 DFlash pipeline model.

The input ABI is the graph-API fused graph's, which Mach's block-spec
executor binds at every device count: a multi-device trace inserts the
mesh's signal buffers at the slot the graph-API signature declared, and a
single-device one declares its own there. The generic
``SpecDecodePipelineModel`` batching packs a composed graph's signature;
this package builds its fused module directly, so it owns its packing.
"""

from __future__ import annotations

from dataclasses import dataclass

from max import tree
from max.driver import Buffer
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.interfaces.batch_processor import (
    UnifiedSpecDecodeBatchProcessor,
)


@dataclass
class UnifiedDflashMiMoV2Inputs(UnifiedSpecDecodeInputs):
    """The fused graph's inputs: tokens, one row-offsets tensor, the logit
    count, a signal buffer per device, the ``{target, draft}`` KV tree, then
    the spec-decode tail."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    signal_buffers: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        buffers = (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs
                else ()
            ),
        )
        return buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=False
        )


class UnifiedDflashMiMoV2ModuleV3BatchProcessor(
    UnifiedSpecDecodeBatchProcessor[UnifiedDflashMiMoV2Inputs]
):
    """Ragged batching for the fused graph, with the per-device signal
    buffers its collectives need."""
