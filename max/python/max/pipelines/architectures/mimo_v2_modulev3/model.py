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
"""MiMo-V2.6-Flash pipeline model on ModuleV3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from max import tree
from max.driver import Buffer
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import default_device
from max.nn.kv_cache import KVCacheInputs
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    ModelInputs,
    ModuleV3PipelineModelWithKVCache,
)
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessor,
    ModuleV3SingleReplicaBatchProcessor,
)

from ..mimo_v2.model import text_model_weights
from .mimo_v2 import MiMoV2
from .model_config import MiMoV2Config


@dataclass
class MiMoV2Inputs(ModelInputs):
    """A class representing inputs for the MiMo-V2 ModuleV3 model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """The graph's inputs, in the order it declares them.

        ModuleV3's ABI is ``(tokens, return_n_logits, input_row_offsets,
        *kv)``; the KV leaves follow in the same order the symbolic inputs
        declare them.
        """
        return (
            self.tokens,
            self.return_n_logits,
            self.input_row_offsets,
            *(
                tree.leaves(self.kv_cache_inputs)
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.lora_buffers,
        )


class MiMoV2ModuleV3BatchProcessor(
    ModuleV3SingleReplicaBatchProcessor[TextContext, MiMoV2Inputs]
):
    """Ragged batching for MiMo-V2 ModuleV3 models (single replica, no
    signals)."""

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer],
    ) -> MiMoV2Inputs:
        return MiMoV2Inputs(
            tokens=tokens,
            input_row_offsets=input_row_offsets,
            return_n_logits=return_n_logits,
            kv_cache_inputs=kv_cache_inputs,
        )


class MiMoV2ModuleV3Model(
    ModuleV3PipelineModelWithKVCache[TextContext],
):
    """Pipeline model for MiMo-V2.6-Flash on ModuleV3.

    The graph takes ModuleV3's inputs: tokens, the logit count and row
    offsets on the first device, then the ``{sliding_attention,
    full_attention}`` KV tree, one collection per device.
    """

    model_config_cls: ClassVar[type[Any]] = MiMoV2Config
    # The lib root's own width: the fused spec model subclass binds a batch
    # processor from a different family here, which a concrete declaration
    # would make an override violation.
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]]] = (
        MiMoV2ModuleV3BatchProcessor
    )

    def _create_model_config(self, state_dict: dict[str, Any]) -> MiMoV2Config:
        del state_dict
        config = self.arch_config_as(MiMoV2Config)
        config.return_logits = self.return_logits
        config.return_hidden_states = self.return_hidden_states
        return config

    def _load_state_dict(self) -> dict[str, Any]:
        """Adapts the text model's tensors, :func:`text_model_weights`.

        The adapter must see exactly those to prove every one is consumed or
        ignored.
        """
        assert self.adapter is not None
        self._checkpoint = self.adapter(
            text_model_weights(
                self.pipeline_config.model.huggingface_weight_repo,
                self.weights,
            ),
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
        )
        return self._checkpoint

    def _instantiate_module(self, model_config: MiMoV2Config) -> MiMoV2:
        n_devices = len(self.devices)
        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))
        with default_device(mesh):
            return MiMoV2(model_config)
