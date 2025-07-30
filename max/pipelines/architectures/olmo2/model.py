# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from max.driver import Tensor
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorValue
from max.graph.weights import Weights, WeightsAdapter

from ..llama3.model import LlamaModelBase
from .model_config import Olmo2Config
from .olmo2 import Olmo2


class Olmo2Model(LlamaModelBase):
    """OLMo2 pipeline model implementation."""

    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    """Normalization layer."""

    attention_bias: bool = False
    """Whether to use attention bias."""

    logits_postprocessor: Callable[[TensorValue], TensorValue] | None = None
    """Postprocessor for the logits."""

    state_dict: dict[str, Any]
    """Weights to load into the model."""

    def _build_graph(
        self,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
        session: Optional[InferenceSession] = None,
    ) -> Graph:
        """Override to use Olmo2Config and Olmo2 model instead of Llama3."""

        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        # Retrieve config using Olmo2Config instead of Llama3Config
        state_dict = self._get_state_dict(weights, adapter)
        model_config = Olmo2Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            logits_postprocessor=self.logits_postprocessor,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
            return_logits=self.return_logits,
        )

        # Get Graph Inputs
        graph_inputs = self.graph_inputs()

        # Build Graph - only single GPU for now
        if len(self.devices) > 1:
            raise NotImplementedError("Multi-GPU OLMo2 is not implemented yet")

        nn_model = Olmo2(model_config)

        # Load weights.
        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
        )

        self.state_dict = nn_model.state_dict()

        with Graph(
            "olmo2",
            input_types=graph_inputs,
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *kv_cache_inputs = (
                graph.inputs
            )
            outputs = nn_model(
                tokens.tensor,
                [inp.tensor for inp in kv_cache_inputs],
                input_row_offsets=input_row_offsets.tensor,
                return_n_logits=return_n_logits.tensor,
            )
            graph.output(*outputs)
            return graph
