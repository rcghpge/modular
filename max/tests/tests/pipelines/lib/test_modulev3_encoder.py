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
"""Pins how a ModuleV3 model without a KV cache loads its encoder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.nn import Module as ModuleV3
from max.experimental.nn import module_dataclass
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.weights import SafetensorWeights, WeightData
from max.pipelines.lib import (
    GraphRole,
    ModelInputs,
    ModelOutputs,
    ModuleV3PipelineModel,
)


@module_dataclass
class _Scale(ModuleV3[[Tensor], Tensor]):
    weight: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight


class _Model(ModuleV3PipelineModel[Any]):
    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        return SimpleNamespace(factor=2.0)

    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        return {
            "weight": np.from_dlpack(state_dict["weight"]) * model_config.factor
        }

    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        return DType.float32

    def _instantiate_module(self, model_config: Any) -> _Scale:
        return _Scale(Tensor.zeros([2], dtype=DType.float32, device=CPU()))

    def _get_compile_input_types(self, model_config: Any) -> tuple[Any, ...]:
        return (TensorType(DType.float32, [2], device=DeviceRef.CPU()),)

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        raise NotImplementedError


def _model() -> _Model:
    model = object.__new__(_Model)
    model.weights = SafetensorWeights([])
    model.adapter = lambda _: {
        "weight": WeightData.from_numpy(np.full(2, 3.0, np.float32), "weight")
    }
    return model


def test_the_encoder_is_named_after_its_module() -> None:
    model = _model()
    model._load_state_dict()

    specs = model.graph_specs(model._create_model_config({}))

    assert list(specs) == [GraphRole.ENCODER]
    assert specs[GraphRole.ENCODER].name == "_Scale"


def test_the_encoder_compiles_with_the_prepared_weights() -> None:
    model = _model()

    encoder = model._load_encoder(InferenceSession(devices=[CPU()]))

    assert encoder is model.compiled_modules[GraphRole.ENCODER]
    out = encoder(Buffer.from_numpy(np.ones(2, dtype=np.float32)))
    assert out.to_numpy().tolist() == [6.0, 6.0]
