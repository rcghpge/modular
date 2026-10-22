# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
#
# Kernel stubs whose stable-tensor arguments (`StableTensor` /
# `StableVariadicTensors`) sit where the kernel verifier forbids them. Each
# violates exactly one rule; `mogg-verify-kernels` must reject every one. Kept
# apart from the valid testing kernels because the package as a whole fails
# verification.
#
from max.gpu.host import DeviceContext, DeviceGraphBuilder

import extensibility
from extensibility import (
    InputTensor,
    InputVariadicTensors,
    OutputTensor,
    StableInputVariadicTensors,
    StableTensor,
    IOSpec,
)


# A stable argument on `execute`: only `build_graph` receives slot addresses.
@extensibility.register("stable_on_execute")
struct StableOnExecute:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        inputs: StableInputVariadicTensors[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass


# A stable output on `build_graph`: outputs are written through live
# addresses.
@extensibility.register("stable_output_on_build_graph")
struct StableOutputOnBuildGraph:
    @staticmethod
    def execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ):
        pass

    @staticmethod
    def build_graph[
        dtype: DType, rank: Int
    ](
        output: StableTensor[
            io_spec=IOSpec.Output, dtype=dtype, rank=rank, ...
        ],
        x: InputTensor[dtype=dtype, rank=rank, ...],
        builder: DeviceGraphBuilder,
    ):
        pass
