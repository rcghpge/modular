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
"""Implements the gpu host package."""


from .info import default_accelerator, get_gpu_target

from ._legacy.constant_memory_mapping import ConstantMemoryMapping
from ._legacy.device_attribute import DeviceAttribute
from ._legacy.device_context import (
    CompletionFlag,
    DeviceBuffer,
    DeviceContext,
    DeviceContextArray,
    DeviceContextList,
    DeviceEvent,
    DeviceFunction,
    DeviceMulticastBuffer,
    DevicePointer,
    DeviceStream,
    HostBuffer,
)
from ._legacy.dim import Dim
from ._legacy.func_attribute import Attribute, FuncAttribute
from ._legacy.launch_attribute import LaunchAttribute
from ._legacy.device_graph import (
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceGraphCache,
    DeviceGraphInput,
    DeviceGraphInputKind,
    DeviceGraphNode,
    StableAddr,
)
