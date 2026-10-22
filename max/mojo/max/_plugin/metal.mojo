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

from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.sys import llvm_intrinsic
from std.sys.info import CompilationTarget

from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)

from ._trait import MaxPlugin, MaxPluginHooks


@inline(.always)
def _barrier():
    # threadgroup_barrier(mem_flags::mem_threadgroup)
    llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(2), Int32(1))


@inline(.always)
def _syncwarp(mask: Int):
    # simdgroup_barrier(mem_flags::mem_none)
    llvm_intrinsic["llvm.air.simdgroup.barrier", NoneType](Int32(0), Int32(4))


struct MetalMaxPlugin(MaxPlugin):
    """Provides default MAX behavior for Metal targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["metal"]()
    comptime HookFor[target: CompilationTarget] = MetalMaxPluginHooks[target]


struct MetalMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp
