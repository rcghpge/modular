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

from max.gpu.memory.memory import (
    _AsyncCopyGroupPluginHookFnType,
    _AsyncCopyWaitGroupPluginHookFnType,
)
from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)

from ._trait import MaxPlugin, MaxPluginHooks


@inline(.always)
def _barrier():
    _ = __mlir_op.`nvvm.barrier`[
        _properties=__mlir_attr.`{operandSegmentSizes = array<i32: 0, 0>}`,
    ]()


@inline(.always)
def _syncwarp(mask: Int):
    __mlir_op.`nvvm.bar.warp.sync`(
        __mlir_op.`index.casts`[_type=__mlir_type.i32](mask.__mlir_index__())
    )


@inline(.always)
def _async_copy_commit_group():
    llvm_intrinsic["llvm.nvvm.cp.async.commit.group", NoneType]()


@inline(.always)
def _async_copy_wait_group(n: Int32):
    llvm_intrinsic["llvm.nvvm.cp.async.wait.group", NoneType](n)


@inline(.always)
def _async_copy_wait_all():
    llvm_intrinsic["llvm.nvvm.cp.async.wait.all", NoneType]()


struct CudaMaxPlugin(MaxPlugin):
    """Provides default MAX behavior for CUDA targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["cuda"]()
    comptime HookFor[target: CompilationTarget] = CudaMaxPluginHooks[target]


struct CudaMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp
    comptime async_copy_commit_group_fn: OptionalReg[
        _AsyncCopyGroupPluginHookFnType
    ] = _async_copy_commit_group
    comptime async_copy_wait_group_fn: OptionalReg[
        _AsyncCopyWaitGroupPluginHookFnType
    ] = _async_copy_wait_group
    comptime async_copy_wait_all_fn: OptionalReg[
        _AsyncCopyGroupPluginHookFnType
    ] = _async_copy_wait_all
