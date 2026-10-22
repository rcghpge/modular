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

from std.atomic import Ordering, fence
from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.sys import llvm_intrinsic
from std.sys.defines import get_defined_bool
from std.sys.info import CompilationTarget

from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)

from ._trait import MaxPlugin, MaxPluginHooks

comptime _USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM = get_defined_bool[
    "USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM", False
]()


@inline(.always)
def _barrier():
    comptime if _USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM:
        llvm_intrinsic["llvm.amdgcn.s.waitcnt", NoneType](Int32(0xC07F))
        llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
    else:
        fence[Ordering.RELEASE, scope="workgroup"]()
        llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
        fence[Ordering.ACQUIRE, scope="workgroup"]()


@inline(.always)
def _syncwarp(mask: Int):
    llvm_intrinsic["llvm.amdgcn.wave.barrier", NoneType]()


struct HipMaxPlugin(MaxPlugin):
    """Provides default MAX behavior for HIP targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["hip"]()
    comptime HookFor[target: CompilationTarget] = HipMaxPluginHooks[target]


struct HipMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp
