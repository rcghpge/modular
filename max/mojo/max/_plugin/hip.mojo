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
from std.memory import bitcast
from std.sys import bit_width_of, llvm_intrinsic, size_of
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


@inline(.always)
def _shuffle_amd_helper[
    dtype: DType, simd_width: SIMDLength
](dst_lane: UInt32, val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    comptime if size_of[SIMD[dtype, simd_width]]() == 4:
        # Handle int32, float32, float16x2, etc.
        var result_packed = llvm_intrinsic["llvm.amdgcn.ds.bpermute", Int32](
            dst_lane * 4, bitcast[.int32, 1](val)
        )
        return bitcast[dtype, simd_width](result_packed)
    else:
        comptime assert simd_width == 1, "Unsupported simd width"

        comptime if dtype == .bool:
            return _shuffle_amd_helper(dst_lane, val.cast[.int32]()).cast[
                dtype
            ]()
        elif bit_width_of[dtype]() == 16:
            var val_splatted = SIMD[dtype, 2](val._refine[new_size=1]())
            return _shuffle_amd_helper(dst_lane, val_splatted)[0]
        elif bit_width_of[dtype]() == 64:
            var val_bitcast = bitcast[.uint32, simd_width * 2](val)
            var val_half1, val_half2 = val_bitcast.deinterleave()
            var shuffle1 = _shuffle_amd_helper(dst_lane, val_half1)
            var shuffle2 = _shuffle_amd_helper(dst_lane, val_half2)
            var result = shuffle1.interleave(shuffle2)
            return bitcast[dtype, simd_width](result)
        else:
            comptime assert False, "unhandled shuffle dtype"


struct HipMaxPlugin(MaxPlugin):
    """Provides default MAX behavior for HIP targets."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["hip"]()
    comptime HookFor[target: CompilationTarget] = HipMaxPluginHooks[target]


struct HipMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = _barrier
    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = _syncwarp

    @inline(.always)
    @staticmethod
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        # FIXME: Set the EXECute mask register to the mask
        return _shuffle_amd_helper(offset, val)
