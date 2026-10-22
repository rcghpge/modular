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
from std.memory import bitcast
from std.sys import bit_width_of, llvm_intrinsic, size_of
from std.sys._assembly import inlined_assembly
from std.sys.info import CompilationTarget

from max.gpu.globals import WARP_SIZE
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


# TODO (#24457): support shuffles with width != 32
comptime _WIDTH_MASK = WARP_SIZE - 1

# shfl.sync.up.b32 prepares this mask differently from other shuffle intrinsics
comptime _WIDTH_MASK_SHUFFLE_UP = 0


@inline(.always)
def _shuffle[
    mnemonic: StringSlice,
    dtype: DType,
    simd_width: SIMDLength,
    *,
    WIDTH_MASK: Int32,
](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
    dtype, simd_width
]:
    comptime assert (
        dtype.is_half_float() or simd_width == 1
    ), "Unsupported simd_width"

    comptime if dtype == .float32:
        return llvm_intrinsic[
            "llvm.nvvm.shfl.sync." + mnemonic + ".f32", Scalar[dtype]
        ](Int32(mask), val, offset, WIDTH_MASK)
    elif dtype in (DType.int32, DType.uint32):
        return llvm_intrinsic[
            "llvm.nvvm.shfl.sync." + mnemonic + ".i32", Scalar[dtype]
        ](Int32(mask), val, offset, WIDTH_MASK)
    elif bit_width_of[dtype]() == 64:
        var val_bitcast = bitcast[.uint32, simd_width * 2](val)
        var val_half1, val_half2 = val_bitcast.deinterleave()
        var shuffle1 = _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
            mask, val_half1, offset
        )
        var shuffle2 = _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
            mask, val_half2, offset
        )
        var result = shuffle1.interleave(shuffle2)
        return bitcast[dtype, simd_width](result)
    elif dtype.is_half_float():
        comptime if simd_width == 1:
            # splat and recurse to meet 32 bitwidth requirements
            var splatted_val = SIMD[dtype, 2](val._refine[new_size=1]())
            return _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
                mask, splatted_val, offset
            )[0]
        else:
            # bitcast and recurse to use i32 intrinsic. Two half values fit
            # into an int32.
            var packed_val = bitcast[.int32, simd_width // 2](val)
            var result_packed = _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
                mask, packed_val, offset
            )
            return bitcast[dtype, simd_width](result_packed)
    elif dtype == .bool:
        comptime assert simd_width == 1, "unhandled simd width"
        return _shuffle[mnemonic, WIDTH_MASK=WIDTH_MASK](
            mask, val.cast[.int32](), offset
        ).cast[dtype]()

    else:
        comptime assert False, "unhandled shuffle dtype"


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

    @inline(.always)
    @staticmethod
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return _shuffle["idx", WIDTH_MASK=Int32(_WIDTH_MASK)](mask, val, offset)

    @inline(.always)
    @staticmethod
    def shuffle_up_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return _shuffle["up", WIDTH_MASK=_WIDTH_MASK_SHUFFLE_UP](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def shuffle_down_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return _shuffle["down", WIDTH_MASK=Int32(_WIDTH_MASK)](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def shuffle_xor_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        return _shuffle["bfly", WIDTH_MASK=Int32(_WIDTH_MASK)](
            mask, val, offset
        )

    @inline(.always)
    @staticmethod
    def vote_fn[ret_type: DType](val: Bool) -> Scalar[ret_type]:
        comptime assert ret_type == .uint32, "Unsupported return type"
        return rebind[Scalar[ret_type]](
            llvm_intrinsic[
                "llvm.nvvm.vote.ballot.sync",
                UInt32,
                UInt32,
                Bool,
                has_side_effect=False,
            ](0xFFFFFFFF, val).cast[.uint32]()
        )

    @inline(.always)
    @staticmethod
    def match_any_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        comptime assert (
            mask_type == .uint32
        ), "NVIDIA match_any returns a 32-bit mask (mask_type == DType.uint32)"
        comptime if size_of[dtype]() == 4:
            return rebind[Scalar[mask_type]](
                inlined_assembly[
                    "match.any.sync.b32 $0, $1, $2;",
                    UInt32,
                    constraints="=r,r,r",
                    has_side_effect=False,
                ](bits, UInt32(0xFFFFFFFF))
            )
        else:
            return rebind[Scalar[mask_type]](
                inlined_assembly[
                    "match.any.sync.b64 $0, $1, $2;",
                    UInt32,
                    constraints="=r,l,r",
                    has_side_effect=False,
                ](bits, UInt32(0xFFFFFFFF))
            )

    @inline(.always)
    @staticmethod
    def match_all_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        comptime assert (
            mask_type == .uint32
        ), "NVIDIA match_all returns a 32-bit mask (mask_type == DType.uint32)"
        # `match.all.sync` writes the membermask (all agree) or 0 into `$0` and
        # the agreement into a predicate `p` we do not need (the mask already
        # encodes it as non-zero / zero).
        comptime if size_of[dtype]() == 4:
            return rebind[Scalar[mask_type]](
                inlined_assembly[
                    "{ .reg .pred p; match.all.sync.b32 $0|p, $1, $2; }",
                    UInt32,
                    constraints="=r,r,r",
                    has_side_effect=False,
                ](bits, UInt32(0xFFFFFFFF))
            )
        else:
            return rebind[Scalar[mask_type]](
                inlined_assembly[
                    "{ .reg .pred p; match.all.sync.b64 $0|p, $1, $2; }",
                    UInt32,
                    constraints="=r,l,r",
                    has_side_effect=False,
                ](bits, UInt32(0xFFFFFFFF))
            )
