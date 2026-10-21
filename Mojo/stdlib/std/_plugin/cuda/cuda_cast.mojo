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
"""NVIDIA PTX lowerings of `SIMD.cast` conversions, for `CUDAPluginHooks`."""

from std.collections import OptionalReg
from std.memory import bitcast
from std.simd import _CastPluginHookFnType
from std.sys._assembly import inlined_assembly
from std.sys.info import CompilationTarget


comptime _CastFn[dst: DType] = OptionalReg[_CastPluginHookFnType[dst]]


def _nvptx_cast_fn[
    src: DType, dst: DType, target: CompilationTarget
]() -> _CastFn[dst]:
    """Returns the PTX lowering of a cast from `src` to `dst` on `target`, or
    `None` to keep the stdlib's own conversion."""
    # Match the dtypes before querying `target`: the arch queries instantiate
    # casts of their own, so querying them for every pair recurses into this
    # hook.
    comptime if src == DType.float32 and dst == DType.float8_e8m0fnu:
        comptime if target._is_sm_100x_or_newer():
            return comptime (_CastFn[dst](_float32_to_ue8m0[dst]))
    elif src == DType.float8_e8m0fnu and dst == DType.float32:
        comptime if target._is_sm_100x_or_newer():
            return comptime (_CastFn[dst](_ue8m0_to_float32[dst]))
    return None


@inline(.always)
def _float32_to_ue8m0[
    dtype: DType, width: SIMDLength, //, dst: DType
](x: SIMD[dtype, width]) -> SIMD[dst, width]:
    """Converts float32 to float8_e8m0fnu with `cvt.rp.ue8m0x2.f32`, which
    rounds toward +infinity like CUTLASS, two elements at a time."""
    comptime asm = "cvt.rp.ue8m0x2.f32 $0, $1, $2;"

    comptime if width > 1:
        var res = SIMD[dst, width]()

        comptime for i in range(0, width, 2):
            var f8x2_f32x2 = inlined_assembly[
                asm, UInt16, constraints="=h,f,f", has_side_effect=False
            ](x[i + 1], x[i])
            var ui8x2 = bitcast[dst, 2](f8x2_f32x2)
            res = res.insert[offset=i](ui8x2)
        return res
    else:
        var f8x2_f32x2 = inlined_assembly[
            asm, UInt16, constraints="=h,f,f", has_side_effect=False
        ](Float32(0.0), x[0])
        var ui8x2 = bitcast[dst, 2](f8x2_f32x2)
        return ui8x2[0]


@inline(.always)
def _ue8m0_to_float32[
    dtype: DType, width: SIMDLength, //, dst: DType
](x: SIMD[dtype, width]) -> SIMD[dst, width]:
    """Converts float8_e8m0fnu to float32 through bfloat16 with
    `cvt.rn.bf16x2.ue8m0x2`, two elements at a time."""
    comptime asm = "cvt.rn.bf16x2.ue8m0x2 $0, $1;"

    comptime if width > 1:
        var res = SIMD[dst, width]()

        comptime for i in range(0, width, 2):
            var ue8m0x2 = SIMD[.uint8, 2](
                bitcast[DType.uint8, 1](x[i]),
                bitcast[DType.uint8, 1](x[i + 1]),
            )
            var bf16x2 = inlined_assembly[
                asm, UInt32, constraints="=r,h", has_side_effect=False
            ](bitcast[DType.uint16, 1](ue8m0x2))
            var f32x2 = bitcast[DType.bfloat16, 2](bf16x2).cast[dst]()
            res = res.insert[offset=i](f32x2)
        return res
    else:
        var ue8m0x2 = SIMD[.uint8, 2](bitcast[DType.uint8, 1](x[0]), UInt8(0))
        var bf16x2 = inlined_assembly[
            asm, UInt32, constraints="=r,h", has_side_effect=False
        ](bitcast[DType.uint16, 1](ue8m0x2))
        var f32x2 = bitcast[DType.bfloat16, 2](bf16x2).cast[dst]()
        return f32x2[0]
