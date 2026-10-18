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
"""NVIDIA PTX lowerings of `std.math` functions, for `CUDAPluginHooks`."""

from std.math.math import _call_ptx_intrinsic, _llvm_unary_fn
from std.sys.info import CompilationTarget, _is_sm_9x_or_newer


def _has_nvptx_op[
    name: StaticString, dtype: DType, target: CompilationTarget
]() -> Bool:
    """Returns whether `_nvptx_unary_math` lowers `name` for `dtype` on
    `target`.

    Narrower dtypes that are left out here reach the float32 lowering through
    the stdlib, which converts them to float32 and calls `name` again.
    """
    comptime if name == "sqrt":
        # NVIDIA has no approximate f64 sqrt (`sqrt.approx.d` does not exist).
        # The stdlib's `llvm.sqrt` lowers to the IEEE `sqrt.rn.f64` instead.
        return dtype != DType.float64
    elif name == "rsqrt" or name == "recip":
        return True
    elif name == "exp2":
        return dtype in (DType.float16, DType.float32) or (
            dtype == DType.bfloat16 and target._is_sm_9x_or_newer()
        )
    elif name == "tanh":
        return dtype in (DType.float16, DType.bfloat16, DType.float32)
    elif name == "log" or name == "log2":
        return dtype == DType.float32
    elif name == "sin" or name == "cos":
        # float64 is claimed only so `_nvptx_unary_math` rejects it, rather
        # than letting the stdlib fall back to `llvm.sin` and `llvm.cos`.
        return dtype in (DType.float32, DType.float64)
    return False


@inline(.always)
def _nvptx_unary_math[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Lowers the `std.math` function `name` to NVIDIA PTX."""
    comptime if name == "sqrt":
        return _sqrt(x)
    elif name == "rsqrt":
        return _rsqrt(x)
    elif name == "recip":
        return _recip(x)
    elif name == "exp2":
        return _exp2(x)
    elif name == "tanh":
        return _tanh(x)
    elif name == "log":
        comptime ln2 = 0.69314718055966295651160180568695068359375
        return ln2 * _log2(x)
    elif name == "log2":
        return _log2(x)
    elif name == "sin":
        comptime assert (
            dtype != DType.float64
        ), "DType.float64 is not supported for sin on NVIDIA GPU"
        return _call_ptx_intrinsic[
            instruction="sin.approx.ftz.f32", constraints="=f,f"
        ](x)
    else:
        comptime assert name == "cos", "no NVIDIA lowering for this function"
        comptime assert (
            dtype != DType.float64
        ), "DType.float64 is not supported for cos on NVIDIA GPU"
        return _call_ptx_intrinsic[
            instruction="cos.approx.ftz.f32", constraints="=f,f"
        ](x)


@inline(.always)
def _sqrt_nvvm(x: SIMD, out res: type_of(x)):
    comptime assert x.dtype == .float32, "must be DType.float32"
    res = {}

    comptime for i in range(x.length):
        res[i] = _llvm_unary_fn["llvm.nvvm.sqrt.approx.ftz.f"](x[i])


@inline(.always)
def _sqrt(x: SIMD) -> type_of(x):
    comptime if x.dtype in (DType.float16, DType.bfloat16):
        return _sqrt_nvvm(x.cast[.float32]()).cast[x.dtype]()
    return _sqrt_nvvm(x)


@inline(.always)
def _rsqrt_nvvm(x: SIMD, out res: type_of(x)):
    comptime assert x.dtype in (
        DType.float32,
        DType.float64,
    ), "must be f32 or f64 type"

    comptime instruction = "llvm.nvvm.rsqrt.approx.ftz.f" if x.dtype == .float32 else "llvm.nvvm.rsqrt.approx.d"
    res = {}

    comptime for i in range(x.length):
        res[i] = _llvm_unary_fn[instruction](x[i])


@inline(.always)
def _rsqrt(x: SIMD) -> type_of(x):
    comptime if x.dtype in (DType.float16, DType.bfloat16):
        return _rsqrt_nvvm(x.cast[.float32]()).cast[x.dtype]()
    return _rsqrt_nvvm(x)


@inline(.always)
def _recip_nvvm(x: SIMD, out res: type_of(x)):
    comptime assert x.dtype in (
        DType.float32,
        DType.float64,
    ), "must be f32 or f64 type"

    comptime instruction = "llvm.nvvm.rcp.approx.ftz.f" if x.dtype == .float32 else "llvm.nvvm.rcp.approx.ftz.d"
    res = {}

    comptime for i in range(x.length):
        res[i] = _llvm_unary_fn[instruction](x[i])


@inline(.always)
def _recip(x: SIMD) -> type_of(x):
    comptime if x.dtype in (DType.float16, DType.bfloat16):
        return _recip_nvvm(x.cast[.float32]()).cast[x.dtype]()
    return _recip_nvvm(x)


@inline(.always)
def _exp2(x: SIMD) -> type_of(x):
    comptime if x.dtype == .float16:
        comptime if _is_sm_9x_or_newer():
            return _call_ptx_intrinsic[
                scalar_instruction="ex2.approx.f16",
                vector2_instruction="ex2.approx.f16x2",
                scalar_constraints="=h,h",
                vector_constraints="=r,r",
            ](x)
        else:
            return _call_ptx_intrinsic[
                instruction="ex2.approx.f16", constraints="=h,h"
            ](x)
    elif x.dtype == .bfloat16:
        comptime assert _is_sm_9x_or_newer(), "ex2.approx.bf16 needs sm_90"
        return _call_ptx_intrinsic[
            scalar_instruction="ex2.approx.ftz.bf16",
            vector2_instruction="ex2.approx.ftz.bf16x2",
            scalar_constraints="=h,h",
            vector_constraints="=r,r",
        ](x)
    return _call_ptx_intrinsic[
        instruction="ex2.approx.ftz.f32", constraints="=f,f"
    ](x)


@inline(.always)
def _log2(x: SIMD) -> type_of(x):
    return _call_ptx_intrinsic[
        instruction="lg2.approx.ftz.f32", constraints="=f,f"
    ](x)


@inline(.always)
def _tanh(x: SIMD) -> type_of(x):
    comptime if x.dtype == .float16:
        return _call_ptx_intrinsic[
            scalar_instruction="tanh.approx.f16",
            vector2_instruction="tanh.approx.f16x2",
            scalar_constraints="=h,h",
            vector_constraints="=r,r",
        ](x)
    elif x.dtype == .bfloat16:
        comptime if _is_sm_9x_or_newer():
            return _call_ptx_intrinsic[
                scalar_instruction="tanh.approx.bf16",
                vector2_instruction="tanh.approx.bf16x2",
                scalar_constraints="=h,h",
                vector_constraints="=r,r",
            ](x)
        else:
            return _call_ptx_intrinsic[
                instruction="tanh.approx.f32", constraints="=f,f"
            ](x.cast[.float32]()).cast[x.dtype]()
    return _call_ptx_intrinsic[
        instruction="tanh.approx.f32", constraints="=f,f"
    ](x)
