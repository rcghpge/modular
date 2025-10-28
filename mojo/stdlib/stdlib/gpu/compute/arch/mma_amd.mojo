# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""AMD CDNA Matrix Cores implementation for matrix multiply-accumulate operations.

This module provides MMA implementations for AMD CDNA2, CDNA3, and CDNA4 data
center GPUs using the MFMA (Matrix Fused Multiply-Add) instructions.

Reference: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
"""

from sys import llvm_intrinsic
from sys.info import _cdna_4_or_newer, _is_amd_rdna
from memory import bitcast

# Import helper functions from parent module
from ..mma import (
    _has_type,
    _has_shape,
    _unsupported_mma_op,
    get_amd_fp8_dtype,
    get_amd_bf8_dtype,
)

# Import RDNA implementation for consumer GPUs
from .mma_amd_rdna import _mma_wmma_rdna


@fieldwise_init
@register_passable("trivial")
struct _AMD_F8F6F4_MATRIX_FORMAT:
    """Represents the matrix format value to control the type and shape for the inputs
    of the llvm.amdgcn.mfma.scale.f8f6f4 intrinsics.
    """

    var _value: Int32
    alias float8_e4m3 = Self(0)
    alias float8_e5m2 = Self(1)
    alias float6_e2m3 = Self(2)
    alias float6_e3m2 = Self(3)
    alias float4_e2m1 = Self(4)

    fn __init__(out self, value: Int):
        self._value = value


@always_inline
fn _mma_amd[block_size: Int = 1](mut d: SIMD, a: SIMD, b: SIMD, c: SIMD):
    @parameter
    if _is_amd_rdna():
        # Use WMMA instructions for RDNA3+ consumer GPUs.
        _mma_wmma_rdna(d, a, b, c)
        return

    alias zero: UInt32 = 0

    # CDNA3 supports the FNUZ float8 dtypes, and CDNA4 supports the Open
    # Compute Project (OCP) float8 dtypes.
    alias fp8_dtype = get_amd_fp8_dtype()
    alias bf8_dtype = get_amd_bf8_dtype()

    @parameter
    fn _f8f6f4_intrinsic() -> SIMD[d.dtype, d.size]:
        constrained[_cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"]()

        alias intrinsic_name = "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4" if _has_shape[
            (32, 32, 4, 4)
        ](
            a.size, b.size, c.size, d.size
        ) else "llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4"

        @parameter
        fn _matrix_format[dtype: DType]() -> _AMD_F8F6F4_MATRIX_FORMAT:
            return (
                _AMD_F8F6F4_MATRIX_FORMAT.float8_e4m3 if dtype
                == fp8_dtype else _AMD_F8F6F4_MATRIX_FORMAT.float8_e5m2
            )

        return llvm_intrinsic[intrinsic_name, SIMD[d.dtype, d.size]](
            bitcast[DType.int32, 8](a),
            bitcast[DType.int32, 8](b),
            c,
            _matrix_format[a.dtype](),
            _matrix_format[b.dtype](),
            zero,
            zero,
            zero,
            zero,
        )

    # ===------------------------------------------------------------------===#
    # F16 = F16 * F16 + F16
    # ===------------------------------------------------------------------===#
    @parameter
    if _has_type[DType.float16](a.dtype, b.dtype, c.dtype, d.dtype):
        constrained[
            False, "Function mma F16 * F16 + F16 is unsupported by AMD GPUs."
        ]()

    # ===------------------------------------------------------------------===#
    # F32 = F16 * F16 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[4](
        a.size, b.size, c.size, d.size
    ):

        @parameter
        if block_size == 16:
            # Note: 4x4x4_16B (i.e., 16 blocks).
            d = llvm_intrinsic[
                "llvm.amdgcn.mfma.f32.4x4x4f16", SIMD[d.dtype, d.size]
            ](a, b, c, zero, zero, zero)
        else:
            d = llvm_intrinsic[
                "llvm.amdgcn.mfma.f32.16x16x16f16", SIMD[d.dtype, d.size]
            ](a, b, c, zero, zero, zero)
    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(4, 4, 16, 16)](
        a.size, b.size, c.size, d.size
    ):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.32x32x8f16", SIMD[d.dtype, d.size]
        ](a, b, c, zero, zero, zero)
    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 8, 4, 4)](
        a.size, b.size, c.size, d.size
    ):
        constrained[_cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"]()
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x32.f16", SIMD[d.dtype, d.size]
        ](a, b, c, zero, zero, zero)
    elif _has_type[
        (DType.float16, DType.float16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 8, 16, 16)](
        a.size, b.size, c.size, d.size
    ):
        constrained[_cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"]()
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.32x32x16.f16", SIMD[d.dtype, d.size]
        ](a, b, c, zero, zero, zero)

    # ===------------------------------------------------------------------===#
    # F32 = BF16 * BF16 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[4](
        a.size, b.size, c.size, d.size
    ):

        @parameter
        if block_size == 16:
            # Note: 4x4x4_16B (i.e., 16 blocks)
            d = llvm_intrinsic[
                "llvm.amdgcn.mfma.f32.4x4x4bf16.1k", SIMD[d.dtype, d.size]
            ](
                bitcast[DType.int16, 4](a),
                bitcast[DType.int16, 4](b),
                c,
                zero,
                zero,
                zero,
            )
        else:
            d = llvm_intrinsic[
                "llvm.amdgcn.mfma.f32.16x16x16bf16.1k", SIMD[d.dtype, d.size]
            ](
                bitcast[DType.int16, 4](a),
                bitcast[DType.int16, 4](b),
                c,
                zero,
                zero,
                zero,
            )
    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(4, 4, 16, 16)](
        a.size, b.size, c.size, d.size
    ):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.32x32x8bf16.1k", SIMD[d.dtype, d.size]
        ](
            bitcast[DType.int16, 4](a),
            bitcast[DType.int16, 4](b),
            c,
            zero,
            zero,
            zero,
        )
    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 8, 4, 4)](
        a.size, b.size, c.size, d.size
    ):
        constrained[_cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"]()
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x32.bf16", SIMD[d.dtype, d.size]
        ](a, b, c, zero, zero, zero)
    elif _has_type[
        (DType.bfloat16, DType.bfloat16, DType.float32, DType.float32)
    ](a.dtype, b.dtype, c.dtype, d.dtype) and _has_shape[(8, 8, 16, 16)](
        a.size, b.size, c.size, d.size
    ):
        constrained[_cdna_4_or_newer(), "MMA shape requires CDNA4 or newer"]()
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.32x32x16.bf16", SIMD[d.dtype, d.size]
        ](a, b, c, zero, zero, zero)

    # ===------------------------------------------------------------------===#
    # F32 = F32 * F32 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[DType.float32](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(1, 1, 4, 4)](a.size, b.size, c.size, d.size):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x4f32", SIMD[d.dtype, d.size]
        ](a, b, c, zero, zero, zero)

    # ===------------------------------------------------------------------===#
    # F32 = F8 * F8 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[(fp8_dtype, fp8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(8, 8, 4, 4)](a.size, b.size, c.size, d.size):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8", SIMD[d.dtype, d.size]
        ](
            bitcast[DType.int64, 1](a),
            bitcast[DType.int64, 1](b),
            c,
            zero,
            zero,
            zero,
        )
    elif _has_type[(fp8_dtype, fp8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(32, 32, 4, 4)](a.size, b.size, c.size, d.size):
        d = _f8f6f4_intrinsic()
    elif _has_type[(fp8_dtype, fp8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(32, 32, 16, 16)](a.size, b.size, c.size, d.size):
        d = _f8f6f4_intrinsic()

    # ===------------------------------------------------------------------===#
    # F32 = BF8 * BF8 + F32
    # ===------------------------------------------------------------------===#
    elif _has_type[(bf8_dtype, bf8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(8, 8, 4, 4)](a.size, b.size, c.size, d.size):
        d = llvm_intrinsic[
            "llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8", SIMD[d.dtype, d.size]
        ](
            bitcast[DType.int64, 1](a),
            bitcast[DType.int64, 1](b),
            c,
            zero,
            zero,
            zero,
        )
    elif _has_type[(bf8_dtype, bf8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(32, 32, 4, 4)](a.size, b.size, c.size, d.size):
        d = _f8f6f4_intrinsic()
    elif _has_type[(bf8_dtype, bf8_dtype, DType.float32, DType.float32)](
        a.dtype, b.dtype, c.dtype, d.dtype
    ) and _has_shape[(32, 32, 16, 16)](a.size, b.size, c.size, d.size):
        d = _f8f6f4_intrinsic()

    else:
        _unsupported_mma_op(d, a, b, c)
