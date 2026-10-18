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
"""AMDGCN lowerings for the `std.math` functions `HIPPlugin` overrides."""

from std.collections.string.string_span import get_static_string
from std.sys import llvm_intrinsic


def _has_amdgcn_op[name: StaticString, dtype: DType]() -> Bool:
    """Returns whether `_amdgcn_unary_math` lowers `name` for `dtype`."""
    comptime if name == "rsqrt" or name == "recip":
        return True
    elif name == "exp2" or name == "log2":
        return dtype in (DType.float16, DType.float32)
    return False


def _amdgcn_intrinsic[name: StaticString, dtype: DType]() -> StaticString:
    comptime if name == "rsqrt":
        return get_static_string["llvm.amdgcn.rsq.", _amdgcn_suffix[dtype]()]()
    elif name == "recip":
        return get_static_string["llvm.amdgcn.rcp.", _amdgcn_suffix[dtype]()]()
    elif name == "exp2":
        return get_static_string["llvm.amdgcn.exp2.", _amdgcn_suffix[dtype]()]()
    else:
        comptime assert name == "log2", "no AMDGCN intrinsic for this function"
        return get_static_string["llvm.amdgcn.log.", _amdgcn_suffix[dtype]()]()


def _amdgcn_suffix[dtype: DType]() -> StaticString:
    comptime if dtype == .float16:
        return "f16"
    elif dtype == .float32:
        return "f32"
    else:
        comptime assert dtype == .float64, "no AMDGCN overload for this dtype"
        return "f64"


@inline(.always)
def _amdgcn_unary_math[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Lowers the `std.math` function `name` to its `llvm.amdgcn.*` intrinsic,
    one element at a time.

    The intrinsics have only f16, f32, and f64 overloads, so other dtypes
    round-trip through f32.
    """
    comptime if dtype not in (DType.float16, DType.float32, DType.float64):
        return _amdgcn_unary_math[name=name](x.cast[.float32]()).cast[dtype]()
    else:
        var res = SIMD[dtype, width]()

        comptime for i in range(width):
            res[i] = llvm_intrinsic[
                _amdgcn_intrinsic[name, dtype](),
                Scalar[dtype],
                has_side_effect=False,
            ](x[i])
        return res
