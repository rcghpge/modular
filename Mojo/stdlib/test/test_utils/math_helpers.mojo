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
"""Provides mathematical helper functions for numerical testing."""


from std.builtin.dtype import _integral_type_of
from std.math import isinf, isnan
from std.memory import bitcast
from std.sys import bit_width_of


def ulp_distance[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Int:
    """Computes the distance between two values in ULPs.

    ULP (Unit in the Last Place) distance measures how many representable
    floating-point values exist between two numbers. The distance is the
    unsigned difference of the magnitude encodings, so it is only defined for
    values of the same class and sign. Everything else compares as maximally
    far apart, which keeps a tolerance check such as `ulp_distance(got, want)
    <= n` from accepting a wrong sign, NaN, or infinity.

    Parameters:
        dtype: The data type.

    Args:
        a: First value.
        b: Second value.

    Returns:
        0 for any two NaNs. `Int.MAX` for a NaN against a non-NaN, an infinity
        against a finite value, or operands of opposite sign (including `+0.0`
        against `-0.0`). Otherwise the distance between the magnitudes.
    """
    comptime if dtype.is_floating_point():
        if isnan(a) or isnan(b):
            return 0 if isnan(a) and isnan(b) else Int.MAX
        if isinf(a) != isinf(b):
            return Int.MAX

    comptime T = _integral_type_of[dtype]()
    # widen to Int first to avoid overflow
    var a_int = Int(bitcast[T](a))
    var b_int = Int(bitcast[T](b))
    if (a_int < 0) != (b_int < 0):
        return Int.MAX
    # Clear the sign bit; Int.MAX >> (64 - w) keeps the low w - 1 bits.
    comptime magnitude_mask = Int.MAX >> (64 - bit_width_of[dtype]())
    return abs((a_int & magnitude_mask) - (b_int & magnitude_mask))
