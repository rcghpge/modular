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

from std.ffi import _external_call_const
from std.math import inf, isnan, nan, remainder
from std.math.math import (
    _is_zero_bits,
    _narrow_f32_to_bf16_bits,
    _scale_down_bits,
    _scale_up_subnormal,
)
from std.sys import size_of
from std.testing import assert_equal, assert_false, assert_true, TestSuite
from std.utils.numerics import FPUtils, max_finite

comptime length = 1 << 12


def _libm[
    dtype: DType, //
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    comptime if dtype == .float64:
        return _external_call_const["remainder", Scalar[dtype]](x, y)
    elif dtype == .float32:
        return _external_call_const["remainderf", Scalar[dtype]](x, y)
    else:
        return _external_call_const["remainderf", Float32](
            x.cast[.float32](), y.cast[.float32]()
        ).cast[dtype]()


struct _Rng:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        # splitmix64
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def bits[dtype: DType](mut self) -> Scalar[dtype]:
        comptime nbits = 8 * size_of[dtype]()
        var v = self.next() >> UInt64(64 - nbits)
        return Scalar[dtype](from_bits=v.cast[FPUtils[dtype].integral_type]())

    def uniform(mut self) -> Float64:
        return Float64(self.next() >> 11) / Float64(1 << 53)


def _specials[dtype: DType]() -> List[Scalar[dtype]]:
    comptime IT = Scalar[FPUtils[dtype].integral_type]
    var big = max_finite[dtype]()
    var tiny = Scalar[dtype](from_bits=IT(1))
    var min_normal = Scalar[dtype](
        from_bits=IT(1 << FPUtils[dtype].mantissa_width())
    )
    return [
        0,
        -0.0,
        1,
        -1,
        0.5,
        1.5,
        2.5,
        -2.5,
        3,
        -7,
        0.75,
        10,
        big,
        -big,
        tiny,
        -tiny,
        min_normal,
        -min_normal,
        inf[dtype](),
        -inf[dtype](),
        nan[dtype](),
    ]


def _inputs[
    dtype: DType
](mode: Int, seed: UInt64) -> Tuple[List[Scalar[dtype]], List[Scalar[dtype]]]:
    """Special-value pairs, then `mode`-specific inputs.

    mode 0: random bit patterns for both operands.
    mode 1: x random bits, y a small integer (some non-integral, some huge).
    mode 2: x = (n + 0.5) * y ties, and x, y of nearby magnitude.
    mode 3: x within a few ulps of n * y for a quotient n of up to
    mantissa_width + 1 bits, where an inexact `/` misjudges trunc(x / y).
    """
    var rng = _Rng(seed)
    var sp = _specials[dtype]()
    var xs = List[Scalar[dtype]]()
    var ys = List[Scalar[dtype]]()
    for a in sp:
        for b in sp:
            xs.append(a)
            ys.append(b)
    var i = len(xs)
    while i < length:
        var x = rng.bits[dtype]()
        var y: Scalar[dtype]
        if mode == 0:
            y = rng.bits[dtype]()
        elif mode == 1:
            comptime span = 2 * (
                FPUtils[dtype].exponent_bias()
                + FPUtils[dtype].mantissa_width()
                + 4
            )
            var n = Float64(Int(rng.uniform() * Float64(span)) - span // 2)
            if i % 17 == 0:
                n += 0.5
            elif i % 29 == 0:
                n *= 1e6
            y = Scalar[dtype](n)
        elif mode == 2:
            y = rng.bits[dtype]()
            var n = Scalar[dtype](Int(rng.next() >> 54))
            if i % 2 == 0:
                x = (n + 0.5) * y
            else:
                x = y * Scalar[dtype](1 + rng.uniform() * 8)
        else:
            comptime IT = FPUtils[dtype].integral_type
            comptime qbits = FPUtils[dtype].mantissa_width() + 1
            y = rng.bits[dtype]()
            var n = Scalar[dtype](Int(rng.next() >> UInt64(64 - qbits)))
            var nudge = Scalar[IT](Int(rng.next() % 7) - 3)
            x = Scalar[dtype](from_bits=(n * y).to_bits().cast[IT]() + nudge)
        xs.append(x)
        ys.append(y)
        i += 1
    return (xs^, ys^)


def _run[dtype: DType, width: Int]() raises where dtype.is_floating_point():
    # remainder is exact, so it must match libm bit for bit, including the
    # sign of a zero result.
    for mode in range(4):
        var inputs = _inputs[dtype](mode, UInt64(1234 + mode))
        ref xs = inputs[0]
        ref ys = inputs[1]
        for base in range(0, length, width):
            var xv = SIMD[dtype, width]()
            var yv = SIMD[dtype, width]()
            comptime for j in range(width):
                xv[j] = xs[base + j]
                yv[j] = ys[base + j]
            var got = remainder(xv, yv)
            comptime for j in range(width):
                var x = xv[j]
                var y = yv[j]
                var expected = _libm(x, y)
                var ok: Bool
                if isnan(expected):
                    ok = isnan(got[j])
                else:
                    ok = got[j].to_bits() == expected.to_bits()
                if not ok:
                    raise Error(
                        t"remainder {dtype} ({x}, {y}) got {got[j]} expected"
                        t" {expected}"
                    )


def _f32_from_bits(bits: UInt64) -> Float32:
    return Float32(from_bits=bits.cast[DType.uint32]())


def test_is_zero_bits() raises:
    var rng = _Rng(7)
    for _ in range(512):
        var x = rng.bits[.float32]()
        var m = x.to_bits() & 0x7FFFFFFF
        assert_equal(_is_zero_bits(x), m == 0)
    assert_true(_is_zero_bits(Float32(0.0)))
    assert_true(_is_zero_bits(Float32(-0.0)))
    assert_false(_is_zero_bits(_f32_from_bits(1)))
    assert_false(_is_zero_bits(_f32_from_bits(0x80000001)))


def _check_scale_up[shift: Int]() raises:
    var rng = _Rng(11 + UInt64(shift))
    for i in range(512):
        var bits = rng.next() & 0x807FFFFF
        if i == 0:
            bits = 1
        elif i == 1:
            bits = 0x007FFFFF
        elif bits & 0x7FFFFF == 0:
            bits |= 1
        var x = _f32_from_bits(bits)
        var expected = (Float64(x) * Float64(2) ** Float64(shift)).cast[
            DType.float32
        ]()
        var got = _scale_up_subnormal[shift](x)
        assert_equal(got.to_bits(), expected.to_bits())


def test_scale_up_subnormal() raises:
    _check_scale_up[23]()
    _check_scale_up[46]()
    _check_scale_up[100]()


def _check_scale_down[shift: Int]() raises:
    var rng = _Rng(13 + UInt64(shift))
    for i in range(1024):
        var bits = rng.next() & 0xFFFFFFFF
        # Zero low bits make exact ties for the subnormal rounding common.
        bits &= ~((UInt64(1) << (rng.next() % 24)) - 1)
        var e = (bits >> 23) & 0xFF
        # Keep y finite and normal; exponents near 1 + shift hit the
        # subnormal range, the rest the normal and underflow-to-zero ranges.
        if i % 2 == 0:
            e = UInt64(max(1 + shift - 26, 1)) + rng.next() % 28
        elif e == 0 or e == 255:
            e = 100
        bits = (bits & 0x807FFFFF) | (e << 23)
        var y = _f32_from_bits(bits)
        var expected = (Float64(y) * Float64(2) ** Float64(-shift)).cast[
            DType.float32
        ]()
        var got = _scale_down_bits[shift](y)
        assert_equal(got.to_bits(), expected.to_bits())


def test_scale_down_bits() raises:
    _check_scale_down[1]()
    _check_scale_down[23]()
    _check_scale_down[46]()
    _check_scale_down[100]()


def test_narrow_f32_to_bf16_bits() raises:
    var rng = _Rng(17)
    for i in range(2048):
        var bits = rng.next() & 0xFFFFFFFF
        if i % 2 == 0:
            # Bring the exponent to the subnormal bfloat16 range and make
            # ties common.
            bits = (bits & 0x807FFFFF) | ((rng.next() % 2) << 23)
            bits &= ~((UInt64(1) << (rng.next() % 17)) - 1)
        if i % 8 == 1:
            bits = (bits & 0x807FFFFF) | 0x7F800000
        var x = _f32_from_bits(bits)
        var expected = x.cast[DType.bfloat16]()
        var got = _narrow_f32_to_bf16_bits(x)
        if isnan(expected):
            assert_true(isnan(got))
        else:
            assert_equal(got.to_bits(), expected.to_bits())


def test_float32() raises:
    _run[.float32, 1]()
    _run[.float32, 4]()


def test_float64() raises:
    _run[.float64, 1]()
    _run[.float64, 4]()


def test_float16() raises:
    _run[.float16, 1]()
    _run[.float16, 4]()


def test_bfloat16() raises:
    _run[.bfloat16, 1]()
    _run[.bfloat16, 4]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
