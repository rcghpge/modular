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

from std.builtin.dtype import _unsigned_integral_type_of
from std.math import inf, isinf, isnan, log10, nan, nextafter
from std.math.math import _call_libm
from std.memory import bitcast
from std.sys import bit_width_of
from std.testing import assert_equal, assert_true, TestSuite
from std.utils.numerics import max_finite

comptime length = 16384
comptime f32_min_subnormal_bits = UInt32(1)
comptime f32_max_subnormal_bits = UInt32(0x007FFFFF)
# Seed for the xorshift64 input sweeps.
comptime sweep_seed = UInt64(0x2545F4914F6CDD1D)
comptime f64_min_subnormal_bits = UInt64(1)
comptime f64_max_subnormal_bits = UInt64(0x000FFFFFFFFFFFFF)


def _xorshift64(mut state: UInt64) -> UInt64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return state


def _ulp_diff[dtype: DType, //](a: Scalar[dtype], b: Scalar[dtype]) -> Int:
    """Distance in ulps between `a` and `b` on their ordered bit patterns.

    Bit-identical values (two NaNs, or a zero or infinity of the same sign)
    and +0 vs -0 are 0 apart. Otherwise a NaN or an infinity on either side,
    or differing signs, are `Int.MAX` apart so that no tolerance accepts them.
    """
    comptime U = _unsigned_integral_type_of[dtype]()
    comptime sign_bit = Scalar[U](1) << Scalar[U](bit_width_of[dtype]() - 1)
    var a_bits = bitcast[U](a)
    var b_bits = bitcast[U](b)
    if a_bits == b_bits:
        return 0
    if isnan(a) or isnan(b) or isinf(a) or isinf(b):
        return Int.MAX
    var a_mag = Int(a_bits & ~sign_bit)
    var b_mag = Int(b_bits & ~sign_bit)
    if (a_bits ^ b_bits) & sign_bit != 0:
        return 0 if a_mag == 0 and b_mag == 0 else Int.MAX
    return abs(a_mag - b_mag)


def test_ulp_diff() raises:
    assert_equal(_ulp_diff(Float32(1), Float32(1)), 0)
    assert_equal(_ulp_diff(Float32(1), nextafter(Float32(1), Float32(2))), 1)
    assert_equal(_ulp_diff(Float32(-1), nextafter(Float32(-1), Float32(-2))), 1)
    assert_equal(_ulp_diff(Float64(1), Float64(1.0000000000000022)), 10)
    assert_equal(_ulp_diff(Float32(0), Float32(-0.0)), 0)
    assert_equal(_ulp_diff(inf[.float64](), inf[.float64]()), 0)
    assert_equal(_ulp_diff(nan[.float64](), nan[.float64]()), 0)
    # Opposite signs must never pass: the ordered-integer difference of
    # log10(0.01) = -2 and a wrong +2 overflows Int64 and used to pass.
    assert_equal(_ulp_diff(Float64(2), Float64(-2)), Int.MAX)
    assert_equal(_ulp_diff(Float32(1), Float32(-1)), Int.MAX)
    assert_equal(_ulp_diff(max_finite[.float32](), inf[.float32]()), Int.MAX)
    assert_equal(_ulp_diff(-max_finite[.float64](), -inf[.float64]()), Int.MAX)
    assert_equal(_ulp_diff(inf[.float32](), -inf[.float32]()), Int.MAX)
    assert_equal(_ulp_diff(nan[.float32](), Float32(1)), Int.MAX)
    assert_equal(_ulp_diff(Float32(0), -Float32(from_bits=Int32(1))), Int.MAX)


def _same[
    dtype: DType, //
](a: Scalar[dtype], b: Scalar[dtype]) -> Bool where dtype.is_floating_point():
    return (isnan(a) and isnan(b)) or a.to_bits() == b.to_bits()


def _log10_f64_inputs() -> List[Float64]:
    # 2^64 / golden ratio, the Fibonacci-hashing multiplier.
    comptime golden_ratio_u64 = UInt64(0x9E3779B97F4A7C15)
    comptime f64_min_normal_bits = UInt64(0x0010000000000000)
    comptime f64_max_finite_bits = UInt64(0x7FEFFFFFFFFFFFFF)
    var v = List[Float64](capacity=length)
    v.append(0.0)
    v.append(-0.0)
    v.append(1.0)
    v.append(-1.0)
    v.append(inf[.float64]())
    v.append(-inf[.float64]())
    v.append(nan[.float64]())
    v.append(bitcast[.float64](f64_min_subnormal_bits))
    v.append(bitcast[.float64](f64_max_subnormal_bits))
    v.append(bitcast[.float64](f64_min_normal_bits))
    v.append(bitcast[.float64](f64_max_finite_bits))
    var p: Float64 = 1.0
    for _ in range(309):
        v.append(p)
        p *= 10.0
    p = 0.1
    for _ in range(323):
        v.append(p)
        p /= 10.0
    # Subnormals: every bit position with a few mantissa tails.
    for i in range(52):
        for j in range(4):
            var bits = (UInt64(1) << UInt64(i)) | (
                (UInt64(j) * golden_ratio_u64) & ((UInt64(1) << UInt64(i)) - 1)
            )
            v.append(bitcast[.float64](bits))
    # Near 1, where log10 is small and cancellation matters.
    var one_bits = bitcast[.uint64](1.0)
    for i in range(1, 400):
        var d = Float64(i) * 1.0e-6
        v.append(1.0 + d)
        v.append(1.0 - d)
        v.append(bitcast[.float64](one_bits + UInt64(i)))
        v.append(bitcast[.float64](one_bits - UInt64(i)))
    for i in range(16):
        v.append(-Float64(i + 1) * 3.7)
    # Dense pseudo-random sweep over all positive normal bit patterns.
    var state = sweep_seed
    comptime normal_span = f64_max_finite_bits - f64_min_normal_bits
    while len(v) < length:
        v.append(
            bitcast[.float64](
                f64_min_normal_bits + _xorshift64(state) % normal_span
            )
        )
    return v^


def test_log10_float64() raises:
    # glibc's log10 is not correctly rounded and is up to 2 ULP from the true
    # value near 1 (e.g. log10(0.9999999999999691)), so allow 2 ULP vs libm.
    comptime max_ulp_log10 = 2
    # 10^n is exactly representable in float64 for n <= 22.
    comptime max_exact_pow10 = 22
    for x in _log10_f64_inputs():
        var expected = _call_libm["log10"](x)
        var got = log10(x)
        var d = _ulp_diff(got, expected)
        assert_true(
            d <= max_ulp_log10,
            String(t"log10({x}) = {got}, expected {expected} ({d} ulp)"),
        )
    var p: Float64 = 1.0
    for n in range(max_exact_pow10 + 1):
        assert_equal(log10(p), Float64(n))
        p *= 10.0
    # SIMD lanes take the same path as scalars.
    var v = SIMD[.float64, 4](0.0, 1.0e-310, 1000.0, -2.0)
    var r = log10(v)
    for i in range(4):
        assert_true(_same(r[i], log10(v[i])))


def _log10_f32_inputs() -> List[Float32]:
    comptime f32_min_normal_bits = UInt32(0x00800000)
    comptime f32_max_finite_bits = UInt32(0x7F7FFFFF)
    comptime f32_inf_bits = 0x7F800000
    var v: List[Float32] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        inf[.float32](),
        -inf[.float32](),
        nan[.float32](),
        bitcast[.float32](f32_min_subnormal_bits),
        bitcast[.float32](f32_max_subnormal_bits),
        bitcast[.float32](f32_min_normal_bits),
        bitcast[.float32](f32_max_finite_bits),
    ]
    v.reserve(length)
    var p: Float32 = 1.0
    for _ in range(39):
        v.append(p)
        p *= 10.0
    # Near 1, where log10 is small and cancellation matters.
    var one_bits = bitcast[.uint32](Float32(1.0))
    for i in range(1, 2048):
        v.append(bitcast[.float32](one_bits + UInt32(i)))
        v.append(bitcast[.float32](one_bits - UInt32(i)))
        var d = Float32(i) * 1.0e-4
        v.append(1.0 + d)
        v.append(1.0 - d)
    for i in range(16):
        v.append(-Float32(i + 1) * 3.7)
    # Pseudo-random sweep over all positive finite bit patterns.
    var state = sweep_seed
    while len(v) < length:
        v.append(
            bitcast[.float32](
                (_xorshift64(state) % f32_inf_bits).cast[.uint32]()
            )
        )
    return v^


def test_log10_float32() raises:
    # Against float64 libm rounded to float32; that reference is itself
    # within 1 ULP of the true value.
    for x in _log10_f32_inputs():
        var expected = _call_libm["log10"](x.cast[.float64]()).cast[.float32]()
        var got = log10(x)
        var d = _ulp_diff(got, expected)
        assert_true(
            d <= 1,
            String(t"log10({x}) = {got}, expected {expected} ({d} ulp)"),
        )
    var p: Float32 = 1.0
    for n in range(11):
        assert_equal(log10(p), Float32(n))
        p *= 10.0
    assert_equal(log10(Float16(100.0)), Float16(2.0))
    assert_equal(log10(BFloat16(0.0)), -inf[.bfloat16]())
    var v = SIMD[.float32, 4](0.0, 1.0e-40, 1000.0, -2.0)
    var r = log10(v)
    for i in range(4):
        assert_true(_same(r[i], log10(v[i])))


def test_log10_float16_exhaustive() raises:
    # Every float16 bit pattern against float64 libm rounded to float16.
    var max_d = 0
    for bits in range(1 << 16):
        var x = bitcast[.float16](UInt16(bits))
        var expected = _call_libm["log10"](x.cast[.float64]()).cast[.float16]()
        var got = log10(x)
        if isnan(expected):
            assert_true(isnan(got), String(t"log10({x}) = {got}, expected NaN"))
            continue
        var d = _ulp_diff(got, expected)
        max_d = max(max_d, d)
        assert_true(
            d <= 1,
            String(t"log10({x}) = {got}, expected {expected} ({d} ulp)"),
        )
    print("log10 float16 max ulp vs libm:", max_d)


def test_log10_subnormal() raises:
    """The log reduction reads a subnormal's exponent from its bits, so the
    smallest and largest subnormal inputs survive flush-to-zero targets. The
    GPU suite repeats this on device; it does not run on macOS."""
    var x32: List[Float32] = [
        bitcast[.float32](f32_min_subnormal_bits),
        bitcast[.float32](f32_max_subnormal_bits),
    ]
    for x in x32:
        var expected = _call_libm["log10"](x.cast[.float64]()).cast[.float32]()
        var got = log10(x)
        var d = _ulp_diff(got, expected)
        assert_true(
            d <= 1, String(t"log10({x}) = {got}, expected {expected} ({d} ulp)")
        )
    var x64: List[Float64] = [
        bitcast[.float64](f64_min_subnormal_bits),
        bitcast[.float64](f64_max_subnormal_bits),
    ]
    for x in x64:
        var expected = _call_libm["log10"](x)
        var got = log10(x)
        var d = _ulp_diff(got, expected)
        assert_true(
            d <= 1, String(t"log10({x}) = {got}, expected {expected} ({d} ulp)")
        )
    # A negative subnormal is still negative, not a zero.
    assert_true(isnan(log10(-bitcast[.float32](f32_min_subnormal_bits))))
    assert_true(isnan(log10(-bitcast[.float64](f64_min_subnormal_bits))))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
