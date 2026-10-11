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

# Subnormal inputs whose results are normal. The float32 `cbrt` path reads
# the exponent from the integer significand, so no retained result depends
# on floating-point arithmetic on the subnormal operand. This also works on
# targets that flush subnormal operands to zero (Apple GPUs).

from std.math import cbrt, inf, isfinite, nan
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

comptime f32_min = Float32(from_bits=UInt32(1))  # 2^-149
comptime f32_max_sub = Float32(from_bits=UInt32(0x007FFFFF))
comptime f64_min = Float64(from_bits=UInt64(1))  # 2^-1074

# Subnormal divisors and dividends on the host, where arithmetic is IEEE and
# every result is exact. `remainder` reads a subnormal divisor through its
# integer significand, so on a target that flushes subnormals (Apple GPUs in
# float32) the zero results below still hold and normal results stay exact;
# the subnormal results are what such a target flushes to zero.

from std.math import remainder
from std.testing import TestSuite, assert_equal, assert_true

comptime f32_min_normal = Float32(from_bits=UInt32(0x00800000))  # 2^-126


def test_remainder_subnormal() raises:
    # Every multiple of the smallest subnormal reduces to a signed zero.
    assert_true(remainder(Float32(1.0), f32_min).to_bits() == 0)
    assert_true(remainder(-f32_max_sub, f32_min).to_bits() == 0x80000000)
    assert_equal(remainder(Float32(8.0), f32_min), Float32(0.0))
    assert_equal(
        remainder(Float32(1.1754943508222875e-38), f32_min * 3), -f32_min
    )
    assert_equal(remainder(f32_max_sub, f32_min * 7), f32_min * 3)
    assert_equal(remainder(f32_min, f32_min * 3), f32_min)
    assert_true(remainder(Float64(1.0), f64_min).to_bits() == 0)
    assert_equal(
        remainder(Float64(2.2250738585072014e-308), f64_min * 3), f64_min
    )
    var v = remainder(
        SIMD[.float32, 4](1.0, 7.0, f32_max_sub, -3.0),
        SIMD[.float32, 4](f32_min, 3.0, f32_min * 7, f32_min),
    )
    assert_equal(v, SIMD[.float32, 4](0.0, 1.0, f32_min * 3, -0.0))


def test_remainder_subnormal_results() raises:
    # Subnormal results from subnormal and from normal inputs; the last pair
    # is two normal inputs one ulp apart. 2^149 mod (2^23 - 1) = 2^11.
    assert_true(remainder(f32_min, f32_min * 3).to_bits() == 1)
    assert_true(remainder(Float32(1.0), f32_max_sub).to_bits() == 0x800)
    assert_true(
        remainder(
            Float32(from_bits=UInt32(0x00800001)), f32_min_normal
        ).to_bits()
        == 1
    )


def test_remainder_normal_results_near_min_normal() raises:
    # Normal results whose reduction runs close to the subnormal boundary,
    # where `p * 0.5` is inexact and the fma residues are smallest.
    assert_equal(
        remainder(f32_min_normal * 3, f32_min_normal * 2), -f32_min_normal
    )
    assert_equal(
        remainder(f32_min_normal * 5, f32_min_normal * 2), f32_min_normal
    )
    assert_equal(
        remainder(f32_min_normal * 7, f32_min_normal * 4), -f32_min_normal
    )
    # 2^26 = 3 * 22369621 + 1, so 2^-100 = n * (3 * 2^-126) + 2^-126.
    assert_equal(
        remainder(Float32(from_bits=UInt32(0x0D800000)), f32_min_normal * 3),
        f32_min_normal,
    )
    assert_equal(remainder(Float32(1.0), f32_min_normal * 3), f32_min_normal)
    var v = remainder(
        SIMD[.float32, 4](f32_min_normal * 3, 1.0, f32_min_normal * 5, -1.0),
        SIMD[.float32, 4](
            f32_min_normal * 2,
            f32_min_normal * 3,
            f32_min_normal * 2,
            f32_min_normal * 3,
        ),
    )
    assert_equal(
        v,
        SIMD[.float32, 4](
            -f32_min_normal, f32_min_normal, f32_min_normal, -f32_min_normal
        ),
    )


# CPU coverage of normal and subnormal outputs, and of the integer-only
# helpers that keep subnormal results on targets that flush arithmetic.

from std.math import hypot, inf, isinf, isnan, nan
from std.math.math import _narrow_f32_to_bf16_bits, _scale_down_bits

from test_utils import ulp_distance

comptime f64_max_sub = Float64(from_bits=UInt64(0x000FFFFFFFFFFFFF))


def _check_cbrt[
    dtype: DType
](got: Scalar[dtype], want: Scalar[dtype], name: String) raises:
    comptime assert dtype in (DType.float32, DType.float64)
    assert_true(
        isfinite(got) and isfinite(want),
        String(t"{name}: expected finite values, got {got} and {want}"),
    )
    comptime sign = UInt64(1) << UInt64(63 if dtype == .float64 else 31)
    var got_bits = got.to_bits[.uint64]()
    var want_bits = want.to_bits[.uint64]()
    assert_equal(
        got_bits & sign,
        want_bits & sign,
        String(t"{name}: wrong sign for {got}, expected {want}"),
    )
    var d = (
        got_bits - want_bits if got_bits >= want_bits else want_bits - got_bits
    )
    assert_true(d <= 1, String(t"{name} = {got}, expected {want} ({d} ulp)"))


def test_check() raises:
    # These bit patterns overflow a signed ULP distance to Int.MIN.
    comptime want = Float64(from_bits=UInt64(0xAAA428A2F98D728A))
    comptime bad = Float64(from_bits=UInt64(0x555BD75D06728D76))
    with assert_raises(contains="wrong sign"):
        _check_cbrt(bad, want, "cbrt(-max_sub)")
    _check_cbrt(want, want, "exact")
    _check_cbrt(Float64(from_bits=UInt64(0xAAA428A2F98D728B)), want, "one ulp")
    with assert_raises(contains="ulp"):
        _check_cbrt(
            Float64(from_bits=UInt64(0xAAA428A2F98D728C)), want, "two ulp"
        )

    comptime for dtype in [DType.float32, DType.float64]:
        with assert_raises(contains="wrong sign"):
            _check_cbrt(Scalar[dtype](-4.0), Scalar[dtype](1.0), "wrong sign")
        with assert_raises(contains="expected finite"):
            _check_cbrt(
                inf[dtype](), Scalar[dtype].MAX_FINITE, "infinite result"
            )
        with assert_raises(contains="expected finite"):
            _check_cbrt(
                Scalar[dtype].MAX_FINITE, inf[dtype](), "infinite reference"
            )
        with assert_raises(contains="expected finite"):
            _check_cbrt(nan[dtype](), Scalar[dtype](1.0), "NaN result")
        with assert_raises(contains="expected finite"):
            _check_cbrt(Scalar[dtype](1.0), nan[dtype](), "NaN reference")


def test_cbrt_subnormal() raises:
    _check_cbrt(cbrt(f32_min), 1.119034708375443e-15, "cbrt(2^-149)")
    # 2^-147 is a perfect cube.
    assert_equal(cbrt(f32_min * 4), Float32(1.7763568394002505e-15))
    _check_cbrt(cbrt(-f32_max_sub), -2.273736618907049e-13, "cbrt(-max_sub)")
    assert_equal(cbrt(f64_min), 1.7031839360032603e-108)
    _check_cbrt(cbrt(-f64_max_sub), -2.8126442852362615e-103, "cbrt(-max_sub)")
    var v = cbrt(SIMD[.float32, 4](f32_min, -f32_max_sub, 8.0, f32_min * 4))
    assert_equal(v[0], cbrt(f32_min))
    assert_equal(v[1], cbrt(-f32_max_sub))
    assert_equal(v[2], 2.0)
    assert_equal(v[3], cbrt(f32_min * 4))


def _check[
    dtype: DType
](got: Scalar[dtype], want: Scalar[dtype], name: String) raises:
    assert_true(not isnan(got) and not isinf(got))
    assert_true(got._to_bits_signed() >= 0)
    var d = ulp_distance(got, want)
    assert_true(d <= 1, String(t"{name} = {got}, expected {want} ({d} ulp)"))


def test_hypot_subnormal() raises:
    # Both subnormal, normal result.
    _check(
        hypot(f32_max_sub, f32_max_sub),
        1.6623998848452245e-38,
        "hypot(max_sub, max_sub)",
    )
    # Normal and subnormal operands of comparable size.
    _check(
        hypot(Float32(1.1754943508222875e-38), Float32(1.1571272515906893e-38)),
        1.6494636579419635e-38,
        "hypot(2^-126, 0.98 * 2^-126)",
    )
    assert_equal(hypot(f32_min, Float32(1.0)), 1.0)
    assert_equal(hypot(f32_min, Float32(0.0)), f32_min)
    _check(
        hypot(f64_max_sub, f64_max_sub),
        3.1467296279827165e-308,
        "hypot(max_sub, max_sub)",
    )


def test_hypot_subnormal_outputs() raises:
    var half_min = Float32(from_bits=Int32(0x00400000))
    assert_equal(Int(hypot(half_min, half_min)._to_bits_signed()), 0x005A827A)
    var bf16_half_min = BFloat16(from_bits=Int16(0x0040))
    assert_equal(
        Int(hypot(bf16_half_min, bf16_half_min)._to_bits_signed()), 0x005B
    )
    assert_equal(hypot(bf16_half_min, BFloat16(0)), bf16_half_min)
    assert_equal(
        Int(
            hypot(
                BFloat16(from_bits=Int16(0x007F)),
                BFloat16(from_bits=Int16(0x000C)),
            )._to_bits_signed()
        ),
        0x0080,
    )

    var xs = SIMD[.float32, 4](
        half_min, f32_max_sub, nan[.float32](), inf[.float32]()
    )
    var ys = SIMD[.float32, 4](
        half_min, f32_max_sub, Float32(1), nan[.float32]()
    )
    var got = hypot(xs, ys)
    assert_equal(Int(got[0]._to_bits_signed()), 0x005A827A)
    _check(got[1], Float32(from_bits=Int32(0x00B504F2)), "hypot SIMD normal")
    assert_true(isnan(got[2]))
    assert_equal(got[3], inf[.float32]())


def _scale_down_reference(y: Float32, shift: Int) -> Float32:
    """Rounds `y * 2^-shift` once, through an exact float64 product."""
    var scale = Float64(from_bits=UInt64(1023 - shift) << 52)
    return (Float64(y) * scale).cast[.float32]()


def _check_scale_down[shift: Int](y: Float32) raises:
    var got = _scale_down_bits[shift](SIMD[.float32, 1](y))[0]
    var want = _scale_down_reference(y, shift)
    assert_equal(
        Int(got._to_bits_signed()),
        Int(want._to_bits_signed()),
        String(t"_scale_down_bits[{shift}]({y})"),
    )


def test_scale_down_bits() raises:
    var state = UInt32(12345)
    # Normal inputs of several exponents with random mantissas, and mantissas
    # that sit exactly on a rounding tie for each subnormal shift amount.
    for exp in range(120, 140):
        for trial in range(64):
            state = state * 1664525 + 1013904223
            var mant = state >> 9
            if trial < 25:
                mant = (mant >> UInt32(trial + 1) << UInt32(trial + 1)) | (
                    UInt32(1) << UInt32(trial)
                )
                mant &= 0x7FFFFF
            var y = Float32(from_bits=(UInt32(exp) << 23) | mant)
            _check_scale_down[126](y)
            _check_scale_down[126](-y)
            _check_scale_down[130](y)
            _check_scale_down[100](y)
    _check_scale_down[126](Float32(0))
    _check_scale_down[126](Float32(1))
    _check_scale_down[126](Float32(from_bits=UInt32(0x00800000)))


def test_narrow_f32_to_bf16_bits() raises:
    var state = UInt32(777)
    for i in range(512):
        state = state * 1664525 + 1013904223
        var bits = state
        if i % 4 == 0:
            # Subnormal float32 inputs.
            bits &= 0x807FFFFF
        elif i % 4 == 1:
            # Exact ties between two bfloat16 values.
            bits = (bits & 0xFFFF0000) | 0x8000
        var x = Float32(from_bits=bits)
        if isnan(x):
            continue
        var got = _narrow_f32_to_bf16_bits(SIMD[.float32, 1](x))[0]
        assert_equal(
            Int(got._to_bits_signed()),
            Int(x.cast[.bfloat16]()._to_bits_signed()),
            String(t"narrow({x})"),
        )
    for x in [inf[.float32](), -inf[.float32](), Float32(0), Float32(-0.0)]:
        var got = _narrow_f32_to_bf16_bits(SIMD[.float32, 1](x))[0]
        assert_equal(
            Int(got._to_bits_signed()),
            Int(x.cast[.bfloat16]()._to_bits_signed()),
        )
    assert_true(
        isnan(_narrow_f32_to_bf16_bits(SIMD[.float32, 1](nan[.float32]()))[0])
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
