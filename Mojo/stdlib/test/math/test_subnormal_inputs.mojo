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

# CPU coverage of normal and subnormal outputs, and of the integer-only
# helpers that keep subnormal results on targets that flush arithmetic.

from std.math import hypot, inf, isinf, isnan, nan
from std.math.math import _narrow_f32_to_bf16_bits, _scale_down_bits
from std.testing import TestSuite, assert_equal, assert_true

from test_utils import ulp_distance

comptime f32_min = Float32(from_bits=UInt32(1))  # 2^-149
comptime f32_max_sub = Float32(from_bits=UInt32(0x007FFFFF))
comptime f64_max_sub = Float64(from_bits=UInt64(0x000FFFFFFFFFFFFF))


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
