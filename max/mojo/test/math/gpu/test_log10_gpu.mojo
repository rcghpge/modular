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
from std.math import inf, isinf, isnan, log10, nan
from std.math.math import _call_libm
from std.memory import bitcast
from std.sys import bit_width_of, default_accelerator, simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from std.testing import assert_equal, assert_true, TestSuite

from std.utils.coord import Coord

comptime length = 16384
comptime f64_min_subnormal_bits: UInt64 = 1
comptime f64_max_subnormal_bits: UInt64 = 0x000F_FFFF_FFFF_FFFF
comptime f32_min_subnormal_bits: UInt32 = 1
comptime f32_max_subnormal_bits: UInt32 = 0x007F_FFFF
# Seed for the xorshift64 input sweeps.
comptime sweep_seed: UInt64 = 0x2545F4914F6CDD1D


def run_on_gpu[
    dtype: DType,
    math_fn: def[fn_dtype: DType, fn_width: SIMDLength](
        SIMD[fn_dtype, fn_width]
    ) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point(),
](ctx: DeviceContext, inputs: List[Scalar[dtype]]) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()
    assert_equal(len(inputs), length)

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = inputs[i]

    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0].value())
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, math_fn(val)
        )

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var out = List[Scalar[dtype]](capacity=length)
    with out_device.map_to_host() as out_host:
        for i in range(length):
            out.append(out_host[i])
    return out^


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


def _xorshift(mut state: UInt64) -> UInt64:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return state


# ===----------------------------------------------------------------------=== #
# log10 float64
# ===----------------------------------------------------------------------=== #

comptime num_log10_specials = 11


def _log10_f64_inputs() -> List[Float64]:
    # 2^64 / golden ratio, the Fibonacci-hashing multiplier.
    comptime golden_ratio_u64: UInt64 = 0x9E37_79B9_7F4A_7C15
    comptime f64_min_normal_bits: UInt64 = 0x0010_0000_0000_0000
    comptime f64_max_finite_bits: UInt64 = 0x7FEF_FFFF_FFFF_FFFF
    var v: List[Float64] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        inf[.float64](),
        -inf[.float64](),
        nan[.float64](),
        bitcast[.float64](f64_min_subnormal_bits),
        bitcast[.float64](f64_max_subnormal_bits),
        bitcast[.float64](f64_min_normal_bits),
        bitcast[.float64](f64_max_finite_bits),
    ]
    v.reserve(length)
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
        var bit = UInt64(1) << UInt64(i)
        for j in range(4):
            v.append(
                bitcast[.float64](
                    bit | ((UInt64(j) * golden_ratio_u64) & (bit - 1))
                )
            )
    # Near 1, where log10 is small and cancellation matters.
    var one_bits = bitcast[.uint64](Float64(1.0))
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
    while len(v) < length:
        v.append(
            bitcast[.float64](
                f64_min_normal_bits
                + _xorshift(state) % (f64_max_finite_bits - f64_min_normal_bits)
            )
        )
    return v^


def test_log10_float64() raises:
    # glibc's log10 is not correctly rounded and is up to 2 ULP from the true
    # value near 1 (e.g. log10(0.9999999999999691)), so allow 2 ULP vs libm.
    comptime max_ulp_log10 = 2
    # 10^n is exactly representable in float64 for n <= 22.
    comptime max_exact_pow10 = 22
    comptime if not default_accelerator().is_apple_gpu():
        with DeviceContext() as ctx:
            var inputs = _log10_f64_inputs()
            var got = run_on_gpu[.float64, log10](ctx, inputs)
            var max_ulp = 0
            var max_ulp_host = 0
            for i in range(length):
                var x = inputs[i]
                var expected = _call_libm["log10"](x)
                var d = _ulp_diff(got[i], expected)
                max_ulp = max(max_ulp, d)
                assert_true(
                    d <= max_ulp_log10,
                    String(
                        t"gpu log10({x}) = {got[i]}, expected"
                        t" {expected} ({d} ulp)"
                    ),
                )
                var host = log10(x)
                var dh = _ulp_diff(host, expected)
                max_ulp_host = max(max_ulp_host, dh)
                assert_true(
                    dh <= max_ulp_log10,
                    String(
                        t"host log10({x}) = {host}, expected"
                        t" {expected} ({dh} ulp)"
                    ),
                )
            for n in range(max_exact_pow10 + 1):
                assert_equal(got[num_log10_specials + n], Float64(n))
            print("log10 float64 max ulp: gpu", max_ulp, "host", max_ulp_host)


# ===----------------------------------------------------------------------=== #
# log10 float32
# ===----------------------------------------------------------------------=== #


def _log10_f32_inputs() -> List[Float32]:
    comptime f32_min_normal_bits: UInt32 = 0x0080_0000
    comptime f32_max_finite_bits: UInt32 = 0x7F7F_FFFF
    comptime f32_inf_bits = 0x7F80_0000
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
            bitcast[.float32]((_xorshift(state) % f32_inf_bits).cast[.uint32]())
        )
    return v^


def test_log10_float32() raises:
    # Against float64 libm rounded to float32; that reference is itself
    # within 1 ULP of the true value. Every target runs the same algorithm,
    # so GPU and host agree to within the rounding of fused multiply-adds.
    comptime max_ulp_f32 = 1
    with DeviceContext() as ctx:
        var inputs = _log10_f32_inputs()
        var got = run_on_gpu[.float32, log10](ctx, inputs)
        var max_ulp = 0
        var max_ulp_host = 0
        for i in range(length):
            var x = inputs[i]
            var expected = _call_libm["log10"](x.cast[.float64]()).cast[
                .float32
            ]()
            var d = _ulp_diff(got[i], expected)
            max_ulp = max(max_ulp, d)
            assert_true(
                d <= max_ulp_f32,
                String(
                    t"gpu log10({x}) = {got[i]}, expected {expected} ({d} ulp)"
                ),
            )
            var host = log10(x)
            var dh = _ulp_diff(host, expected)
            max_ulp_host = max(max_ulp_host, dh)
            assert_true(
                dh <= max_ulp_f32,
                String(
                    t"host log10({x}) = {host}, expected {expected} ({dh} ulp)"
                ),
            )
        for n in range(11):
            assert_equal(got[num_log10_specials + n], Float32(n))
        print("log10 float32 max ulp: gpu", max_ulp, "host", max_ulp_host)


# ===----------------------------------------------------------------------=== #
# log10 subnormal inputs
# ===----------------------------------------------------------------------=== #


def _check_subnormals_on_gpu[
    dtype: DType
](
    ctx: DeviceContext, subnormals: List[Scalar[dtype]]
) raises where dtype.is_floating_point():
    var inputs = List[Scalar[dtype]](capacity=length)
    for x in subnormals:
        inputs.append(x)
        inputs.append(-x)
    while len(inputs) < length:
        inputs.append(1.0)
    var got = run_on_gpu[dtype, log10](ctx, inputs)
    for i in range(len(subnormals)):
        var x = subnormals[i]
        var expected = _call_libm["log10"](x.cast[.float64]()).cast[dtype]()
        var d = _ulp_diff(got[2 * i], expected)
        assert_true(
            d <= 1,
            String(
                t"gpu log10({x}) = {got[2 * i]}, expected {expected} ({d} ulp)"
            ),
        )
        # A negative subnormal is still negative, not a zero.
        assert_true(
            isnan(got[2 * i + 1]),
            String(t"gpu log10({-x}) = {got[2 * i + 1]}, expected nan"),
        )


def test_log10_subnormal() raises:
    """The log reduction reads a subnormal's exponent from its bits, so
    subnormal inputs survive targets that flush subnormal arithmetic operands.
    """
    with DeviceContext() as ctx:
        _check_subnormals_on_gpu[.float32](
            ctx,
            [
                bitcast[.float32](f32_min_subnormal_bits),
                bitcast[.float32](UInt32(2)),
                bitcast[.float32](UInt32(0x0000_0123)),
                bitcast[.float32](UInt32(0x0040_0000)),
                bitcast[.float32](UInt32(0x0055_5555)),
                bitcast[.float32](f32_max_subnormal_bits),
            ],
        )
        _check_subnormals_on_gpu[.bfloat16](
            ctx,
            [
                bitcast[.bfloat16](UInt16(1)),
                bitcast[.bfloat16](UInt16(0x0023)),
                bitcast[.bfloat16](UInt16(0x0040)),
                bitcast[.bfloat16](UInt16(0x007F)),
            ],
        )
        _check_subnormals_on_gpu[.float16](
            ctx,
            [
                bitcast[.float16](UInt16(1)),
                bitcast[.float16](UInt16(0x0123)),
                bitcast[.float16](UInt16(0x0200)),
                bitcast[.float16](UInt16(0x03FF)),
            ],
        )
        comptime if not default_accelerator().is_apple_gpu():
            _check_subnormals_on_gpu[.float64](
                ctx,
                [
                    bitcast[.float64](f64_min_subnormal_bits),
                    bitcast[.float64](f64_max_subnormal_bits),
                ],
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
