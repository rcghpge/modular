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
from std.math import hypot, inf, isinf, isnan, nan
from std.sys import bit_width_of, default_accelerator, simd_width_of
from std.utils.numerics import FPUtils, max_finite

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from std.testing import TestSuite, assert_equal, assert_raises

from std.utils.coord import Coord

comptime length = 8192
comptime num_fill_modes = 3
comptime seed = 1234


def _libm[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    """Calls host libm directly; f16/bf16 round the f32 libm result."""
    comptime if dtype == .float64:
        return _external_call_const["hypot", Scalar[dtype]](x, y)
    elif dtype == .float32:
        return _external_call_const["hypotf", Scalar[dtype]](x, y)
    return _external_call_const["hypotf", Float32](
        x.cast[.float32](), y.cast[.float32]()
    ).cast[dtype]()


def _check[
    dtype: DType, //, max_ulp: Int
](
    got: Scalar[dtype],
    expected: Scalar[dtype],
    x: Scalar[dtype],
    y: Scalar[dtype],
    side: StaticString,
    i: Int,
) raises -> Int:
    var ok: Bool
    var d = 0
    var got_bits = Int(got._to_bits_signed())
    var expected_bits = Int(expected._to_bits_signed())
    if isnan(expected):
        ok = isnan(got)
    elif isinf(expected):
        ok = got_bits == expected_bits and got_bits > 0
    elif isnan(got) or isinf(got) or got_bits < 0 or expected_bits < 0:
        ok = False
    elif expected_bits == 0:
        ok = got_bits == 0
    else:
        # Nonnegative finite encodings cannot overflow this signed distance.
        d = abs(got_bits - expected_bits)
        ok = d <= max_ulp
    if not ok:
        raise Error(
            t"{side} hypot {dtype} at {i}: ({x}, {y}) got {got} expected",
            t" {expected} ({d} ulp)",
        )
    return d


def run_hypot[
    dtype: DType, //, max_ulp: Int
](
    ctx: DeviceContext,
    x_device: DeviceBuffer[dtype],
    y_device: DeviceBuffer[dtype],
) raises where dtype.is_floating_point():
    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    var out_device = ctx.enqueue_create_buffer[dtype](length)
    var x_buffer = Span(unsafe_ptr=x_device.unsafe_ptr(), length=length)
    var y_buffer = Span(unsafe_ptr=y_device.unsafe_ptr(), length=length)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=length)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var x_buffer, var y_buffer}:
        var idx = Int(idx0[0].value())
        var a = x_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        var b = y_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](idx, hypot(a, b))

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    var worst = 0
    var worst_host = 0
    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var x = x_host[i]
            var y = y_host[i]
            var expected = _libm(x, y)
            worst = max(
                worst,
                _check[max_ulp](
                    out_host[i],
                    expected,
                    x,
                    y,
                    "gpu",
                    i,
                ),
            )
            worst_host = max(
                worst_host,
                _check[max_ulp](hypot(x, y), expected, x, y, "host", i),
            )
    print(t"hypot {dtype} max ulp gpu: {worst} host: {worst_host}")


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
        """Returns a uniformly random bit pattern of `dtype`."""
        comptime nbits = bit_width_of[dtype]()
        var v = self.next() >> UInt64(64 - nbits)
        return Scalar[dtype](from_bits=v.cast[FPUtils[dtype].integral_type]())

    def uniform(mut self) -> Float64:
        return Float64(self.next() >> 11) / Float64(1 << 53)


def _specials[dtype: DType]() -> List[Scalar[dtype]]:
    var big = max_finite[dtype]()
    var tiny = Scalar[dtype](from_bits=Scalar[FPUtils[dtype].integral_type](1))
    var min_normal = Scalar[dtype](
        from_bits=Scalar[FPUtils[dtype].integral_type](
            1 << FPUtils[dtype].mantissa_width()
        )
    )
    var half_min_normal = Scalar[dtype](
        from_bits=Scalar[FPUtils[dtype].integral_type](
            1 << (FPUtils[dtype].mantissa_width() - 1)
        )
    )
    var max_subnormal = Scalar[dtype](
        from_bits=Scalar[FPUtils[dtype].integral_type](
            (1 << FPUtils[dtype].mantissa_width()) - 1
        )
    )
    var specials: List[Scalar[dtype]] = [
        half_min_normal,
        max_subnormal,
        Scalar[dtype](0),
        Scalar[dtype](-0.0),
        Scalar[dtype](1),
        Scalar[dtype](-1),
        Scalar[dtype](0.5),
        Scalar[dtype](1.5),
        Scalar[dtype](2.5),
        Scalar[dtype](-2.5),
        Scalar[dtype](3),
        Scalar[dtype](-7),
        Scalar[dtype](0.75),
        Scalar[dtype](10),
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

    comptime if dtype == .bfloat16:
        # Paired with max_subnormal, the f32 result is subnormal but casts
        # to the minimum normal bf16 value.
        specials.append(
            Scalar[dtype](
                from_bits=Scalar[FPUtils[dtype].integral_type](0x000C)
            )
        )
    return specials^


def _fill_pairs[
    dtype: DType
](
    x_device: DeviceBuffer[dtype],
    y_device: DeviceBuffer[dtype],
    mode: Int,
    seed: UInt64,
) raises:
    """Fills x and y with special-value pairs, then `mode`-specific inputs.

    mode 0: random bit patterns for both operands.
    mode 1: x random bits, y a small integer (some non-integral, some huge).
    mode 2: x = (n + 0.5) * y, and x, y of nearby magnitude.
    """
    var rng = _Rng(seed)
    var sp = _specials[dtype]()
    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host:
        var i = 0
        for a in sp:
            for b in sp:
                x_host[i] = a
                y_host[i] = b
                i += 1
        while i < length:
            var x = rng.bits[dtype]()
            var y: Scalar[dtype]
            if mode == 0:
                y = rng.bits[dtype]()
            elif mode == 1:
                var r = rng.uniform()
                comptime span = 2 * (
                    FPUtils[dtype].exponent_bias()
                    + FPUtils[dtype].mantissa_width()
                    + 4
                )
                var n = Float64(Int(r * Float64(span)) - span // 2)
                if i % 17 == 0:
                    n += 0.5
                elif i % 29 == 0:
                    n *= 1e6
                y = Scalar[dtype](n)
            else:
                y = rng.bits[dtype]()
                var n = Scalar[dtype](Int(rng.next() >> 54))
                if i % 2 == 0:
                    x = (n + 0.5) * y
                else:
                    x = y * Scalar[dtype](1 + rng.uniform() * 8)
            x_host[i] = x
            y_host[i] = y
            i += 1


def _run_all[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var x = ctx.enqueue_create_buffer[dtype](length)
    var y = ctx.enqueue_create_buffer[dtype](length)
    # Compare representable steps to the rounded libm oracle, not real-valued
    # ULP error. Narrow types can round the float32 paths to adjacent values.
    comptime hypot_ulp = 2 if dtype in (DType.float32, DType.float64) else 1
    for mode in range(num_fill_modes):
        _fill_pairs(x, y, mode, UInt64(seed + mode))
        run_hypot[hypot_ulp](ctx, x, y)


def _test_oracle[dtype: DType]() raises:
    def check(got: Scalar[dtype], expected: Scalar[dtype]) raises -> Int:
        return _check[2](got, expected, 0, 0, "oracle", 0)

    var big = max_finite[dtype]()
    var infinity = inf[dtype]()
    var snan = Scalar[dtype](from_bits=infinity._to_bits_signed() | 1)
    with assert_raises():
        _ = check(infinity, big)
    with assert_raises():
        _ = check(big, infinity)
    with assert_raises():
        _ = check(snan, infinity)
    with assert_raises():
        _ = check(snan, big)
    with assert_raises():
        _ = check(Scalar[dtype](-0.0), Scalar[dtype](0.0))
    with assert_raises():
        _ = check(Scalar[dtype](-2.0), Scalar[dtype](2.0))
    var adjacent = Scalar[dtype](
        from_bits=Scalar[dtype](1)._to_bits_signed() + 1
    )
    assert_equal(check(adjacent, Scalar[dtype](1)), 1)
    assert_equal(check(infinity, infinity), 0)
    assert_equal(check(snan, nan[dtype]()), 0)


def test_oracle() raises:
    _test_oracle[.float32]()
    _test_oracle[.float64]()
    _test_oracle[.float16]()
    _test_oracle[.bfloat16]()


def test_hypot() raises:
    with DeviceContext() as ctx:
        _run_all[.float32](ctx)
        _run_all[.float16](ctx)
        _run_all[.bfloat16](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _run_all[.float64](ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
