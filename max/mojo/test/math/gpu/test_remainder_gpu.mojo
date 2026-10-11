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
from std.sys import default_accelerator, simd_width_of, size_of
from std.utils.numerics import FPUtils, max_finite

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from std.testing import TestSuite

from std.utils.coord import Coord

comptime length = 8192
comptime num_fill_modes = 4
comptime seed = 1234


def _libm[
    dtype: DType, //
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    """Calls host libm directly; f16/bf16 round the f32 libm result."""
    comptime if dtype == .float64:
        return _external_call_const["remainder", Scalar[dtype]](x, y)
    elif dtype == .float32:
        return _external_call_const["remainderf", Scalar[dtype]](x, y)
    else:
        return _external_call_const["remainderf", Float32](
            x.cast[.float32](), y.cast[.float32]()
        ).cast[dtype]()


def _check[
    dtype: DType, //
](
    got: Scalar[dtype],
    expected: Scalar[dtype],
    x: Scalar[dtype],
    y: Scalar[dtype],
    side: StaticString,
    i: Int,
) raises:
    # remainder is exact, so it must match libm bit for bit; comparing bits
    # also distinguishes +0 from -0.
    var ok: Bool
    if isnan(expected):
        ok = isnan(got)
    else:
        ok = got.to_bits() == expected.to_bits()
    if not ok:
        raise Error(
            t"{side} remainder {dtype} at {i}: ({x}, {y}) got {got} expected",
            t" {expected}",
        )


def run_remainder[
    dtype: DType, //
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
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, remainder(a, b)
        )

    elementwise[pack_size, target="gpu"](func, Coord(length), ctx)

    with x_device.map_to_host() as x_host, y_device.map_to_host() as y_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var x = x_host[i]
            var y = y_host[i]
            var expected = _libm(x, y)
            _check(out_host[i], expected, x, y, "gpu", i)
            _check(remainder(x, y), expected, x, y, "host", i)


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
        comptime nbits = 8 * size_of[dtype]()
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
    var max_sub = Scalar[dtype](
        from_bits=Scalar[FPUtils[dtype].integral_type](
            (1 << FPUtils[dtype].mantissa_width()) - 1
        )
    )
    return [
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
        max_sub,
        -max_sub,
        tiny * 3,
        min_normal * 1.5,
        inf[dtype](),
        -inf[dtype](),
        nan[dtype](),
    ]


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
    mode 2: x = (n + 0.5) * y ties, and x, y of nearby magnitude.
    mode 3: x within a few ulps of n * y for a quotient n of up to
    mantissa_width + 1 bits, where an inexact `/` misjudges trunc(x / y).
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
                x = Scalar[dtype](
                    from_bits=(n * y).to_bits().cast[IT]() + nudge
                )
            x_host[i] = x
            y_host[i] = y
            i += 1


def _run_all[
    dtype: DType
](ctx: DeviceContext) raises where dtype.is_floating_point():
    var x = ctx.enqueue_create_buffer[dtype](length)
    var y = ctx.enqueue_create_buffer[dtype](length)
    for mode in range(num_fill_modes):
        _fill_pairs(x, y, mode, UInt64(seed + mode))
        run_remainder(ctx, x, y)


def test_remainder() raises:
    with DeviceContext() as ctx:
        _run_all[.float32](ctx)
        _run_all[.float16](ctx)
        _run_all[.bfloat16](ctx)
        comptime if not default_accelerator().is_apple_gpu():
            _run_all[.float64](ctx)


comptime UnaryFn = def[fn_dtype: DType, fn_width: SIMDLength](
    SIMD[fn_dtype, fn_width]
) thin -> SIMD[fn_dtype, fn_width] where fn_dtype.is_floating_point()

comptime f32_min = Float32(from_bits=UInt32(1))  # 2^-149
comptime f32_max_sub = Float32(from_bits=UInt32(0x007FFFFF))
comptime f64_min = Float64(from_bits=UInt64(1))  # 2^-1074
comptime f64_max_sub = Float64(from_bits=UInt64(0x000FFFFFFFFFFFFF))


def _run_unary[
    dtype: DType, //, math_fn: UnaryFn
](ctx: DeviceContext, xs: List[Scalar[dtype]]) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Applies `math_fn` to `xs` on the GPU."""
    var n = len(xs)
    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()
    # Pad to a whole number of packs so every lane holds a real input.
    var padded = ((n + pack_size - 1) // pack_size) * pack_size
    var in_device = ctx.enqueue_create_buffer[dtype](padded)
    var out_device = ctx.enqueue_create_buffer[dtype](padded)
    with in_device.map_to_host() as in_host:
        for i in range(padded):
            in_host[i] = xs[i] if i < n else xs[0]
    var in_buffer = Span(unsafe_ptr=in_device.unsafe_ptr(), length=padded)
    var out_buffer = Span(unsafe_ptr=out_device.unsafe_ptr(), length=padded)

    @inline(.always)
    def func[
        simd_width: Int, alignment: Int = 1
    ](idx0: Coord) {var out_buffer, var in_buffer}:
        var idx = Int(idx0[0].value())
        var val = in_buffer.unsafe_ptr().unsafe_load[width=simd_width](idx)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            idx, math_fn(val)
        )

    elementwise[pack_size, target="gpu"](func, Coord(padded), ctx)

    var out = List[Scalar[dtype]]()
    with out_device.map_to_host() as out_host:
        for i in range(n):
            out.append(out_host[i])
    return out^


def _check_exact[
    dtype: DType, //
](
    xs: List[Scalar[dtype]],
    got: List[Scalar[dtype]],
    want: List[Scalar[dtype]],
    name: String,
) raises:
    for i in range(len(xs)):
        if got[i].to_bits() != want[i].to_bits():
            raise Error(t"gpu {name}({xs[i]}) = {got[i]}, expected {want[i]}")


def _remainder_by_min[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    comptime y = SIMD[dtype, width](
        from_bits=SIMD[FPUtils[dtype].integral_type, width](1)
    )
    return remainder(x, y)


def _remainder_by_min_normal[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    comptime y = SIMD[dtype, width](
        from_bits=SIMD[FPUtils[dtype].integral_type, width](
            1 << FPUtils[dtype].mantissa_width()
        )
    )
    return remainder(x, y)


def _remainder_of_one[
    dtype: DType, width: SIMDLength
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    return remainder(SIMD[dtype, width](1.0), x)


# Remainders by a subnormal divisor, and of a subnormal dividend by a small
# divisor, have subnormal results; they must survive targets that flush
# subnormals (Apple GPUs in float32).
def test_remainder_subnormal_f32() raises:
    with DeviceContext() as ctx:
        var xs: List[Float32] = [f32_min, f32_min * 4, -f32_max_sub, 8.0]
        # Every input is a multiple of 2^-149, so each remainder is a zero of
        # the input's sign.
        _check_exact(
            xs,
            _run_unary[_remainder_by_min](ctx, xs),
            [0.0, 0.0, -0.0, 0.0],
            "remainder by 2^-149",
        )
        var positive: List[Float32] = [f32_min, f32_min * 4, 8.0]
        _check_exact(
            positive,
            _run_unary[_remainder_of_one](ctx, positive),
            [0.0, 0.0, 1.0],
            "remainder of 1.0",
        )
        # 2^149 mod (2^23 - 1) = 2^11, a subnormal result.
        var sub: List[Float32] = [f32_max_sub]
        _check_exact(
            sub,
            _run_unary[_remainder_of_one](ctx, sub),
            [f32_min * 2048],
            "remainder of 1.0 by the largest subnormal",
        )
        # Subnormal dividend with the smallest normal divisor: the remainder
        # is x itself, or x - 2^-126 when x is above half of it.
        var tiny_y: List[Float32] = [
            f32_min * 3,
            -f32_min * 5,
            f32_min * Float32(3 << 21),
        ]
        _check_exact(
            tiny_y,
            _run_unary[_remainder_by_min_normal](ctx, tiny_y),
            [f32_min * 3, -f32_min * 5, -f32_min * Float32(1 << 21)],
            "remainder by 2^-126",
        )


def test_remainder_subnormal_f64() raises:
    comptime if not default_accelerator().is_apple_gpu():
        with DeviceContext() as ctx:
            var xs: List[Float64] = [f64_min, -f64_max_sub, 8.0]
            _check_exact(
                xs,
                _run_unary[_remainder_by_min](ctx, xs),
                [0.0, -0.0, 0.0],
                "remainder by 2^-1074",
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
