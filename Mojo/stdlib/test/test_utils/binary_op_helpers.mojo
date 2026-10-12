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
"""Provides inputs and a libm reference for testing two-argument math
functions such as `hypot` and `remainder`."""

from std.ffi import _external_call_const
from std.math import inf, nan
from std.sys import bit_width_of
from std.utils.numerics import FPUtils, max_finite


struct SplitMix64:
    """A splitmix64 pseudo-random generator.

    It gives every platform the same sequence for a seed, so a failing input
    reproduces anywhere.
    """

    var state: UInt64
    """The generator state."""

    def __init__(out self, seed: UInt64):
        """Creates a generator.

        Args:
            seed: The initial state.
        """
        self.state = seed

    def next(mut self) -> UInt64:
        """Advances the generator.

        Returns:
            The next 64 random bits.
        """
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    def bits[
        dtype: DType
    ](mut self) -> Scalar[dtype] where dtype.is_floating_point():
        """Returns a value with a uniformly random bit pattern.

        Parameters:
            dtype: The floating-point type to return.

        Returns:
            A random `dtype` value, which may be a NaN, an infinity or a
            subnormal.
        """
        comptime nbits = bit_width_of[dtype]()
        var v = self.next() >> UInt64(64 - nbits)
        return Scalar[dtype](from_bits=v.cast[FPUtils[dtype].integral_type]())

    def uniform(mut self) -> Float64:
        """Returns a uniformly random value in [0, 1).

        Returns:
            A float64 with 53 random bits.
        """
        return Float64(self.next() >> 11) / Float64(1 << 53)


def libm_call_binary[
    dtype: DType, //, fn_fp32: StaticString, fn_fp64: StaticString
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """Calls a two-argument libm function on the host.

    float16 and bfloat16 arguments go through the float32 function, and its
    result is rounded to `dtype`.

    Parameters:
        dtype: The floating-point type of the arguments.
        fn_fp32: Name of the float32 version of the libm function.
        fn_fp64: Name of the float64 version of the libm function.

    Args:
        x: The first argument.
        y: The second argument.

    Returns:
        The libm result.
    """
    comptime match dtype:
        case .float64:
            return _external_call_const[fn_fp64, Scalar[dtype]](x, y)
        case .float32:
            return _external_call_const[fn_fp32, Scalar[dtype]](x, y)
        case .float16 | .bfloat16:
            return _external_call_const[fn_fp32, Float32](
                x.cast[.float32](), y.cast[.float32]()
            ).cast[dtype]()
        case _:
            comptime assert False, "expected a floating point"


@fieldwise_init
struct PairMode(EnumLike, Equatable, ImplicitlyCopyable):
    """The kind of random `(x, y)` pairs that `pair_inputs` generates."""

    var _value: Int

    comptime random_bits = Self(0)
    """Both `x` and `y` have uniformly random bit patterns."""

    comptime integer_y = Self(1)
    """`x` has a random bit pattern and `y` is a small integer, sometimes plus
    one half and sometimes scaled by 1e6."""

    comptime small_quotients = Self(2)
    """`x / y` is a half-integer below 1024, which `remainder` must round as
    a tie, or a random value from 1 to 9."""

    comptime near_multiples = Self(3)
    """`x` is within 3 ULPs of `n * y` for a quotient `n` of up to
    `mantissa_width + 1` bits, where an inexact `x / y` misjudges
    `trunc(x / y)`."""

    comptime _enum_case_names = ParameterList.of[
        "random_bits".value,
        "integer_y".value,
        "small_quotients".value,
        "near_multiples".value,
    ].values

    comptime _enum_case_types = TypeList.splat[
        ParameterList[Self._enum_case_names].size, NoneType
    ].values

    @inline(.always)
    def _get_enum_discriminant(self) -> Int:
        return self._value

    @inline(.always)
    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        comptime assert False, "PairMode has no payload"


def pair_inputs[
    dtype: DType
](
    specials: ImmSpan[Scalar[dtype], _],
    mode: PairMode,
    seed: UInt64,
    length: Int,
) -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
] where dtype.is_floating_point():
    """Returns every pair of `specials`, then random `(x, y)` pairs of the
    kind `mode` selects, up to `length` pairs in all.

    Parameters:
        dtype: The floating-point type of the inputs.

    Args:
        specials: The values to pair with each other first.
        mode: The kind of random pairs that fill the rest.
        seed: The seed for the random pairs.
        length: The total number of pairs, unless the special pairs alone
            exceed it.

    Returns:
        The x values and the y values.
    """
    var rng = SplitMix64(seed)
    var xs = List[Scalar[dtype]]()
    var ys = List[Scalar[dtype]]()
    for a in specials:
        for b in specials:
            xs.append(a)
            ys.append(b)
    var i = len(xs)
    while i < length:
        var x = rng.bits[dtype]()
        var y: Scalar[dtype]
        match mode:
            case .random_bits:
                y = rng.bits[dtype]()
            case .integer_y:
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
            case .small_quotients:
                y = rng.bits[dtype]()
                var n = Scalar[dtype](Int(rng.next() >> 54))
                if i % 2 == 0:
                    x = (n + 0.5) * y
                else:
                    x = y * Scalar[dtype](1 + rng.uniform() * 8)
            case .near_multiples:
                comptime IT = FPUtils[dtype].integral_type
                comptime qbits = FPUtils[dtype].mantissa_width() + 1
                y = rng.bits[dtype]()
                var n = Scalar[dtype](Int(rng.next() >> UInt64(64 - qbits)))
                var nudge = Scalar[IT](Int(rng.next() % 7) - 3)
                x = Scalar[dtype](
                    from_bits=(n * y).to_bits().cast[IT]() + nudge
                )
        xs.append(x)
        ys.append(y)
        i += 1
    return (xs^, ys^)


comptime _FPIntegralType[dtype: DType] = Scalar[FPUtils[dtype].integral_type]
comptime _tiny[dtype: DType] = Scalar[dtype](
    from_bits=_FPIntegralType[dtype](1)
)
comptime _min_normal[dtype: DType] = Scalar[dtype](
    from_bits=_FPIntegralType[dtype](1 << FPUtils[dtype].mantissa_width())
)


def _specials_shared[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    """Returns the special values that both `hypot_specials` and
    `remainder_specials` contain."""
    var big = max_finite[dtype]()
    var tiny = _tiny[dtype]
    var min_normal = _min_normal[dtype]
    var max_subnormal = Scalar[dtype](
        from_bits=_FPIntegralType[dtype](
            (1 << FPUtils[dtype].mantissa_width()) - 1
        )
    )
    return [
        max_subnormal,
        -max_subnormal,
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


def hypot_specials[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    """Returns the special values whose pairs the `hypot` tests check first.

    Parameters:
        dtype: The floating-point type of the values.

    Returns:
        Signed zeros, small integers and halves, the extremes of the finite,
        normal and subnormal ranges, the infinities and NaN.
    """
    var specials = _specials_shared[dtype]()
    var half_min_normal = Scalar[dtype](
        from_bits=_FPIntegralType[dtype](
            1 << (FPUtils[dtype].mantissa_width() - 1)
        )
    )
    specials.append(half_min_normal)
    comptime if dtype == .bfloat16:
        # Paired with max_subnormal, the f32 result is subnormal but casts
        # to the minimum normal bf16 value.
        specials.append(Scalar[dtype](from_bits=_FPIntegralType[dtype](0x000C)))
    return specials^


def remainder_specials[
    dtype: DType
]() -> List[Scalar[dtype]] where dtype.is_floating_point():
    """Returns the special values whose pairs the `remainder` tests check
    first.

    Parameters:
        dtype: The floating-point type of the values.

    Returns:
        Signed zeros, small integers and halves, the extremes of the finite,
        normal and subnormal ranges, subnormal multiples, the infinities and
        NaN.
    """
    var tiny = _tiny[dtype]
    var min_normal = _min_normal[dtype]
    var specials = _specials_shared[dtype]()
    specials.extend([tiny * 3, min_normal * 1.5])
    return specials^
