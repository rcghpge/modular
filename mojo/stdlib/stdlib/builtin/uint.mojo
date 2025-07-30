# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Implements the UInt class.

These are Mojo built-ins, so you don't need to import them.
"""

from hashlib.hasher import Hasher
from math import CeilDivable
from sys import bitwidthof

from builtin.math import Absable

from utils._visualizers import lldb_formatter_wrapping_type


@lldb_formatter_wrapping_type
@register_passable("trivial")
struct UInt(
    Absable,
    Boolable,
    CeilDivable,
    Comparable,
    Copyable,
    Defaultable,
    ExplicitlyCopyable,
    Hashable,
    Indexer,
    KeyElement,
    Movable,
    Representable,
    Stringable,
    Writable,
):
    """This type represents an unsigned integer.

    The size of this unsigned integer is platform-dependent.

    If you wish to use a fixed size unsigned integer, consider using
    `UInt8`, `UInt16`, `UInt32`, or `UInt64`.
    """

    alias BITWIDTH = Int(bitwidthof[DType.index]())
    """The bit width of the integer type."""

    alias MAX = UInt((1 << Self.BITWIDTH) - 1)
    """Returns the maximum integer value."""

    alias MIN = UInt(0)
    """Returns the minimum value of type."""

    var value: __mlir_type.index
    """The underlying storage for the integer value.


    Note that it is the same type as the `Int.value` field.
    MLIR doesn't differentiate between signed and unsigned integers
    when it comes to storing them with the index dialect.
    The difference is in the operations that are performed on them,
    which have signed and unsigned variants.
    """

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline("builtin")
    fn __init__(out self):
        """Default constructor that produces zero."""
        self.value = __mlir_attr.`0 : index`

    @doc_private
    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: __mlir_type.index):
        """Construct UInt from the given index value.

        Args:
            value: The init value.
        """
        self.value = value

    @doc_private
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: __mlir_type.`!pop.scalar<index>`):
        """Construct UInt from the given Index value.

        Args:
            value: The init value.
        """
        self.value = __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.index](
            value
        )

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: IntLiteral):
        """Construct UInt from the given IntLiteral value.

        Args:
            value: The init value.
        """
        self = value.__uint__()

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Int):
        """Construct UInt from the given Int value.

        Args:
            value: The init value.
        """
        self.value = value.value

    @always_inline("nodebug")
    fn __init__[T: Indexer](out self, value: T):
        """Construct UInt from the given Indexable value.

        Parameters:
            T: The type that that can index into a collection or pointer.

        Args:
            value: The init value.
        """
        self = value.__index__()

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")  # TODO: should be "builtin"
    fn __lt__(self, rhs: UInt) -> Bool:
        """Return whether this UInt is strictly less than another.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this UInt is less than the other UInt and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ult>`
        ](self.value, rhs.value)

    @always_inline("nodebug")  # TODO: should be "builtin"
    fn __le__(self, rhs: UInt) -> Bool:
        """Compare this Int to the RHS using LE comparison.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this Int is less-or-equal than the RHS Int and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ule>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __eq__(self, rhs: UInt) -> Bool:
        """Compare this UInt to the RHS using EQ comparison.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this UInt is equal to the RHS UInt and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate eq>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __ne__(self, rhs: UInt) -> Bool:
        """Compare this UInt to the RHS using NE comparison.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this UInt is non-equal to the RHS UInt and False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ne>`
        ](self.value, rhs.value)

    @always_inline("nodebug")  # TODO: should be "builtin"
    fn __gt__(self, rhs: UInt) -> Bool:
        """Return whether this UInt is strictly greater than another.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this UInt is greater than the other UInt and False
            otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate ugt>`
        ](self.value, rhs.value)

    @always_inline("nodebug")  # TODO: should be "builtin"
    fn __ge__(self, rhs: UInt) -> Bool:
        """Return whether this UInt is greater than or equal to another.

        Args:
            rhs: The other UInt to compare against.

        Returns:
            True if this UInt is greater than or equal to the other UInt and
            False otherwise.
        """
        return __mlir_op.`index.cmp`[
            pred = __mlir_attr.`#index<cmp_predicate uge>`
        ](self.value, rhs.value)

    @always_inline("builtin")
    fn __pos__(self) -> UInt:
        """Return +self.

        Returns:
            The +self value.
        """
        return self

    @always_inline("builtin")
    fn __invert__(self) -> UInt:
        """Return ~self.

        Returns:
            The ~self value.
        """
        return self ^ Self.MAX

    @always_inline("builtin")
    fn __add__(self, rhs: UInt) -> UInt:
        """Return `self + rhs`.

        Args:
            rhs: The value to add.

        Returns:
            `self + rhs` value.
        """
        return __mlir_op.`index.add`(self.value, rhs.value)

    @always_inline("builtin")
    fn __sub__(self, rhs: UInt) -> UInt:
        """Return `self - rhs`.

        Args:
            rhs: The value to subtract.

        Returns:
            `self - rhs` value.
        """
        return __mlir_op.`index.sub`(self.value, rhs.value)

    @always_inline("builtin")
    fn __mul__(self, rhs: UInt) -> UInt:
        """Return `self * rhs`.

        Args:
            rhs: The value to multiply with.

        Returns:
            `self * rhs` value.
        """
        return __mlir_op.`index.mul`(self.value, rhs.value)

    fn __truediv__(self, rhs: UInt) -> Float64:
        """Return the floating point division of `self` and `rhs`.

        Args:
            rhs: The value to divide on.

        Returns:
            `Float64(self)/Float64(rhs)` value.
        """
        return Float64(self) / Float64(rhs)

    @always_inline("nodebug")
    fn __floordiv__(self, rhs: UInt) -> UInt:
        """Return the division of `self` and `rhs` rounded down to the nearest
        integer.

        Args:
            rhs: The value to divide on.

        Returns:
            `floor(self/rhs)` value.
        """
        if rhs == 0:
            # this should raise an exception.
            return 0
        return __mlir_op.`index.divu`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __mod__(self, rhs: UInt) -> UInt:
        """Return the remainder of self divided by rhs.

        Args:
            rhs: The value to divide on.

        Returns:
            The remainder of dividing self by rhs.
        """
        if rhs == 0:
            # this should raise an exception
            return 0
        return __mlir_op.`index.remu`(self.value, rhs.value)

    @always_inline("nodebug")
    fn __divmod__(self, rhs: UInt) -> Tuple[UInt, UInt]:
        """Computes both the quotient and remainder using integer division.

        Args:
            rhs: The value to divide on.

        Returns:
            The quotient and remainder as a `Tuple(self // rhs, self % rhs)`.
        """
        if rhs == 0:
            # this should raise an exception
            return UInt(0), UInt(0)
        return self // rhs, self % rhs

    @always_inline("nodebug")
    fn __pow__(self, exp: Self) -> Self:
        """Return the value raised to the power of the given exponent.

        Computes the power of an integer using the Russian Peasant Method.

        Args:
            exp: The exponent value.

        Returns:
            The value of `self` raised to the power of `exp`.
        """
        var res: UInt = 1
        var x = self
        var n = exp
        while n > 0:
            if n & 1 != 0:
                res *= x
            x *= x
            n >>= 1
        return res

    @always_inline("builtin")
    fn __lshift__(self, rhs: UInt) -> UInt:
        """Return `self << rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self << rhs`.
        """
        return __mlir_op.`index.shl`(self.value, rhs.value)

    @always_inline("nodebug")  # TODO: should be "builtin"
    fn __rshift__(self, rhs: UInt) -> UInt:
        """Return `self >> rhs`.

        Args:
            rhs: The value to shift with.

        Returns:
            `self >> rhs`.
        """
        return __mlir_op.`index.shru`(self.value, rhs.value)

    @always_inline("builtin")
    fn __and__(self, rhs: UInt) -> UInt:
        """Return `self & rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self & rhs`.
        """
        return __mlir_op.`index.and`(self.value, rhs.value)

    @always_inline("builtin")
    fn __xor__(self, rhs: UInt) -> UInt:
        """Return `self ^ rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self ^ rhs`.
        """
        return __mlir_op.`index.xor`(self.value, rhs.value)

    @always_inline("builtin")
    fn __or__(self, rhs: UInt) -> UInt:
        """Return `self | rhs`.

        Args:
            rhs: The RHS value.

        Returns:
            `self | rhs`.
        """
        return __mlir_op.`index.or`(self.value, rhs.value)

    # ===-------------------------------------------------------------------===#
    # In place operations.
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __iadd__(mut self, rhs: UInt):
        """Compute `self + rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self + rhs

    @always_inline("nodebug")
    fn __isub__(mut self, rhs: UInt):
        """Compute `self - rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self - rhs

    @always_inline("nodebug")
    fn __imul__(mut self, rhs: UInt):
        """Compute self*rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self * rhs

    @always_inline("nodebug")
    fn __itruediv__(mut self, rhs: UInt):
        """Compute `self / rhs`, convert to int, and save the result in self.

        Since `floor(self / rhs)` is equivalent to `self // rhs`, this yields
        the same as `__ifloordiv__`.

        Args:
            rhs: The RHS value.
        """
        self = self // rhs

    @always_inline("nodebug")
    fn __ifloordiv__(mut self, rhs: UInt):
        """Compute `self // rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self // rhs

    @always_inline("nodebug")
    fn __imod__(mut self, rhs: UInt):
        """Compute `self % rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self % rhs

    @always_inline("nodebug")
    fn __ipow__(mut self, rhs: UInt):
        """Compute `pow(self, rhs)` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self**rhs

    @always_inline("nodebug")
    fn __ilshift__(mut self, rhs: UInt):
        """Compute `self << rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self << rhs

    @always_inline("nodebug")
    fn __irshift__(mut self, rhs: UInt):
        """Compute `self >> rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self >> rhs

    @always_inline("nodebug")
    fn __iand__(mut self, rhs: UInt):
        """Compute `self & rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self & rhs

    @always_inline("nodebug")
    fn __ixor__(mut self, rhs: UInt):
        """Compute `self ^ rhs` and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self ^ rhs

    @always_inline("nodebug")
    fn __ior__(mut self, rhs: UInt):
        """Compute self|rhs and save the result in self.

        Args:
            rhs: The RHS value.
        """
        self = self | rhs

    # ===-------------------------------------------------------------------===#
    # Reversed operations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __radd__(self, value: UInt) -> UInt:
        """Return `value + self`.

        Args:
            value: The other value.

        Returns:
            `value + self`.
        """
        return self + value

    @always_inline("builtin")
    fn __rsub__(self, value: UInt) -> UInt:
        """Return `value - self`.

        Args:
            value: The other value.

        Returns:
            `value - self`.
        """
        return value - self

    @always_inline("builtin")
    fn __rmul__(self, value: UInt) -> UInt:
        """Return `value * self`.

        Args:
            value: The other value.

        Returns:
            `value * self`.
        """
        return self * value

    @always_inline("nodebug")
    fn __rfloordiv__(self, value: UInt) -> UInt:
        """Return `value // self`.

        Args:
            value: The other value.

        Returns:
            `value // self`.
        """
        return value // self

    @always_inline("nodebug")
    fn __rmod__(self, value: UInt) -> UInt:
        """Return `value % self`.

        Args:
            value: The other value.

        Returns:
            `value % self`.
        """
        return value % self

    @always_inline("nodebug")
    fn __rpow__(self, value: UInt) -> UInt:
        """Return `pow(value,self)`.

        Args:
            value: The other value.

        Returns:
            `pow(value,self)`.
        """
        return value**self

    @always_inline("builtin")
    fn __rlshift__(self, value: UInt) -> UInt:
        """Return `value << self`.

        Args:
            value: The other value.

        Returns:
            `value << self`.
        """
        return value << self

    @always_inline("nodebug")  # TODO: should be "builtin"
    fn __rrshift__(self, value: UInt) -> UInt:
        """Return `value >> self`.

        Args:
            value: The other value.

        Returns:
            `value >> self`.
        """
        return value >> self

    @always_inline("builtin")
    fn __rand__(self, value: UInt) -> UInt:
        """Return `value & self`.

        Args:
            value: The other value.

        Returns:
            `value & self`.
        """
        return value & self

    @always_inline("builtin")
    fn __ror__(self, value: UInt) -> UInt:
        """Return `value | self`.

        Args:
            value: The other value.

        Returns:
            `value | self`.
        """
        return value | self

    @always_inline("builtin")
    fn __rxor__(self, value: UInt) -> UInt:
        """Return `value ^ self`.

        Args:
            value: The other value.

        Returns:
            `value ^ self`.
        """
        return value ^ self

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __bool__(self) -> Bool:
        """Convert this Int to Bool.

        Returns:
            False Bool value if the value is equal to 0 and True otherwise.
        """
        return self != 0

    @always_inline("builtin")
    fn __index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return self.value

    @always_inline("builtin")
    fn __int__(self) -> Int:
        """Gets the integral value, wrapping to a negative number on overflow.

        Returns:
            The value as an integer.
        """
        return self.value

    @always_inline("builtin")
    fn __abs__(self) -> Self:
        """Return the absolute value of the UInt value.

        Returns:
            The absolute value.
        """
        return self

    @always_inline("builtin")
    fn __ceil__(self) -> Self:
        """Return the ceiling of the UInt value, which is itself.

        Returns:
            The UInt value itself.
        """
        return self

    @always_inline("builtin")
    fn __floor__(self) -> Self:
        """Return the floor of the UInt value, which is itself.

        Returns:
            The UInt value itself.
        """
        return self

    @always_inline("builtin")
    fn __round__(self) -> Self:
        """Return the rounded value of the UInt value, which is itself.

        Returns:
            The UInt value itself.
        """
        return self

    @always_inline("builtin")
    fn __round__(self, ndigits: UInt) -> Self:
        """Return the rounded value of the UInt value, which is itself.

        Args:
            ndigits: The number of digits to round to.

        Returns:
            The UInt value itself if ndigits >= 0 else the rounded value.
        """
        return self

    @always_inline("builtin")
    fn __trunc__(self) -> Self:
        """Return the truncated UInt value, which is itself.

        Returns:
            The Int value itself.
        """
        return self

    @always_inline("builtin")
    fn __ceildiv__(self, denominator: Self) -> Self:
        """Return the rounded-up result of dividing self by denominator.


        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
        return __mlir_op.`index.ceildivu`(self.value, denominator.value)

    @always_inline("builtin")
    fn is_power_of_two(self) -> Bool:
        """Check if the integer is a (non-zero) power of two.

        Returns:
            True if the integer is a power of two, False otherwise.
        """
        return (self & (self - 1) == 0) & (self != 0)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Formats this integer to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write(UInt64(self))

    @no_inline
    fn __str__(self) -> String:
        """Convert this UInt to a string.

        A small example.
        ```mojo
        %# from testing import assert_equal
        x = UInt(50)
        assert_equal(String(x), "50")
        ```

        Returns:
            The string representation of this UInt.
        """
        return String.write(self)

    fn __repr__(self) -> String:
        """Convert this UInt to a string.

        A small example.
        ```mojo
        %# from testing import assert_equal
        x = UInt(50)
        assert_equal(repr(x), "UInt(50)")
        ```

        Returns:
            The string representation of this UInt.
        """
        return String("UInt(", self, ")")

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with this uint value.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_simd(UInt64(self))
