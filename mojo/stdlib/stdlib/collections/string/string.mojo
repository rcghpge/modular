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
"""The core `String` type implementation for Mojo.

This module provides the primary `String` type and its fundamental operations.
The `String` type is a mutable string, and is designed to handle UTF-8 encoded
text efficiently while providing a safe and ergonomic interface for string
manipulation.

Related types:

- [`StringSlice`](/mojo/stdlib/collections/string/string_slice/). A non-owning
  view of string data, which can be either mutable or immutable.
- [`StaticString`](/mojo/stdlib/collections/string/string_slice/#aliases). An
  alias for an immutable constant `StringSlice`.
- [`StringLiteral`](/mojo/stdlib/builtin/string_literal/StringLiteral/). A
  string literal. String literals are compile-time values. For use at runtime,
  you usually want wrap a `StringLiteral` in a `String` (for a mutable string)
  or `StaticString` (for an immutable constant string).

Key Features:
- Short string optimization (SSO) and lazy copying of constant string data.
- O(1) copy operation.
- Memory-safe string operations.
- Efficient string concatenation and slicing.
- String-to-number conversions (
  [`atof()`](/mojo/stdlib/collections/string/string/atof),
  [`atol()`](/mojo/stdlib/collections/string/string/atol)).
- Character code conversions (
  [`chr()`](/mojo/stdlib/collections/string/string/chr),
  [`ord()`](/mojo/stdlib/collections/string/string/ord)).
- String formatting with
  [`format()`](/mojo/stdlib/collections/string/string/String/#format).

The `String` type has Unicode support through UTF-8 encoding. A handful of
operations are known to not be Unicode / UTF-8 compliant yet, but will be fixed
as time permits.

This type is in the prelude, so it is automatically imported into every Mojo
program.

Example:

```mojo
# String creation and basic operations
var s1 = "Hello"
var s2 = "World"
var combined = s1 + " " + s2  # "Hello World"

# String-to-number conversion
var num = atof("3.14")
var int_val = atol("42")

# Character operations
var char = chr(65)  # "A"
var code = ord("A")  # 65

# String formatting
print("Codepoint {} is {}".format(code, char)) # Codepoint 65 is A

# ASCII utilities
var ascii_str = ascii("Hello")  # ASCII-only string
```
"""

from collections import KeyElement
from collections._index_normalization import normalize_index
from collections.string import CodepointsIter
from collections.string._parsing_numbers.parsing_floats import _atof
from collections.string._unicode import (
    is_lowercase,
    is_uppercase,
    to_lowercase,
    to_uppercase,
)
from collections.string.format import _CurlyEntryFormattable, _FormatCurlyEntry
from collections.string.string_slice import (
    CodepointSliceIter,
    _to_string_list,
    _utf8_byte_type,
)
from hashlib.hasher import Hasher
from os import PathLike, abort
from os.atomic import Atomic
from sys import bitwidthof, sizeof
from sys.info import is_32bit
from sys.ffi import c_char

from bit import count_leading_zeros
from memory import memcpy, memset, memcmp
from python import PythonConvertible, PythonObject, ConvertibleFromPython

from utils import IndexList, Variant
from utils._select import _select_register_value as select
from utils.write import _WriteBufferStack


# ===----------------------------------------------------------------------=== #
# String
# ===----------------------------------------------------------------------=== #


struct String(
    Boolable,
    Comparable,
    ConvertibleFromPython,
    Defaultable,
    ExplicitlyCopyable,
    FloatableRaising,
    IntableRaising,
    KeyElement,
    PathLike,
    PythonConvertible,
    Representable,
    Sized,
    Stringable,
    Writable,
    Writer,
    _CurlyEntryFormattable,
):
    """Represents a mutable string.

    See the [`string` module](/mojo/stdlib/collections/string/string/) for
    more information and examples.
    """

    # Fields: String has two forms - the declared form here, and the "inline"
    # form when '_capacity_or_data.is_inline()' is true. The inline form
    # clobbers these fields (except the top byte of the capacity field) with
    # the string data.
    var _ptr_or_data: UnsafePointer[UInt8]
    """The underlying storage for the string data."""
    var _len_or_data: Int
    """The number of bytes in the string data."""
    var _capacity_or_data: UInt
    """The capacity and bit flags for this String."""

    # Useful string aliases.
    alias ASCII_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
    alias ASCII_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alias ASCII_LETTERS = Self.ASCII_LOWERCASE + Self.ASCII_UPPERCASE
    alias DIGITS = "0123456789"
    alias HEX_DIGITS = Self.DIGITS + "abcdef" + "ABCDEF"
    alias OCT_DIGITS = "01234567"
    alias PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
    alias PRINTABLE = Self.DIGITS + Self.ASCII_LETTERS + Self.PUNCTUATION + " \t\n\r\v\f"

    # ===------------------------------------------------------------------=== #
    # String Implementation Details
    # ===------------------------------------------------------------------=== #
    # This is the number of bytes that can be stored inline in the string value.
    # 'String' is 3 words in size and we use the top byte of the capacity field
    # to store flags.
    alias INLINE_CAPACITY = Int.BITWIDTH // 8 * 3 - 1
    # When FLAG_HAS_NUL_TERMINATOR is set, the byte past the end of the string
    # is known to be an accessible 'nul' terminator.
    alias FLAG_HAS_NUL_TERMINATOR = UInt(1) << (UInt.BITWIDTH - 3)
    # When FLAG_IS_REF_COUNTED is set, the string is pointing to a mutable buffer
    # that may have other references to it.
    alias FLAG_IS_REF_COUNTED = UInt(1) << (UInt.BITWIDTH - 2)
    # When FLAG_IS_INLINE is set, the string is inline or "Short String
    # Optimized" (SSO). The first 23 bytes of the fields are treated as UTF-8
    # data
    alias FLAG_IS_INLINE = UInt(1) << (UInt.BITWIDTH - 1)
    # gives us 5 bits for the length.
    alias INLINE_LENGTH_START = UInt(Int.BITWIDTH - 8)
    alias INLINE_LENGTH_MASK = UInt(0b1_1111 << Self.INLINE_LENGTH_START)
    # This is the size to offset the pointer by, to get access to the
    # atomic reference count prepended to the UTF-8 data.
    alias REF_COUNT_SIZE = sizeof[Atomic[DType.index]]()

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __del__(owned self):
        """Destroy the string data."""
        self._drop_ref()

    @always_inline("nodebug")
    fn __init__(out self):
        """Construct an empty string."""
        self._capacity_or_data = Self.FLAG_IS_INLINE
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    @always_inline("nodebug")
    fn __init__(out self, *, capacity: Int):
        """Construct an empty string with a given capacity.

        Args:
            capacity: The capacity of the string to allocate.
        """
        if capacity <= Self.INLINE_CAPACITY:
            self._capacity_or_data = Self.FLAG_IS_INLINE
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(self)
            )
        else:
            self._capacity_or_data = (capacity + 7) >> 3
            self._ptr_or_data = Self._alloc(self._capacity_or_data << 3)
            self._len_or_data = 0
            self._set_ref_counted()

    @always_inline("nodebug")
    @implicit  # does not allocate.
    fn __init__(out self, data: StaticString):
        """Construct a `String` from a `StaticString` without allocating.

        Args:
            data: The static constant string to refer to.
        """
        self._len_or_data = data._slice._len
        self._ptr_or_data = data._slice._data
        # Always use static constant representation initially, defer inlining
        # decision until mutation to avoid unnecessary memcpy.
        self._capacity_or_data = 0

    @always_inline("nodebug")
    @implicit  # does not allocate.
    fn __init__(out self, data: StringLiteral):
        """Construct a `String` from a `StringLiteral` without allocating.

        Args:
            data: The static constant string to refer to.
        """
        self._len_or_data = __mlir_op.`pop.string.size`(data.value)
        self._ptr_or_data = UnsafePointer(
            __mlir_op.`pop.string.address`(data.value)
        ).bitcast[Byte]()
        # Always use static constant representation initially, defer inlining
        # decision until mutation to avoid unnecessary memcpy.
        self._capacity_or_data = Self.FLAG_HAS_NUL_TERMINATOR

    fn __init__(out self, *, bytes: Span[Byte, *_]):
        """Construct a string by copying the data. This constructor is explicit
        because it can involve memory allocation.

        Args:
            bytes: The bytes to copy.
        """
        var length = len(bytes)
        self = Self(unsafe_uninit_length=length)
        memcpy(self.unsafe_ptr_mut(), bytes.unsafe_ptr(), length)

    fn __init__[T: Stringable](out self, value: T):
        """Initialize from a type conforming to `Stringable`.

        Parameters:
            T: The type conforming to Stringable.

        Args:
            value: The object to get the string representation of.
        """
        self = value.__str__()

    fn __init__[T: StringableRaising](out self, value: T) raises:
        """Initialize from a type conforming to `StringableRaising`.

        Parameters:
            T: The type conforming to Stringable.

        Args:
            value: The object to get the string representation of.

        Raises:
            If there is an error when computing the string representation of the type.
        """
        self = value.__str__()

    fn __init__[
        *Ts: Writable
    ](out self, *args: *Ts, sep: StaticString = "", end: StaticString = ""):
        """
        Construct a string by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.

        Examples:

        Construct a String from several `Writable` arguments:

        ```mojo
        var string = String(1, 2.0, "three", sep=", ")
        print(string) # "1, 2.0, three"
        ```
        """
        self = String()
        var buffer = _WriteBufferStack(self)
        alias length = args.__len__()

        @parameter
        for i in range(length):
            args[i].write_to(buffer)

            if i < length - 1:
                sep.write_to(buffer)

        end.write_to(buffer)
        buffer.flush()

    # TODO(MOCO-1791): Default arguments and param inference aren't powerful
    # to declare sep/end as StringSlice.
    @staticmethod
    fn __init__[
        *Ts: Writable
    ](
        out self,
        args: VariadicPack[_, _, Writable, *Ts],
        sep: StaticString = "",
        end: StaticString = "",
    ):
        """
        Construct a string by passing a variadic pack.

        Args:
            args: A VariadicPack of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.

        Examples:

        ```mojo
        fn variadic_pack_to_string[
            *Ts: Writable,
        ](*args: *Ts) -> String:
            return String(args)

        string = variadic_pack_to_string(1, ", ", 2.0, ", ", "three")
        %# from testing import assert_equal
        %# assert_equal(string, "1, 2.0, three")
        ```
        .
        """
        self = String()
        var buffer = _WriteBufferStack(self)
        alias length = args.__len__()

        @parameter
        for i in range(length):
            args[i].write_to(buffer)

            if i < length - 1:
                sep.write_to(buffer)

        end.write_to(buffer)
        buffer.flush()

    @always_inline("nodebug")
    fn copy(self) -> Self:
        """Explicitly copy the provided value.

        Returns:
            A copy of the value.
        """
        return self  # Just use the implicit copyinit.

    @always_inline("nodebug")
    fn __init__(out self, *, unsafe_uninit_length: UInt):
        """Construct a String with the specified length, with uninitialized
        memory. This is unsafe, as it relies on the caller initializing the
        elements with unsafe operations, not assigning over the uninitialized
        data.

        Args:
            unsafe_uninit_length: The number of bytes to allocate.
        """
        self = Self(capacity=unsafe_uninit_length)
        self.set_byte_length(unsafe_uninit_length)

    fn __init__(
        out self,
        *,
        unsafe_from_utf8_ptr: UnsafePointer[c_char, mut=_, origin=_],
    ):
        """Creates a string from a UTF-8 encoded nul-terminated pointer.

        Args:
            unsafe_from_utf8_ptr: An `UnsafePointer[Byte]` of null-terminated bytes encoded in UTF-8.

        Safety:
            - `unsafe_from_utf8_ptr` MUST be valid UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` MUST be null terminated.
        """
        # Copy the data.
        self = String(StringSlice(unsafe_from_utf8_ptr=unsafe_from_utf8_ptr))

    fn __init__(
        out self, *, unsafe_from_utf8_ptr: UnsafePointer[UInt8, mut=_, origin=_]
    ):
        """Creates a string from a UTF-8 encoded nul-terminated pointer.

        Args:
            unsafe_from_utf8_ptr: An `UnsafePointer[Byte]` of null-terminated bytes encoded in UTF-8.

        Safety:
            - `unsafe_from_utf8_ptr` MUST be valid UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` MUST be null terminated.
        """
        # Copy the data.
        self = String(StringSlice(unsafe_from_utf8_ptr=unsafe_from_utf8_ptr))

    @always_inline("nodebug")
    fn __moveinit__(out self, owned other: Self):
        """Move initialize the string from another string.

        Args:
            other: The string to move.
        """
        self._ptr_or_data = other._ptr_or_data
        self._len_or_data = other._len_or_data
        self._capacity_or_data = other._capacity_or_data

    @always_inline("nodebug")
    fn __copyinit__(out self, other: Self):
        """Copy initialize the string from another string.

        Args:
            other: The string to copy.
        """
        # Keep inline strings inline, and static strings static.
        self._ptr_or_data = other._ptr_or_data
        self._len_or_data = other._len_or_data
        self._capacity_or_data = other._capacity_or_data

        # Increment the refcount if it has a mutable buffer.
        self._add_ref()

    # ===------------------------------------------------------------------=== #
    # Capacity Field Helpers
    # ===------------------------------------------------------------------=== #

    # This includes getting and setting flags from the capcity field such as
    # null terminator, inline, and indirect. If indirect the length is also
    # stored in the capacity field.

    @always_inline("nodebug")
    fn capacity(self) -> UInt:
        # Max inline capacity before reallocation.
        if self._is_inline():
            return Self.INLINE_CAPACITY
        if not self._is_ref_counted():
            return self._len_or_data
        return self._capacity_or_data << 3

    @always_inline("nodebug")
    fn _has_nul_terminator(self) -> Bool:
        return Bool(self._capacity_or_data & Self.FLAG_HAS_NUL_TERMINATOR)

    @always_inline("nodebug")
    fn _clear_nul_terminator(mut self):
        self._capacity_or_data &= ~Self.FLAG_HAS_NUL_TERMINATOR

    @always_inline("nodebug")
    fn _is_inline(self) -> Bool:
        return Bool(self._capacity_or_data & Self.FLAG_IS_INLINE)

    @always_inline("nodebug")
    fn _set_ref_counted(mut self):
        self._capacity_or_data |= Self.FLAG_IS_REF_COUNTED

    @always_inline("nodebug")
    fn _is_ref_counted(self) -> Bool:
        return Bool(self._capacity_or_data & Self.FLAG_IS_REF_COUNTED)

    # ===------------------------------------------------------------------=== #
    # Pointer Field Helpers
    # ===------------------------------------------------------------------=== #

    # This includes helpers for the allocated atomic ref count used for
    # out-of-line strings, which is stored before the UTF-8 data.

    @always_inline("nodebug")
    fn _refcount(self) -> ref [self._ptr_or_data.origin] Atomic[DType.index]:
        # The header is stored before the string data.
        return (self._ptr_or_data - Self.REF_COUNT_SIZE).bitcast[
            Atomic[DType.index]
        ]()[]

    @always_inline("nodebug")
    fn _is_unique(mut self) -> Bool:
        """Return true if the refcount is 1."""
        if self._capacity_or_data & Self.FLAG_IS_REF_COUNTED:
            return self._refcount().load() == 1
        else:
            return False

    @always_inline("nodebug")
    fn _add_ref(mut self):
        """Atomically increment the refcount."""
        if self._capacity_or_data & Self.FLAG_IS_REF_COUNTED:
            _ = self._refcount().fetch_add(1)

    @always_inline("nodebug")
    fn _drop_ref(mut self):
        """Atomically decrement the refcount and deallocate self if the result
        hits zero."""
        # If indirect or inline we don't need to do anything.
        if self._capacity_or_data & Self.FLAG_IS_REF_COUNTED:
            var ptr = self._ptr_or_data - Self.REF_COUNT_SIZE
            var refcount = ptr.bitcast[Atomic[DType.index]]()
            if refcount[].fetch_sub(1) == 1:
                ptr.free()

    @staticmethod
    fn _alloc(capacity: Int) -> UnsafePointer[Byte]:
        """Allocate space for a new out-of-line string buffer."""
        var ptr = UnsafePointer[Byte].alloc(capacity + Self.REF_COUNT_SIZE)

        # Initialize the Atomic refcount into the header.
        __get_address_as_uninit_lvalue(
            ptr.bitcast[Atomic[DType.index]]().address
        ) = Atomic[DType.index](1)

        # Return a pointer to right after the header, which is where the string
        # data will be stored.
        return ptr + Self.REF_COUNT_SIZE

    # ===------------------------------------------------------------------=== #
    # Factory dunders
    # ===------------------------------------------------------------------=== #

    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """Write a byte span to this String.

        Args:
            bytes: The byte span to write to this String. Must NOT be
                null terminated.
        """
        self._iadd(bytes)

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """

        @parameter
        for i in range(args.__len__()):
            args[i].write_to(self)

    @staticmethod
    fn write[
        *Ts: Writable
    ](*args: *Ts, sep: StaticString = "", end: StaticString = "") -> Self:
        """Construct a string by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.

        Returns:
            A string formed by formatting the argument sequence.

        This is used only when reusing the `write_to` method for
        `__str__` in order to avoid an endless loop recalling
        the constructor.
        """
        var string = String()
        var buffer = _WriteBufferStack(string)
        alias length = args.__len__()

        @parameter
        for i in range(length):
            args[i].write_to(buffer)

            if i < length - 1:
                sep.write_to(buffer)

        end.write_to(buffer)
        buffer.flush()
        return string^

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    fn __getitem__[I: Indexer](self, idx: I) -> String:
        """Gets the character at the specified position.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The index value.

        Returns:
            A new string containing the character at the specified position.
        """
        # TODO(#933): implement this for unicode when we support llvm intrinsic evaluation at compile time
        var normalized_idx = normalize_index["String"](idx, len(self))
        var result = String(capacity=1)
        result.append_byte(self.unsafe_ptr()[normalized_idx])
        return result^

    fn __getitem__(self, span: Slice) -> String:
        """Gets the sequence of characters at the specified positions.

        Args:
            span: A slice that specifies positions of the new substring.

        Returns:
            A new string containing the string at the specified positions.
        """
        var start: Int
        var end: Int
        var step: Int
        # TODO(#933): implement this for unicode when we support llvm intrinsic evaluation at compile time

        start, end, step = span.indices(self.byte_length())
        var r = range(start, end, step)
        if step == 1:
            return String(
                StringSlice(ptr=self.unsafe_ptr() + start, length=len(r))
            )

        var result = String(capacity=len(r))
        var ptr = self.unsafe_ptr()
        for i in r:
            result.append_byte(ptr[i])
        return result^

    fn __eq__(self, rhs: String) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        # Early exit if lengths differ
        var self_len = self.byte_length()
        var other_len = rhs.byte_length()
        if self_len != other_len:
            return False
        var self_ptr = self.unsafe_ptr()
        var rhs_ptr = rhs.unsafe_ptr()
        # same pointer and length, so equal
        if self_len == 0 or self_ptr == rhs_ptr:
            return True
        # Compare memory directly
        return memcmp(self_ptr, rhs_ptr, self_len) == 0

    @always_inline("nodebug")
    fn __eq__(self, other: StringSlice) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        return self.as_string_slice() == other

    @always_inline("nodebug")
    fn __ne__(self, other: String) -> Bool:
        """Compares two Strings if they do not have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are not equal and False otherwise.
        """
        return not (self == other)

    @always_inline("nodebug")
    fn __ne__(self, other: StringSlice) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        return self.as_string_slice() != other

    @always_inline("nodebug")
    fn __lt__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using LT comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True if this String is strictly less than the RHS String and False
            otherwise.
        """
        return self.as_string_slice() < rhs.as_string_slice()

    @always_inline("nodebug")
    fn __le__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using LE comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True iff this String is less than or equal to the RHS String.
        """
        return not (rhs < self)

    @always_inline("nodebug")
    fn __gt__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using GT comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True iff this String is strictly greater than the RHS String.
        """
        return rhs < self

    @always_inline("nodebug")
    fn __ge__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using GE comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True iff this String is greater than or equal to the RHS String.
        """
        return not (self < rhs)

    @staticmethod
    fn _add(lhs: Span[Byte], rhs: Span[Byte]) -> String:
        var lhs_len = len(lhs)
        var rhs_len = len(rhs)

        var result = String(unsafe_uninit_length=lhs_len + rhs_len)
        var result_ptr = result.unsafe_ptr_mut()
        memcpy(result_ptr, lhs.unsafe_ptr(), lhs_len)
        memcpy(result_ptr + lhs_len, rhs.unsafe_ptr(), rhs_len)
        return result^

    fn __add__(self, other: StringSlice) -> String:
        """Creates a string by appending a string slice at the end.

        Args:
            other: The string slice to append.

        Returns:
            The new constructed string.
        """
        return Self._add(self.as_bytes(), other.as_bytes())

    fn append_byte(mut self, byte: Byte):
        """Append a byte to the string.

        Args:
            byte: The byte to append.
        """
        self._clear_nul_terminator()
        var len = self.byte_length()
        self.reserve(len + 1)
        self.unsafe_ptr_mut()[len] = byte
        self.set_byte_length(len + 1)

    fn __radd__(self, other: StringSlice[mut=False]) -> String:
        """Creates a string by prepending another string slice to the start.

        Args:
            other: The string to prepend.

        Returns:
            The new constructed string.
        """
        return Self._add(other.as_bytes(), self.as_bytes())

    fn _iadd(mut self, other: Span[mut=False, Byte]):
        var other_len = len(other)
        if other_len == 0:
            return
        var old_len = self.byte_length()
        var new_len = old_len + other_len
        memcpy(
            self.unsafe_ptr_mut(new_len) + old_len,
            other.unsafe_ptr(),
            other_len,
        )
        self.set_byte_length(new_len)
        self._clear_nul_terminator()

    fn __iadd__(mut self, other: StringSlice[mut=False]):
        """Appends another string slice to this string.

        Args:
            other: The string to append.
        """
        self._iadd(other.as_bytes())

    @deprecated("Use `str.codepoints()` or `str.codepoint_slices()` instead.")
    fn __iter__(self) -> CodepointSliceIter[__origin_of(self)]:
        """Iterate over the string, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return self.codepoint_slices()

    fn __reversed__(self) -> CodepointSliceIter[__origin_of(self), False]:
        """Iterate backwards over the string, returning immutable references.

        Returns:
            A reversed iterator of references to the string elements.
        """
        return CodepointSliceIter[__origin_of(self), forward=False](self)

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __bool__(self) -> Bool:
        """Checks if the string is not empty.

        Returns:
            True if the string length is greater than zero, and False otherwise.
        """
        return self.byte_length() > 0

    fn __len__(self) -> Int:
        """Get the string length of in bytes.

        This function returns the number of bytes in the underlying UTF-8
        representation of the string.

        To get the number of Unicode codepoints in a string, use
        `len(str.codepoints())`.

        Returns:
            The string length in bytes.

        # Examples

        Query the length of a string, in bytes and Unicode codepoints:

        ```mojo
        from testing import assert_equal

        var s = "ನಮಸ್ಕಾರ"

        assert_equal(len(s), 21)
        assert_equal(len(s.codepoints()), 7)
        ```

        Strings containing only ASCII characters have the same byte and
        Unicode codepoint length:

        ```mojo
        from testing import assert_equal

        var s = "abc"

        assert_equal(len(s), 3)
        assert_equal(len(s.codepoints()), 3)
        ```
        .
        """
        return self.byte_length()

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Gets the string itself.

        This method ensures that you can pass a `String` to a method that
        takes a `Stringable` value.

        Returns:
            The string itself.
        """
        return self

    fn __repr__(self) -> String:
        """Return a Mojo-compatible representation of the `String` instance.

        Returns:
            A new representation of the string.
        """
        return StringSlice(self).__repr__()

    @always_inline("nodebug")
    fn __fspath__(self) -> String:
        """Return the file system path representation (just the string itself).

        Returns:
          The file system path representation as a string.
        """
        return self

    fn to_python_object(var self) raises -> PythonObject:
        """Convert this value to a PythonObject.

        Returns:
            A PythonObject representing the value.
        """
        return PythonObject(self)

    fn __init__(out self, obj: PythonObject) raises:
        """Construct a `String` from a PythonObject.

        Args:
            obj: The PythonObject to convert from.

        Raises:
            An error if the conversion failed.
        """
        var str_obj = obj.__str__()
        self = String(StringSlice(unsafe_borrowed_obj=str_obj))
        # keep python object alive so the copy can occur
        _ = str_obj

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this string to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write_bytes(
            Span(ptr=self.unsafe_ptr(), length=self.byte_length())
        )

    fn join[*Ts: Writable](self, *elems: *Ts) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            Ts: The types of the elements.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """
        var sep = rebind[StaticString](  # FIXME(#4414): this should not be so
            StringSlice(ptr=self.unsafe_ptr(), length=self.byte_length())
        )
        return String(elems, sep=sep)

    fn join[
        T: Copyable & Movable & Writable
    ](self, elems: List[T, *_]) -> String:
        """Joins string elements using the current string as a delimiter.
        Defaults to writing to the stack if total bytes of `elems` is less than
        `buffer_size`, otherwise will allocate once to the heap and write
        directly into that. The `buffer_size` defaults to 4096 bytes to match
        the default page size on arm64 and x86-64.

        Parameters:
            T: The type of the elements. Must implement the `Copyable`,
                `Movable` and `Writable` traits.

        Args:
            elems: The input values.

        Returns:
            The joined string.

        Notes:
            - Defaults to writing directly to the string if the bytes
            fit in an inline `String`, otherwise will process it by chunks.
            - The `buffer_size` defaults to 4096 bytes to match the default
            page size on arm64 and x86-64, but you can increase this if you're
            joining a very large `List` of elements to write into the stack
            instead of the heap.
        """
        return self.as_string_slice().join(elems)

    fn codepoints(self) -> CodepointsIter[__origin_of(self)]:
        """Returns an iterator over the `Codepoint`s encoded in this string slice.

        Returns:
            An iterator type that returns successive `Codepoint` values stored in
            this string slice.

        # Examples

        Print the characters in a string:

        ```mojo
        from testing import assert_equal

        var s = "abc"
        var iter = s.codepoints()
        assert_equal(iter.__next__(), Codepoint.ord("a"))
        assert_equal(iter.__next__(), Codepoint.ord("b"))
        assert_equal(iter.__next__(), Codepoint.ord("c"))
        assert_equal(iter.__has_next__(), False)
        ```

        `codepoints()` iterates over Unicode codepoints, and supports multibyte
        codepoints:

        ```mojo
        from testing import assert_equal

        # A visual character composed of a combining sequence of 2 codepoints.
        var s = "á"
        assert_equal(s.byte_length(), 3)

        var iter = s.codepoints()
        assert_equal(iter.__next__(), Codepoint.ord("a"))
         # U+0301 Combining Acute Accent
        assert_equal(iter.__next__().to_u32(), 0x0301)
        assert_equal(iter.__has_next__(), False)
        ```
        .
        """
        return self.as_string_slice().codepoints()

    fn codepoint_slices(self) -> CodepointSliceIter[__origin_of(self)]:
        """Returns an iterator over single-character slices of this string.

        Each returned slice points to a single Unicode codepoint encoded in the
        underlying UTF-8 representation of this string.

        Returns:
            An iterator of references to the string elements.

        # Examples

        Iterate over the character slices in a string:

        ```mojo
        from testing import assert_equal, assert_true

        var s = "abc"
        var iter = s.codepoint_slices()
        assert_true(iter.__next__() == "a")
        assert_true(iter.__next__() == "b")
        assert_true(iter.__next__() == "c")
        assert_equal(iter.__has_next__(), False)
        ```
        .
        """
        return self.as_string_slice().codepoint_slices()

    @always_inline("nodebug")
    fn unsafe_ptr(
        self,
    ) -> UnsafePointer[Byte, mut=False, origin = __origin_of(self)]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The pointer to the underlying memory.
        """

        if self._is_inline():
            # The string itself holds the data.
            return (
                UnsafePointer(to=self)
                .bitcast[Byte]()
                .origin_cast[False, __origin_of(self)]()
            )
        else:
            return self._ptr_or_data.origin_cast[False, __origin_of(self)]()

    fn unsafe_ptr_mut(
        mut self, var capacity: UInt = 0
    ) -> UnsafePointer[Byte, mut=True, origin = __origin_of(self)]:
        """Retrieves a mutable pointer to the unique underlying memory. Passing
        a larger capacity will reallocate the string to the new capacity if
        larger than the existing capacity, allowing you to write more data.

        Args:
            capacity: The new capacity of the string.

        Returns:
            The pointer to the underlying memory.
        """
        var new_cap = max(self.capacity(), capacity)
        # Decide on strategy for making the string mutable
        if new_cap <= Self.INLINE_CAPACITY:
            if not self._is_inline():
                self._inline_string()
        elif not self._is_unique() or new_cap > self.capacity():
            self._realloc_mutable(new_cap)

        return self.unsafe_ptr().origin_cast[True, __origin_of(self)]()

    fn unsafe_cstr_ptr(
        mut self,
    ) -> UnsafePointer[c_char, mut=False, origin = __origin_of(self)]:
        """Retrieves a C-string-compatible pointer to the underlying memory.

        The returned pointer is guaranteed to be null, or NUL terminated.

        Returns:
            The pointer to the underlying memory.
        """
        # Add a nul terminator, making the string mutable if not already
        if not self._has_nul_terminator():
            var ptr = self.unsafe_ptr_mut(capacity=len(self) + 1)
            var len = self.byte_length()
            ptr[len] = 0
            self._capacity_or_data |= Self.FLAG_HAS_NUL_TERMINATOR
            return self.unsafe_ptr().bitcast[c_char]()

        return self.unsafe_ptr().bitcast[c_char]()

    fn as_bytes(self) -> Span[Byte, __origin_of(self)]:
        """Returns a contiguous slice of the bytes owned by this string.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.
        """

        return Span[Byte, __origin_of(self)](
            ptr=self.unsafe_ptr(), length=self.byte_length()
        )

    fn as_bytes_mut(mut self) -> Span[Byte, __origin_of(self)]:
        """Returns a mutable contiguous slice of the bytes owned by this string.
        This name has a _mut suffix so the as_bytes() method doesn't have to
        guarantee mutability.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.
        """
        return Span[Byte, __origin_of(self)](
            ptr=self.unsafe_ptr_mut(), length=self.byte_length()
        )

    fn as_string_slice(self) -> StringSlice[__origin_of(self)]:
        """Returns a string slice of the data owned by this string.

        Returns:
            A string slice pointing to the data owned by this string.
        """
        # FIXME(MSTDL-160):
        #   Enforce UTF-8 encoding in String so this is actually
        #   guaranteed to be valid.
        return StringSlice(unsafe_from_utf8=self.as_bytes())

    fn as_string_slice_mut(mut self) -> StringSlice[__origin_of(self)]:
        """Returns a mutable string slice of the data owned by this string.

        Returns:
            A string slice pointing to the data owned by this string.
        """
        return StringSlice(unsafe_from_utf8=self.as_bytes_mut())

    fn byte_length(self) -> Int:
        """Get the string length in bytes.

        Returns:
            The length of this string in bytes.
        """
        if self._is_inline():
            return (
                self._capacity_or_data & Self.INLINE_LENGTH_MASK
            ) >> Self.INLINE_LENGTH_START
        else:
            return self._len_or_data

    fn set_byte_length(mut self, new_len: Int):
        if self._is_inline():
            self._capacity_or_data = (
                self._capacity_or_data & ~Self.INLINE_LENGTH_MASK
            ) | (new_len << Self.INLINE_LENGTH_START)
        else:
            self._len_or_data = new_len

    fn count(self, substr: StringSlice) -> Int:
        """Return the number of non-overlapping occurrences of substring
        `substr` in the string.

        If sub is empty, returns the number of empty strings between characters
        which is the length of the string plus one.

        Args:
          substr: The substring to count.

        Returns:
          The number of occurrences of `substr`.
        """
        return self.as_string_slice().count(substr)

    fn __contains__(self, substr: StringSlice) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return substr in self.as_string_slice()

    fn find(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """

        return self.as_string_slice().find(substr, start)

    fn rfind(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset of the last occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """

        return self.as_string_slice().rfind(substr, start=start)

    fn isspace(self) -> Bool:
        """Determines whether every character in the given String is a
        python whitespace String. This corresponds to Python's
        [universal separators](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if the whole String is made up of whitespace characters
                listed above, otherwise False.
        """
        return self.as_string_slice().isspace()

    fn split(self, sep: StringSlice, maxsplit: Int = -1) -> List[String]:
        """Split the string by a separator.

        Args:
            sep: The string to split on.
            maxsplit: The maximum amount of items to split from String.
                Defaults to unlimited.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting a space
        _ = "hello world".split(" ") # ["hello", "world"]
        # Splitting adjacent separators
        _ = "hello,,world".split(",") # ["hello", "", "world"]
        # Splitting with maxsplit
        _ = "1,2,3".split(",", 1) # ['1', '2,3']
        # Splitting with an empty separator
        _ = "123".split("") # ["", "1", "2", "3", ""]
        ```
        """
        return _to_string_list(
            self.as_string_slice().split(sep, maxsplit=maxsplit)
        )

    fn split(self, sep: NoneType = None, maxsplit: Int = -1) -> List[String]:
        """Split the string by every Whitespace separator.

        Args:
            sep: None.
            maxsplit: The maximum amount of items to split from String. Defaults
                to unlimited.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting an empty string or filled with whitespaces
        _ = "      ".split() # []
        _ = "".split() # []

        # Splitting a string with leading, trailing, and middle whitespaces
        _ = "      hello    world     ".split() # ["hello", "world"]
        # Splitting adjacent universal newlines:
        _ = "hello \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029world".split()  # ["hello", "world"]
        ```
        """
        return _to_string_list(
            self.as_string_slice().split(sep, maxsplit=maxsplit)
        )

    fn splitlines(self, keepends: Bool = False) -> List[String]:
        """Split the string at line boundaries. This corresponds to Python's
        [universal newlines:](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """
        return _to_string_list(self.as_string_slice().splitlines(keepends))

    fn replace(self, old: StringSlice, new: StringSlice) -> String:
        """Return a copy of the string with all occurrences of substring `old`
        if replaced by `new`.

        Args:
            old: The substring to replace.
            new: The substring to replace with.

        Returns:
            The string where all occurrences of `old` are replaced with `new`.
        """
        return StringSlice(self).replace(old, new)

    fn strip(self, chars: StringSlice) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading and trailing characters
        removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading or trailing characters.
        """

        return self.lstrip(chars).rstrip(chars)

    fn strip(self) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading and trailing whitespaces
        removed. This only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A copy of the string with no leading or trailing whitespaces.
        """
        return self.lstrip().rstrip()

    fn rstrip(self, chars: StringSlice) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with trailing characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no trailing characters.
        """

        return self.as_string_slice().rstrip(chars)

    fn rstrip(self) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with trailing whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A copy of the string with no trailing whitespaces.
        """
        return self.as_string_slice().rstrip()

    fn lstrip(self, chars: StringSlice) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading characters.
        """

        return self.as_string_slice().lstrip(chars)

    fn lstrip(self) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A copy of the string with no leading whitespaces.
        """
        return self.as_string_slice().lstrip()

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher.update(self.as_string_slice())

    fn lower(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to lowercase.

        Returns:
            A new string where cased letters have been converted to lowercase.
        """

        return self.as_string_slice().lower()

    fn upper(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to uppercase.

        Returns:
            A new string where cased letters have been converted to uppercase.
        """

        return self.as_string_slice().upper()

    fn startswith(
        self, prefix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the string starts with the specified prefix between start
        and end positions. Returns True if found and False otherwise.

        Args:
            prefix: The prefix to check.
            start: The start offset from which to check.
            end: The end offset from which to check.

        Returns:
            True if the `self[start:end]` is prefixed by the input prefix.
        """
        return self.as_string_slice().startswith(prefix, start, end)

    fn endswith(
        self, suffix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the string end with the specified suffix between start
        and end positions. Returns True if found and False otherwise.

        Args:
            suffix: The suffix to check.
            start: The start offset from which to check.
            end: The end offset from which to check.

        Returns:
            True if the `self[start:end]` is suffixed by the input suffix.
        """
        return self.as_string_slice().endswith(suffix, start, end)

    fn removeprefix(
        self, prefix: StringSlice, /
    ) -> StringSlice[__origin_of(self)]:
        """Returns a new string with the prefix removed if it was present.

        Args:
            prefix: The prefix to remove from the string.

        Returns:
            `string[len(prefix):]` if the string starts with the prefix string,
            or a copy of the original string otherwise.

        Examples:

        ```mojo
        print(String('TestHook').removeprefix('Test')) # 'Hook'
        print(String('BaseTestCase').removeprefix('Test')) # 'BaseTestCase'
        ```
        """
        return self.as_string_slice().removeprefix(prefix)

    fn removesuffix(
        self, suffix: StringSlice, /
    ) -> StringSlice[__origin_of(self)]:
        """Returns a new string with the suffix removed if it was present.

        Args:
            suffix: The suffix to remove from the string.

        Returns:
            `string[:-len(suffix)]` if the string ends with the suffix string,
            or a copy of the original string otherwise.

        Examples:

        ```mojo
        print(String('TestHook').removesuffix('Hook')) # 'Test'
        print(String('BaseTestCase').removesuffix('Test')) # 'BaseTestCase'
        ```
        """
        return self.as_string_slice().removesuffix(suffix)

    fn __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.
        If the string cannot be parsed as an int, an error is raised.

        Returns:
            An integer value that represents the string, or otherwise raises.
        """
        return atol(self)

    fn __float__(self) raises -> Float64:
        """Parses the string as a float point number and returns that value. If
        the string cannot be parsed as a float, an error is raised.

        Returns:
            A float value that represents the string, or otherwise raises.
        """
        return atof(self)

    fn __mul__(self, n: Int) -> String:
        """Concatenates the string `n` times.

        Args:
            n : The number of times to concatenate the string.

        Returns:
            The string concatenated `n` times.
        """
        return self.as_string_slice() * n

    fn format[*Ts: _CurlyEntryFormattable](self, *args: *Ts) raises -> String:
        """Produce a formatted string using the current string as a template.

        The template, or "format string" can contain literal text and/or
        replacement fields delimited with curly braces (`{}`). Returns a copy of
        the format string with the replacement fields replaced with string
        representations of the `args` arguments.

        For more information, see the discussion in the
        [`format` module](/mojo/stdlib/collections/string/format/).

        Args:
            args: The substitution values.

        Parameters:
            Ts: The types of substitution values that implement `Representable`
                and `Stringable` (to be changed and made more flexible).

        Returns:
            The template with the given values substituted.

        Example:

        ```mojo
        # Manual indexing:
        print("{0} {1} {0}".format("Mojo", 1.125)) # Mojo 1.125 Mojo
        # Automatic indexing:
        print("{} {}".format(True, "hello world")) # True hello world
        ```
        """
        return _FormatCurlyEntry.format(self, args)

    fn isdigit(self) -> Bool:
        """A string is a digit string if all characters in the string are digits
        and there is at least one character in the string.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are digits and it's not empty else False.
        """
        return self.as_string_slice().is_ascii_digit()

    fn isupper(self) -> Bool:
        """Returns True if all cased characters in the string are uppercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are uppercase and there
            is at least one cased character, False otherwise.
        """
        return self.as_string_slice().isupper()

    fn islower(self) -> Bool:
        """Returns True if all cased characters in the string are lowercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are lowercase and there
            is at least one cased character, False otherwise.
        """
        return self.as_string_slice().islower()

    fn isprintable(self) -> Bool:
        """Returns True if all characters in the string are ASCII printable.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are printable else False.
        """
        return self.as_string_slice().is_ascii_printable()

    fn rjust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string right justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns right justified string, or self if width is not bigger than self length.
        """
        return self.as_string_slice().rjust(width, fillchar)

    fn ljust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string left justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns left justified string, or self if width is not bigger than self length.
        """
        return self.as_string_slice().ljust(width, fillchar)

    fn center(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string center justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns center justified string, or self if width is not bigger than self length.
        """
        return self.as_string_slice().center(width, fillchar)

    fn resize(mut self, length: Int, fill_byte: UInt8 = 0):
        """Resize the string to a new length.

        Args:
            length: The new length of the string.
            fill_byte: The byte to fill any new space with.

        Notes:
            If the new length is greater than the current length, the string is
            extended by the difference, and the new bytes are initialized to
            `fill_byte`.
        """
        self._clear_nul_terminator()
        var old_len = self.byte_length()
        if length > old_len:
            memset(
                self.unsafe_ptr_mut(length) + old_len,
                fill_byte,
                length - old_len,
            )
        self.set_byte_length(length)

    fn resize(mut self, *, unsafe_uninit_length: Int):
        """Resizes the string to the given new size leaving any new data
        uninitialized.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        string is extended and the new data is left uninitialized.

        Args:
            unsafe_uninit_length: The new size.
        """
        self._clear_nul_terminator()
        if unsafe_uninit_length > self.capacity():
            self.reserve(unsafe_uninit_length)
        self.set_byte_length(unsafe_uninit_length)

    fn reserve(mut self, new_capacity: UInt):
        """Reserves the requested capacity.

        Args:
            new_capacity: The new capacity in stored bytes.

        Notes:
            If the current capacity is greater or equal, this is a no-op.
            Otherwise, the storage is reallocated and the data is moved.
        """
        if new_capacity <= self.capacity():
            return
        self._realloc_mutable(new_capacity)

    # Make a string mutable on the stack.
    fn _inline_string(mut self):
        var length = len(self)
        var new_string = Self()
        new_string.set_byte_length(length)
        var dst = UnsafePointer(to=new_string).bitcast[Byte]()
        var src = self.unsafe_ptr()
        for i in range(length):
            dst[i] = src[i]
        self = new_string^

    # This is the out-of-line implementation of reserve called when we need
    # to grow the capacity of the string. Make sure our capacity at least
    # doubles to avoid O(n^2) behavior, and make use of extra space if it exists.
    fn _realloc_mutable(mut self, capacity: UInt):
        # Get these fields before we change _capacity_or_data
        var byte_len = self.byte_length()
        var old_ptr = self.unsafe_ptr()
        var new_capacity = (max(capacity, self.capacity() * 2) + 7) >> 3
        var new_ptr = self._alloc(new_capacity << 3)
        memcpy(new_ptr, old_ptr, byte_len)
        # If mutable buffer drop the ref count
        self._drop_ref()
        self._len_or_data = byte_len
        self._ptr_or_data = new_ptr
        # Assign directly to clear existing flags
        self._capacity_or_data = new_capacity
        self._set_ref_counted()


# ===----------------------------------------------------------------------=== #
# ord
# ===----------------------------------------------------------------------=== #


fn ord(s: StringSlice) -> Int:
    """Returns an integer that represents the codepoint of a single-character
    string.

    Given a string containing a single character `Codepoint`, return an integer
    representing the codepoint of that character. For example, `ord("a")`
    returns the integer `97`. This is the inverse of the `chr()` function.

    This function is in the prelude, so you don't need to import it.

    Args:
        s: The input string, which must contain only a single- character.

    Returns:
        An integer representing the code point of the given character.
    """
    return Int(Codepoint.ord(s))


# ===----------------------------------------------------------------------=== #
# chr
# ===----------------------------------------------------------------------=== #


fn chr(c: Int) -> String:
    """Returns a String based on the given Unicode code point. This is the
    inverse of the `ord()` function.

    This function is in the prelude, so you don't need to import it.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a single character based on the given code point.

    Example:
    ```mojo
    print(chr(97), chr(8364)) # "a €"
    ```
    """

    if c < 0b1000_0000:  # 1 byte ASCII char
        var str = String(capacity=1)
        str.append_byte(c)
        return str^

    var char_opt = Codepoint.from_u32(c)
    if not char_opt:
        # TODO: Raise ValueError instead.
        return abort[String](
            String("chr(", c, ") is not a valid Unicode codepoint")
        )

    # SAFETY: We just checked that `char` is present.
    return String(char_opt.unsafe_value())


# ===----------------------------------------------------------------------=== #
# ascii
# ===----------------------------------------------------------------------=== #


fn _chr_ascii(c: UInt8) -> String:
    """Returns a string based on the given ASCII code point.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a single character based on the given code point.
    """
    var result = String(capacity=1)
    result.append_byte(c)
    return result


fn _repr_ascii(c: UInt8) -> String:
    """Returns a printable representation of the given ASCII code point.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a representation of the given code point.
    """
    alias ord_tab = ord("\t")
    alias ord_new_line = ord("\n")
    alias ord_carriage_return = ord("\r")
    alias ord_back_slash = ord("\\")

    if c == ord_back_slash:
        return r"\\"
    elif Codepoint(c).is_ascii_printable():
        return _chr_ascii(c)
    elif c == ord_tab:
        return r"\t"
    elif c == ord_new_line:
        return r"\n"
    elif c == ord_carriage_return:
        return r"\r"
    else:
        var uc = c.cast[DType.uint8]()
        if uc < 16:
            return hex(uc, prefix=r"\x0")
        else:
            return hex(uc, prefix=r"\x")


fn ascii(value: StringSlice) -> String:
    """Get the ASCII representation of the object.

    Args:
        value: The object to get the ASCII representation of.

    Returns:
        A string containing the ASCII representation of the object.
    """
    alias ord_squote = ord("'")
    var result = String()
    var use_dquote = False

    for idx in range(len(value._slice)):
        var char = value._slice[idx]
        result += _repr_ascii(char)
        use_dquote = use_dquote or (char == ord_squote)

    if use_dquote:
        return '"' + result + '"'
    else:
        return "'" + result + "'"


# ===----------------------------------------------------------------------=== #
# atol
# ===----------------------------------------------------------------------=== #


fn atol(str_slice: StringSlice, base: Int = 10) raises -> Int:
    """Parses and returns the given string as an integer in the given base.

    If base is set to 0, the string is parsed as an integer literal, with the
    following considerations:
    - '0b' or '0B' prefix indicates binary (base 2)
    - '0o' or '0O' prefix indicates octal (base 8)
    - '0x' or '0X' prefix indicates hexadecimal (base 16)
    - Without a prefix, it's treated as decimal (base 10)

    This follows [Python's integer literals format](
    https://docs.python.org/3/reference/lexical_analysis.html#integers).

    This function is in the prelude, so you don't need to import it.

    Args:
        str_slice: A string to be parsed as an integer in the given base.
        base: Base used for conversion, value must be between 2 and 36, or 0.

    Returns:
        An integer value that represents the string.

    Raises:
        If the given string cannot be parsed as an integer value or if an
        incorrect base is provided.

    Examples:

    ```text
    >>> atol("32")
    32
    >>> atol("FF", 16)
    255
    >>> atol("0xFF", 0)
    255
    >>> atol("0b1010", 0)
    10
    ```
    """

    if (base != 0) and (base < 2 or base > 36):
        raise Error("Base must be >= 2 and <= 36, or 0.")
    if not str_slice:
        raise Error(_str_to_base_error(base, str_slice))

    var real_base: Int
    var ord_num_max: Int

    var ord_letter_max = (-1, -1)
    var result = 0
    var is_negative: Bool
    var has_prefix: Bool
    var start: Int
    var str_len = str_slice.byte_length()

    start, is_negative = _trim_and_handle_sign(str_slice, str_len)

    alias ord_0 = ord("0")
    alias ord_letter_min = (ord("a"), ord("A"))
    alias ord_underscore = ord("_")

    if base == 0:
        var real_base_new_start = _identify_base(str_slice, start)
        real_base = real_base_new_start[0]
        start = real_base_new_start[1]
        has_prefix = real_base != 10
        if real_base == -1:
            raise Error(_str_to_base_error(base, str_slice))
    else:
        start, has_prefix = _handle_base_prefix(start, str_slice, str_len, base)
        real_base = base

    if real_base <= 10:
        ord_num_max = ord(String(real_base - 1))
    else:
        ord_num_max = ord("9")
        ord_letter_max = (
            ord("a") + (real_base - 11),
            ord("A") + (real_base - 11),
        )

    var buff = str_slice.unsafe_ptr()
    var found_valid_chars_after_start = False
    var has_space_after_number = False

    # Prefixed integer literals with real_base 2, 8, 16 may begin with leading
    # underscores under the conditions they have a prefix
    var was_last_digit_underscore = not (real_base in (2, 8, 16) and has_prefix)
    for pos in range(start, str_len):
        var ord_current = Int(buff[pos])
        if ord_current == ord_underscore:
            if was_last_digit_underscore:
                raise Error(_str_to_base_error(base, str_slice))
            else:
                was_last_digit_underscore = True
                continue
        else:
            was_last_digit_underscore = False
        if ord_0 <= ord_current <= ord_num_max:
            result += ord_current - ord_0
            found_valid_chars_after_start = True
        elif ord_letter_min[0] <= ord_current <= ord_letter_max[0]:
            result += ord_current - ord_letter_min[0] + 10
            found_valid_chars_after_start = True
        elif ord_letter_min[1] <= ord_current <= ord_letter_max[1]:
            result += ord_current - ord_letter_min[1] + 10
            found_valid_chars_after_start = True
        elif Codepoint(UInt8(ord_current)).is_posix_space():
            has_space_after_number = True
            start = pos + 1
            break
        else:
            raise Error(_str_to_base_error(base, str_slice))
        if pos + 1 < str_len and not Codepoint(buff[pos + 1]).is_posix_space():
            var nextresult = result * real_base
            if nextresult < result:
                raise Error(
                    _str_to_base_error(base, str_slice)
                    + " String expresses an integer too large to store in Int."
                )
            result = nextresult

    if was_last_digit_underscore or (not found_valid_chars_after_start):
        raise Error(_str_to_base_error(base, str_slice))

    if has_space_after_number:
        for pos in range(start, str_len):
            if not Codepoint(buff[pos]).is_posix_space():
                raise Error(_str_to_base_error(base, str_slice))
    if is_negative:
        result = -result
    return result


fn _trim_and_handle_sign(str_slice: StringSlice, str_len: Int) -> (Int, Bool):
    """Trims leading whitespace, handles the sign of the number in the string.

    Args:
        str_slice: A StringSlice containing the number to parse.
        str_len: The length of the string.

    Returns:
        A tuple containing:
        - The starting index of the number after whitespace and sign.
        - A boolean indicating whether the number is negative.
    """
    var buff = str_slice.unsafe_ptr()
    var start: Int = 0
    while start < str_len and Codepoint(buff[start]).is_posix_space():
        start += 1
    var p: Bool = buff[start] == ord("+")
    var n: Bool = buff[start] == ord("-")
    return start + (Int(p) or Int(n)), n


fn _handle_base_prefix(
    pos: Int, str_slice: StringSlice, str_len: Int, base: Int
) -> (Int, Bool):
    """Adjusts the starting position if a valid base prefix is present.

    Handles "0b"/"0B" for base 2, "0o"/"0O" for base 8, and "0x"/"0X" for base
    16. Only adjusts if the base matches the prefix.

    Args:
        pos: Current position in the string.
        str_slice: The input StringSlice.
        str_len: Length of the input string.
        base: The specified base.

    Returns:
        A tuple containing:
            - Updated position after the prefix, if applicable.
            - A boolean indicating if the prefix was valid for the given base.
    """
    var start = pos
    var buff = str_slice.unsafe_ptr()
    if start + 1 < str_len:
        var prefix_char = chr(Int(buff[start + 1]))
        if buff[start] == ord("0") and (
            (base == 2 and (prefix_char == "b" or prefix_char == "B"))
            or (base == 8 and (prefix_char == "o" or prefix_char == "O"))
            or (base == 16 and (prefix_char == "x" or prefix_char == "X"))
        ):
            start += 2
    return start, start != pos


fn _str_to_base_error(base: Int, str_slice: StringSlice) -> String:
    return String(
        "String is not convertible to integer with base ",
        base,
        ": '",
        str_slice,
        "'",
    )


fn _identify_base(str_slice: StringSlice, start: Int) -> Tuple[Int, Int]:
    var length = str_slice.byte_length()
    # just 1 digit, assume base 10
    if start == (length - 1):
        return 10, start
    if str_slice[start] == "0":
        var second_digit = str_slice[start + 1]
        if second_digit == "b" or second_digit == "B":
            return 2, start + 2
        if second_digit == "o" or second_digit == "O":
            return 8, start + 2
        if second_digit == "x" or second_digit == "X":
            return 16, start + 2
        # checking for special case of all "0", "_" are also allowed
        var was_last_character_underscore = False
        for i in range(start + 1, length):
            if str_slice[i] == "_":
                if was_last_character_underscore:
                    return -1, -1
                else:
                    was_last_character_underscore = True
                    continue
            else:
                was_last_character_underscore = False
            if str_slice[i] != "0":
                return -1, -1
    elif ord("1") <= ord(str_slice[start]) <= ord("9"):
        return 10, start
    else:
        return -1, -1

    return 10, start


fn _atof_error[reason: StaticString = ""](str_ref: StringSlice) -> Error:
    @parameter
    if reason:
        return Error(
            "String is not convertible to float: '",
            str_ref,
            "' because ",
            reason,
        )
    return Error("String is not convertible to float: '", str_ref, "'")


fn atof(str_slice: StringSlice) raises -> Float64:
    """Parses the given string as a floating point and returns that value.

    For example, `atof("2.25")` returns `2.25`.

    This function is in the prelude, so you don't need to import it.

    Raises:
        If the given string cannot be parsed as an floating point value, for
        example in `atof("hi")`.

    Args:
        str_slice: A string to be parsed as a floating point.

    Returns:
        An floating point value that represents the string, or otherwise raises.
    """
    return _atof(str_slice)


# ===----------------------------------------------------------------------=== #
# Other utilities
# ===----------------------------------------------------------------------=== #


fn _toggle_ascii_case(char: UInt8) -> UInt8:
    """Assuming char is a cased ASCII character, this function will return the
    opposite-cased letter.
    """

    # ASCII defines A-Z and a-z as differing only in their 6th bit,
    # so converting is as easy as a bit flip.
    return char ^ (1 << 5)


fn _calc_initial_buffer_size_int32(n0: Int) -> Int:
    # See https://commaok.xyz/post/lookup_tables/ and
    # https://lemire.me/blog/2021/06/03/computing-the-number-of-digits-of-an-integer-even-faster/
    # for a description.
    alias lookup_table = VariadicList[Int](
        4294967296,
        8589934582,
        8589934582,
        8589934582,
        12884901788,
        12884901788,
        12884901788,
        17179868184,
        17179868184,
        17179868184,
        21474826480,
        21474826480,
        21474826480,
        21474826480,
        25769703776,
        25769703776,
        25769703776,
        30063771072,
        30063771072,
        30063771072,
        34349738368,
        34349738368,
        34349738368,
        34349738368,
        38554705664,
        38554705664,
        38554705664,
        41949672960,
        41949672960,
        41949672960,
        42949672960,
        42949672960,
    )
    var n = UInt32(n0)
    var log2 = Int(
        (bitwidthof[DType.uint32]() - 1) ^ count_leading_zeros(n | 1)
    )
    return (n0 + lookup_table[Int(log2)]) >> 32


fn _calc_initial_buffer_size_int64(n0: UInt64) -> Int:
    var result: Int = 1
    var n = n0
    while True:
        if n < 10:
            return result
        if n < 100:
            return result + 1
        if n < 1_000:
            return result + 2
        if n < 10_000:
            return result + 3
        n //= 10_000
        result += 4


fn _calc_initial_buffer_size(n0: Int) -> Int:
    var sign = 0 if n0 > 0 else 1

    # Add 1 for the terminator
    return sign + n0._decimal_digit_count() + 1


fn _calc_initial_buffer_size(n: Float64) -> Int:
    return 128 + 1  # Add 1 for the terminator


fn _calc_initial_buffer_size[dtype: DType](n0: Scalar[dtype]) -> Int:
    @parameter
    if dtype.is_integral():
        var n = abs(n0)
        var sign = 0 if n0 > 0 else 1

        @parameter
        if is_32bit() or bitwidthof[dtype]() <= 32:
            return sign + _calc_initial_buffer_size_int32(Int(n)) + 1
        else:
            return (
                sign
                + _calc_initial_buffer_size_int64(n.cast[DType.uint64]())
                + 1
            )

    return 128 + 1  # Add 1 for the terminator


fn _calc_format_buffer_size[dtype: DType]() -> Int:
    """Returns a buffer size in bytes that is large enough to store a formatted
    number of the specified dtype.
    """

    # TODO:
    #   Use a smaller size based on the `dtype`, e.g. we don't need as much
    #   space to store a formatted int8 as a float64.
    @parameter
    if dtype.is_integral():
        return 64 + 1
    else:
        return 128 + 1  # Add 1 for the terminator
