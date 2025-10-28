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
"""Implements the Pointer type.

You can import these APIs from the `memory` package. For example:

```mojo
from memory import Pointer
```
"""


# ===-----------------------------------------------------------------------===#
# AddressSpace
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct AddressSpace(
    EqualityComparable,
    Identifiable,
    ImplicitlyCopyable,
    Intable,
    Movable,
    Stringable,
    Writable,
):
    """Address space of the pointer.

    This type represents memory address spaces for both CPU and GPU targets.
    On CPUs, typically only GENERIC is used. On GPUs (NVIDIA/AMD), various
    address spaces provide access to different memory regions with different
    performance characteristics.
    """

    var _value: Int

    # CPU address space
    alias GENERIC = AddressSpace(0)
    """Generic address space. Used for CPU memory and default GPU memory."""

    # GPU address spaces
    # See https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space
    # And https://llvm.org/docs/AMDGPUUsage.html#address-spaces
    alias GLOBAL = AddressSpace(1)
    """Global GPU memory address space."""
    alias SHARED = AddressSpace(3)
    """Shared GPU memory address space (per thread block/workgroup)."""
    alias CONSTANT = AddressSpace(4)
    """Constant GPU memory address space (read-only)."""
    alias LOCAL = AddressSpace(5)
    """Local GPU memory address space (per thread, private)."""
    alias SHARED_CLUSTER = AddressSpace(7)
    """Shared cluster GPU memory address space (NVIDIA-specific)."""

    @always_inline("builtin")
    fn __init__(out self, value: Int):
        """Initializes the address space from the underlying integral value.

        Args:
          value: The address space value.
        """
        self._value = value

    @always_inline("builtin")
    fn value(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("builtin")
    fn __int__(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Checks if the two address spaces are equal.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self is other

    @always_inline("nodebug")
    fn __is__(self, other: Self) -> Bool:
        """Checks if the two address spaces are equal.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are not equal and False otherwise.
        """
        return self.value() == other.value()

    @always_inline("nodebug")
    fn __isnot__(self, other: Self) -> Bool:
        """Checks if the two address spaces are not equal.

        Args:
          other: The other address space value.

        Returns:
          True if the two address spaces are not equal and False otherwise.
        """
        return self.value() != other.value()

    @always_inline("nodebug")
    fn __str__(self) -> String:
        """Gets a string representation of the AddressSpace.

        Returns:
            The string representation of the AddressSpace.
        """
        return String.write(self)

    @always_inline("nodebug")
    fn write_to(self, mut writer: Some[Writer]):
        """Formats the address space to the provided Writer.

        Args:
            writer: The object to write to.
        """
        if self is AddressSpace.GENERIC:
            writer.write("AddressSpace.GENERIC")
        elif self is AddressSpace.GLOBAL:
            writer.write("AddressSpace.GLOBAL")
        elif self is AddressSpace.SHARED:
            writer.write("AddressSpace.SHARED")
        elif self is AddressSpace.CONSTANT:
            writer.write("AddressSpace.CONSTANT")
        elif self is AddressSpace.LOCAL:
            writer.write("AddressSpace.LOCAL")
        elif self is AddressSpace.SHARED_CLUSTER:
            writer.write("AddressSpace.SHARED_CLUSTER")
        else:
            writer.write("AddressSpace(", self.value(), ")")


# ===-----------------------------------------------------------------------===#
# Deprecated aliases for backward compatibility
# ===-----------------------------------------------------------------------===#

alias _GPUAddressSpace = AddressSpace
"""Deprecated: Use `AddressSpace` instead. This alias is provided for backward
compatibility and will be removed in a future release."""

alias GPUAddressSpace = AddressSpace
"""Deprecated: Use `AddressSpace` instead. This alias is provided for backward
compatibility and will be removed in a future release."""


# ===-----------------------------------------------------------------------===#
# Pointer
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct Pointer[
    mut: Bool, //,
    type: AnyType,
    origin: Origin[mut],
    address_space: AddressSpace = AddressSpace.GENERIC,
](ImplicitlyCopyable, Movable, Stringable):
    """Defines a non-nullable safe pointer.

    For a comparison with other pointer types, see [Intro to
    pointers](/mojo/manual/pointers/) in the Mojo Manual.

    Parameters:
        mut: Whether the pointee data may be mutated through this.
        type: Type of the underlying data.
        origin: The origin of the pointer.
        address_space: The address space of the pointee data.
    """

    # Aliases
    alias _mlir_type = __mlir_type[
        `!lit.ref<`,
        type,
        `, `,
        origin._mlir_origin,
        `, `,
        address_space._value._mlir_value,
        `>`,
    ]
    alias _with_origin = Pointer[type, _, address_space]

    alias Mutable = Self._with_origin[MutableOrigin.cast_from[origin]]
    """The mutable version of the `Pointer`."""
    alias Immutable = Self._with_origin[ImmutableOrigin.cast_from[origin]]
    """The immutable version of the `Pointer`."""
    # Fields
    var _value: Self._mlir_type
    """The underlying MLIR representation."""

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @doc_private
    @implicit
    @always_inline("nodebug")
    fn __init__(
        other: Self._with_origin[_],
        out self: Self._with_origin[ImmutableOrigin.cast_from[other.origin]],
    ):
        """Implicitly cast the mutable origin of self to an immutable one.

        Args:
            other: The `Pointer` to cast.
        """
        self = {_mlir_value = other._value}

    @doc_private
    @always_inline("nodebug")
    fn __init__(out self, *, _mlir_value: Self._mlir_type):
        """Constructs a Pointer from its MLIR prepresentation.

        Args:
             _mlir_value: The MLIR representation of the pointer.
        """
        self._value = _mlir_value

    @always_inline("nodebug")
    fn __init__(
        out self, *, ref [origin, address_space._value._mlir_value]to: type
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(_mlir_value=__get_mvalue_as_litref(to))

    @always_inline
    fn get_immutable(self) -> Self.Immutable:
        """Constructs a new Pointer with the same underlying target
        and an ImmutableOrigin.

        Returns:
            A new Pointer with the same target as self and an ImmutableOrigin.

        Notes:
            This does **not** copy the underlying data.
        """
        return rebind[Self.Immutable](self)

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self) -> ref [origin, address_space] type:
        """Enable subscript syntax `ptr[]` to access the element.

        Returns:
            A reference to the underlying value in memory.
        """
        return __get_litref_as_mvalue(self._value)

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Pointer[type, _, address_space]) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return UnsafePointer(to=self[]) == UnsafePointer(to=rhs[])

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ne__(self, rhs: Pointer[type, _, address_space]) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @no_inline
    fn __str__(self) -> String:
        """Gets a string representation of the Pointer.

        Returns:
            The string representation of the Pointer.
        """
        return String(UnsafePointer(to=self[]))

    @always_inline("nodebug")
    fn __merge_with__[
        other_type: type_of(Pointer[type, _, address_space]),
    ](
        self,
        out result: Pointer[
            mut = mut & other_type.origin.mut,
            type=type,
            origin = origin_of(origin, other_type.origin),
            address_space=address_space,
        ],
    ):
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return {_mlir_value = self._value}  # allow lit.ref to convert.
