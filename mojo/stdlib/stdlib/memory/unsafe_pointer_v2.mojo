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
from sys import align_of, is_gpu, is_nvidia_gpu, size_of
from sys.intrinsics import gather, scatter, strided_load, strided_store

from builtin.simd import _simd_construction_checks
from memory import memcpy
from memory.memory import _free, _malloc
from memory.maybe_uninitialized import UnsafeMaybeUninitialized
from memory.unsafe_pointer import UnsafePointer, _default_invariant
from os import abort
from python import PythonObject

# ===----------------------------------------------------------------------=== #
# alloc
# ===----------------------------------------------------------------------=== #


@always_inline
fn alloc[
    type: AnyType, /
](count: Int, *, alignment: Int = align_of[type]()) -> UnsafePointerV2[
    type,
    MutableOrigin.external,
    address_space = AddressSpace.GENERIC,
]:
    """Allocates contiguous storage for `count` elements of `type` with
    alignment `alignment`.

    Parameters:
        type: The type of the elements to allocate storage for.

    Args:
        count: Number of elements to allocate.
        alignment: The alignment of the allocation.

    Returns:
        Pointer to the newly allocated uninitialized array.

    Constraints:
        `count` must be positive and `size_of[type]()` must be > 0.

    Safety:

    - The returned memory is uninitialized; reading before writing is undefined.
    - The returned pointer has an empty mutable origin; you must call `free()`
      to release it.

    Example:

    ```mojo
    var p = alloc[Int32](4)
    p.store(0, Int32(42))
    p.store(1, Int32(7))
    p.store(2, Int32(9))
    var a = p.load(0)
    print(a[0], p.load(1)[0], p.load(2)[0])
    p.free()
    ```
    """
    return UnsafePointer[type].alloc(count, alignment=alignment)


# ===----------------------------------------------------------------------=== #
# UnsafePointer aliases
# ===----------------------------------------------------------------------=== #


alias UnsafeMutPointer[
    type: AnyType,
    origin: MutableOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = UnsafePointerV2[mut=True, type, origin, address_space=address_space]
"""A mutable unsafe pointer."""

alias UnsafeImmutPointer[
    type: AnyType,
    origin: ImmutableOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = UnsafePointerV2[type, origin, address_space=address_space]
"""An immutable unsafe pointer."""

alias OpaquePointerV2[
    mut: Bool, //,
    origin: Origin[mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = UnsafePointerV2[NoneType, origin, address_space=address_space]
"""An opaque pointer, equivalent to the C `(const) void*` type."""

alias OpaqueMutPointer[
    origin: MutableOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = OpaquePointerV2[origin, address_space=address_space]
"""A mutable opaque pointer, equivalent to the C `void*` type."""

alias OpaqueImmutPointer[
    origin: ImmutableOrigin,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = OpaquePointerV2[origin, address_space=address_space]
"""An immutable opaque pointer, equivalent to the C `const void*` type."""


alias ExternalMutPointer[
    type: AnyType,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = ExternalPointer[
    type,
    mut=True,
    address_space=address_space,
]
"""A mutable `ExternalPointer`."""

alias ExternalImmutPointer[
    type: AnyType,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = ExternalPointer[
    type,
    mut=True,
    address_space=address_space,
]
"""An immutable `ExternalPointer`."""

alias ExternalPointer[
    type: AnyType,
    *,
    mut: Bool = False,
    address_space: AddressSpace = AddressSpace.GENERIC,
] = UnsafePointerV2[
    mut=mut,
    type,
    Origin[mut].external,
    address_space=address_space,
]
"""`ExternalPointer[T]` is an alias of `UnsafePointerV2` with an `external`
origin, informing the compiler this pointer does not alias any existing values.

Parameters:
    type: The type the pointer points to.
    mut: Whether the pointer is mutable - defaults to `False`.
    address_space: The address space associated with the pointer.

Unlike `UnsafePointer`, `ExternalPointer` does not have a parametric
`origin` making it unable to alias any existing values. This can lead to
unexpected behavior if not used correctly. Before you use `ExternalPointer`,
consider using a more memory safe type instead: `Pointer[T]`, `Span[T]`,
`ref T`, or `OwnedPointer[T]`. If you _do_ need the unsafe functionality of
pointer-like behavior (pointer arithmetic, unchecked dereferencing etc.) and are
unable to use a memory-safe abstraction, consider using `UnsafePointer` with a
parameterized `origin`.

Safety:
- **Origin management**: An `ExternalPointer` does not alias any values.
  See below for more information on how to correctly use origins with
  `ExternalPointer`.
- All safety requirements of `UnsafePointer` apply to `ExternalPointer`
  as well, including: manual memory management, null checking, uninitialized
  memory access, and bounds checking.

Notes:

Correct origin usage:

`ExternalPointer[T]` contains an `external` origin, meaning it points to
memory that does not alias any other values. Using an `ExternalPointer`
**will not** extend the lifetime of any other values in scope. This includes
returning an `ExternalPointer` from a struct's member function, as it
will not extend the struct's lifetime _even though_ the struct holds a copy of
the same pointer. It is seldom correct to return an `ExternalPointer`
from a member function.

Consider the following example:

```mojo
struct MyContainer[T: AnyType & Movable]:
    var data: ExternalMutPointer[T]

    fn __init__(out self, var value: T)
        self.data = alloc[T](1)
        self.data.init_pointee_move(value^)

    fn __del__(deinit self):
        self.data.destroy_pointee()
        self.data.free()

    fn get_underlying_pointer(self) -> ExternalMutPointer[T]:
        return self.data

def main():
    var container = MyContainer[Int](42)
    var ptr = container.get_underlying_pointer()
    # container^.__del__() will be called here, deallocating the pointer.

    # `ptr` now points to deallocated memory.
    print(ptr[]) # <-- Undefined behavior!
```

The above example shows how easily undefined behavior can arise when using
`ExternalPointer` incorrectly. Due to Mojo's ASAP destruction rules, the
compiler will destroy `container` before `ptr` is read from - meaning
`ptr` now points to deallocated memory. This is because `ptr` does not
have an origin tied back to `container`.

To prevent bugs like this, `get_underlying_pointer` should be implemented in
one of the following ways.

Option 1:

Have this function "consume" `self` via `deinit`. This will - at compile time -
prevent the usage of `container` after this function is called and it will
prevent `container` from deallocating the memory as its `__del__` method will
not be called by the compiler.

```mojo
struct MyContainer[T: AnyType & Movable]:
    var data: ExternalMutPointer[T]

    # Notice the use of `deinit` to consume `self`.
    fn get_underlying_pointer(deinit self) -> ExternalMutPointer[T]:
        return self.data

def main():
    var container = MyContainer[Int](42)
    # Notice the `^` transfer sigil.
    var ptr = container^.get_underlying_pointer()

    # This will print `42` since `container.__del__()` is not called, leaving
    # the allocated memory intact.
    print(ptr[])

    # Don't forget the cleanup the memory!
    ptr.destroy_pointee()
    ptr.free()
```

Option 2:

Have the function return a pointer type with an origin tied back to `self`. This
will inform the compiler that the returned pointer is aliasing `container`,
preventing the compiler from destroying `container` until `ptr` and
`container` are both unused.

```mojo
struct MyContainer[T: AnyType & Movable]:
    var data: ExternalMutPointer[T]

    # Notice the use of `origin=origin_of(self)`.
    fn get_underlying_pointer(
        self,
    ) -> UnsafeImmutPointer[T, origin=origin_of(self)]:
        return self.data
            # cast the pointer to immutable
            .as_immutable()
            # cast the origin to the origin of `self`
            .unsafe_origin_cast[origin_of(self)]()

def main():
    var container = MyContainer[Int](42)
    var ptr = container.get_underlying_pointer()
    # The compiler does *not* destroy `container` here, as `ptr` contains
    # an origin tied back to `container`.

    print(ptr[]) # <-- This will now print `42`!

    # `container^.__del__()` will be called here, freeing the memory.
```

Option 3:

Ensure the function resets its data field to null. This will ensure that when
`self` is destroyed, it does _not_ deallocate the memory exposed via
`steal_underlying_pointer`. This option should seldom be used as it can still lead
to bugs if `self` does not account for a null pointer.

```mojo
struct MyContainer[T: AnyType & Movable]:
    var data: ExternalMutPointer[T]

    fn steal_underlying_pointer(
        mut self,
    ) -> ExternalMutPointer[T]:
        var result = self.data
        self.data = {} # <-- Reset the data field to null
        return result

    fn __del__(deinit self):
        # Since `data` can now be null, we need to add explicit null checks.
        if self.data:
            self.data.destroy_pointee()
            self.data.free()

def main():
    var container = MyContainer[Int](42)
    var ptr = container.steal_underlying_pointer()

    # `container.__del__()` *is* called here, but is now null, leaving
    #`ptr` pointing to valid memory.

    print(ptr[]) # <-- This will now print `42`!

    # Don't forget the cleanup the memory!
    ptr.destroy_pointee()
    ptr.free()
```

When to Use `ExternalPointer`:

1. Memory Allocation

`ExternalPointer` is the return type of `alloc()` because
newly allocated memory has no origin aliasing pre-existing values. Data
structures containing heap allocated memory often hold an `ExternalPointer`.

2. Foreign Function Interface (FFI)

When interfacing with C functions via `external_call`, `ExternalPointer` is
often the correct type to use as a return type or as an out parameter. This is
because C functions can return pointers to memory outside the known origins in
the Mojo program.

```mojo
from sys.ffi import external_call

# `getenv` returns a pointer to memory outside of the Mojo program.
#
# Notably a return type of `ExternalPointer[Byte]` is immutable by default as
# we are not allowed to modify the memory it points to.
#
# If the FFI function returned a mutable pointer, we could specify `mut=True`.
var env_ptr = external_call["getenv", ExternalPointer[Byte]](
    # parameters...
)
```

3. Memory Management Handoff

Use `ExternalPointer` when a function needs to "leak" or "give up" its pointer
with the expectation that the caller is now responsible for proper management
and cleanup. This however, should generally be avoided as it can
lead to undefined behavior if not implemented correctly. See the section on
"Correct origin usage" above for more information.

When NOT to Use `ExternalPointer`:

1. Function Arguments

It is almost never correct to take an `ExternalPointer` as an argument to a
function. Prefer using a more memory safe type instead: e.g. `Pointer[T]` or
`Span[T]` when the function does not need to take ownership of the pointer. Use
`OwnedPointer[T]` when the function does need to take ownership of the pointer.

2. Return types of public member functions.

Data structures should almost never expose their internal `ExternalPointer` via
public functions unless that function is specifically designed to transfer
ownership and management responsibility to the caller. Instead, public methods
should return references or memory safe reference types: e.g. `Pointer[T]` or
`Span[T]`.
"""


@register_passable("trivial")
struct UnsafePointerV2[
    mut: Bool, //,
    type: AnyType,
    origin: Origin[mut],
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    Comparable,
    Defaultable,
    ImplicitlyBoolable,
    ImplicitlyCopyable,
    Intable,
    Movable,
    Stringable,
    Writable,
):
    """`UnsafePointerV2[T]` represents an indirect reference to one or more values
    of type `T` consecutively in memory, and can refer to uninitialized memory.

    Because it supports referring to uninitialized memory, it provides unsafe
    methods for initializing and destroying instances of `T`, as well as methods
    for accessing the values once they are initialized. You should instead use
    safer pointers when possible.

    Differences from `UnsafePointer` (V1):

    - `UnsafePointerV2` fixes the unsafe implicit mutability and origin casting
      issues of `UnsafePointer`.
    - `UnsafePointerV2` has an inferred mutability parameter.
    - `UnsafePointerV2` does _not_ have a defaulted origin parameter, this must
      be explicitly specified or unbound.

    Important things to know:

    - This pointer is unsafe and nullable. No bounds checks; reading before
      writing is undefined.
    - It does not own existing memory. When memory is heap-allocated with
      `alloc()`, you must call `free()`.
    - For simple read/write access, use `(ptr + i)[]` or `ptr[i]` where `i`
      is the offset size.

    For more information see [Unsafe
    pointers](/mojo/manual/pointers/unsafe-pointers) in the Mojo Manual. For a
    comparison with other pointer types, see [Intro to
    pointers](/mojo/manual/pointers/).

    Parameters:
        mut: Whether the origin is mutable.
        type: The type the pointer points to.
        origin: The origin of the memory being addressed.
        address_space: The address space associated with the UnsafePointer allocated memory.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        type,
        `, `,
        address_space._value._mlir_value,
        `>`,
    ]
    """The underlying pointer type."""

    alias _with_origin[
        with_mut: Bool, //, with_origin: Origin[with_mut]
    ] = UnsafePointerV2[
        mut=with_mut,
        type,
        with_origin,
        address_space=address_space,
    ]

    alias _AsV1 = UnsafePointer[
        type, mut=mut, origin=origin, address_space=address_space
    ]

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var address: Self._mlir_type
    """The underlying pointer."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline("builtin")
    fn __init__(out self):
        """Create a null pointer."""
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]

    @doc_private
    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Self._mlir_type):
        """Create a pointer from a low-level pointer primitive.

        Args:
            value: The MLIR value of the pointer to construct with.
        """
        self.address = value

    @always_inline("nodebug")
    fn __init__(
        out self, *, ref [origin, address_space._value._mlir_value]to: type
    ):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(to)))

    @always_inline("builtin")
    @implicit
    fn __init__(
        other: UnsafePointerV2[mut=True, **_],
        out self: UnsafePointerV2[
            other.type,
            ImmutableOrigin.cast_from[other.origin],
            address_space = other.address_space,
        ],
    ):
        """Implicitly casts a mutable pointer to immutable.

        Args:
            other: The mutable pointer to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type = type_of(self)._mlir_type
        ](other.address)

    @always_inline("builtin")
    @implicit
    fn __init__(
        other: UnsafePointerV2[mut=True, type, **_],
        out self: UnsafePointerV2[
            other.type,
            MutableAnyOrigin,
            address_space = other.address_space,
        ],
    ):
        """Implicitly casts a mutable pointer to `MutableAnyOrigin`.

        Args:
            other: The mutable pointer to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type = type_of(self)._mlir_type
        ](other.address)

    @always_inline("builtin")
    @implicit
    fn __init__(
        other: UnsafePointerV2[**_],
        out self: UnsafePointerV2[
            other.type,
            ImmutableAnyOrigin,
            address_space = other.address_space,
        ],
    ):
        """Implicitly casts a pointer to `ImmutableAnyOrigin`.

        Args:
            other: The pointer to cast from.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[
            _type = type_of(self)._mlir_type
        ](other.address)

    @doc_private
    @implicit
    fn __init__(
        other: UnsafePointerV2[mut=False, type, **_],
        out self: UnsafePointerV2[
            other.type, MutableOrigin.cast_from[MutableAnyOrigin], **_
        ],
    ):
        constrained[
            False, "Invalid UnsafePointer conversion from immutable to mutable"
        ]()
        self = abort[type_of(self)]()

    fn __init__(
        out self: UnsafePointerV2[type, origin],
        *,
        ref [origin]unchecked_downcast_value: PythonObject,
    ):
        """Downcast a `PythonObject` known to contain a Mojo object to a pointer.

        This operation is only valid if the provided Python object contains
        an initialized Mojo object of matching type.

        Args:
            unchecked_downcast_value: The Python object to downcast from.
        """
        self = unchecked_downcast_value.unchecked_downcast_value_ptr[type]()

    # ===-------------------------------------------------------------------===#
    # V1 <-> V2 conversion
    # ===-------------------------------------------------------------------===#

    @doc_private
    @always_inline("builtin")
    @implicit
    fn __init__(
        out self, other: UnsafePointer[type, address_space=address_space, **_]
    ):
        """Cast a V1 pointer to a V2 pointer."""
        self.address = __mlir_op.`pop.pointer.bitcast`[_type = Self._mlir_type](
            other.address
        )

    @doc_private
    @always_inline("builtin")
    fn _as_v1(
        self,
        out result: UnsafePointer[
            type, mut=mut, origin=origin, address_space=address_space
        ],
    ):
        result = UnsafePointer(self.address)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self) -> ref [origin, address_space] type:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.

        Safety:
            The pointer must not be null and must point to initialized memory.
        """
        return self._as_v1()[]

    @always_inline("nodebug")
    fn offset[I: Indexer, //](self, idx: I) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The offset of the new pointer.

        Returns:
            The new constructed UnsafePointer.
        """
        return self._as_v1().offset(idx)

    @always_inline("nodebug")
    fn __getitem__[
        I: Indexer, //
    ](self, offset: I) -> ref [origin, address_space] type:
        """Return a reference to the underlying data, offset by the given index.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset reference.
        """
        return self._as_v1()[offset]

    @always_inline("nodebug")
    fn __add__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return self.offset(offset)

    @always_inline
    fn __sub__[I: Indexer, //](self, offset: I) -> Self:
        """Return a pointer at an offset from the current one.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.

        Returns:
            An offset pointer.
        """
        return self + (-1 * index(offset))

    @always_inline
    fn __iadd__[I: Indexer, //](mut self, offset: I):
        """Add an offset to this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self + offset

    @always_inline
    fn __isub__[I: Indexer, //](mut self, offset: I):
        """Subtract an offset from this pointer.

        Parameters:
            I: A type that can be used as an index.

        Args:
            offset: The offset index.
        """
        self = self - offset

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(
        self, rhs: UnsafePointerV2[type, address_space=address_space, **_]
    ) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return Int(self) == Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ne__(
        self, rhs: UnsafePointerV2[type, address_space=address_space, **_]
    ) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return not (self == rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __lt__(
        self, rhs: UnsafePointerV2[type, address_space=address_space, **_]
    ) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) < Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __le__(
        self, rhs: UnsafePointerV2[type, address_space=address_space, **_]
    ) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) <= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return Int(self) <= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __gt__(
        self, rhs: UnsafePointerV2[type, address_space=address_space, **_]
    ) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) > Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __gt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) > Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ge__(
        self, rhs: UnsafePointerV2[type, address_space=address_space, **_]
    ) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) >= Int(rhs)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline("nodebug")
    fn __ge__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a higher than or equal
           address than rhs.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if this pointer represents a higher than or equal address and
            False otherwise.
        """
        return Int(self) >= Int(rhs)

    @always_inline("builtin")
    fn __merge_with__[
        other_type: type_of(
            UnsafePointerV2[
                type,
                origin=_,
                address_space=address_space,
            ]
        ),
    ](self) -> UnsafePointerV2[
        type=type,
        origin = origin_of(origin, other_type.origin),
        address_space=address_space,
    ]:
        """Returns a pointer merged with the specified `other_type`.

        Parameters:
            other_type: The type of the pointer to merge with.

        Returns:
            A pointer merged with the specified `other_type`.
        """
        return self.address  # allow kgen.pointer to convert.

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return self._as_v1().__bool__()

    @always_inline
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return self._as_v1().__int__()

    @no_inline
    fn __str__(self) -> String:
        """Gets a string representation of the pointer.

        Returns:
            The string representation of the pointer.
        """
        return self._as_v1().__str__()

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Formats this pointer address to the provided Writer.

        Args:
            writer: The object to write to.
        """
        return self._as_v1().write_to(writer)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn swap_pointees[
        U: Movable, //
    ](
        self: UnsafeMutPointer[U, address_space = AddressSpace.GENERIC],
        other: UnsafeMutPointer[U, address_space = AddressSpace.GENERIC],
    ):
        """Swap the values at the pointers.

        This function assumes that `self` and `other` _may_ overlap in memory.
        If that is not the case, or when references are available, you should
        use `builtin.swap` instead.

        Parameters:
            U: The type the pointers point to, which must be `Movable`.

        Args:
            other: The other pointer to swap with.

        Safety:
            - `self` and `other` must both point to valid, initialized instances
              of `T`.
        """
        self._as_v1().swap_pointees(other._as_v1())

    @always_inline("nodebug")
    fn as_noalias_ptr(self) -> Self:
        """Cast the pointer to a new pointer that is known not to locally alias
        any other pointer. In other words, the pointer transitively does not
        alias any other memory value declared in the local function context.

        This information is relayed to the optimizer. If the pointer does
        locally alias another memory value, the behaviour is undefined.

        Returns:
            A noalias pointer.
        """
        return self._as_v1().as_noalias_ptr()

    @always_inline("nodebug")
    fn load[
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[mut](),
    ](self: UnsafePointerV2[Scalar[dtype], **_]) -> SIMD[dtype, width]:
        """Loads `width` elements from the value the pointer points to.

        Use `alignment` to specify minimal known alignment in bytes; pass a
        smaller value (such as 1) if loading from packed/unaligned memory. The
        `volatile`/`invariant` flags control reordering and common-subexpression
        elimination semantics for special cases.

        Example:

        ```mojo
        var p = UnsafePointer[Int32].alloc(8)
        p.store(0, SIMD[DType.int32, 4](1, 2, 3, 4))
        var v = p.load[width=4]()
        print(v)  # => [1, 2, 3, 4]
        p.free()
        ```

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of the SIMD vector.
            width: The number of elements to load.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.
            invariant: Whether the load is from invariant memory.

        Returns:
            The loaded SIMD vector.
        """
        return self._as_v1().load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
        ]()

    @always_inline("nodebug")
    fn load[
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[mut](),
    ](self: UnsafePointerV2[Scalar[dtype], **_], offset: Scalar) -> SIMD[
        dtype, width
    ]:
        """Loads the value the pointer points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.
            The offset must be integer.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self._as_v1().load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
        ](offset)

    @always_inline("nodebug")
    fn load[
        I: Indexer,
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[mut](),
    ](self: UnsafePointerV2[Scalar[dtype], **_], offset: I) -> SIMD[
        dtype, width
    ]:
        """Loads the value the pointer points to with the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self._as_v1().load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
        ](offset)

    @always_inline("nodebug")
    fn store[
        I: Indexer,
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: UnsafePointerV2[mut=True, Scalar[dtype], **_],
        offset: I,
        val: SIMD[dtype, width],
    ):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.
            The offset must be integer.

        Parameters:
            I: A type that can be used as an index.
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        return self._as_v1().store[
            width=width, alignment=alignment, volatile=volatile
        ](offset, val)

    @always_inline("nodebug")
    fn store[
        dtype: DType,
        offset_type: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: UnsafePointerV2[mut=True, Scalar[dtype], **_],
        offset: Scalar[offset_type],
        val: SIMD[dtype, width],
    ):
        """Stores a single element value at the given offset.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector elements.
            offset_type: The data type of the offset value.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        return self._as_v1().store[
            width=width, alignment=alignment, volatile=volatile
        ](offset, val)

    @always_inline("nodebug")
    fn store[
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: UnsafePointerV2[mut=True, Scalar[dtype], **_],
        val: SIMD[dtype, width],
    ):
        """Stores a single element value `val` at element offset 0.

        Specify `alignment` when writing to packed/unaligned memory. Requires a
        mutable pointer. For writing at an element offset, use the overloads
        that accept an index or scalar offset.

        Example:

        ```mojo
        var p = UnsafePointer[Float32].alloc(4)
        var vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
        p.store(vec)
        var out = p.load[width=4]()
        print(out)  # => [1.0, 2.0, 3.0, 4.0]
        p.free()
        ```

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The number of elements to store.
            alignment: The minimal alignment (bytes) of the address.
            volatile: Whether the operation is volatile.

        Args:
            val: The SIMD value to store.
        """
        return self._as_v1().store[
            width=width, alignment=alignment, volatile=volatile
        ](val)

    @always_inline("nodebug")
    fn _store[
        dtype: DType,
        width: Int,
        *,
        alignment: Int = align_of[dtype](),
        volatile: Bool = False,
    ](
        self: UnsafePointerV2[mut=True, Scalar[dtype], **_],
        val: SIMD[dtype, width],
    ):
        return self._as_v1()._store[
            width=width, alignment=alignment, volatile=volatile
        ](val)

    @always_inline("nodebug")
    fn strided_load[
        dtype: DType, T: Intable, //, width: Int
    ](self: UnsafePointerV2[Scalar[dtype], **_], stride: T) -> SIMD[
        dtype, width
    ]:
        """Performs a strided load of the SIMD vector.

        Parameters:
            dtype: DType of returned SIMD value.
            T: The Intable type of the stride.
            width: The SIMD width.

        Args:
            stride: The stride between loads.

        Returns:
            A vector which is stride loaded.
        """
        return self._as_v1().strided_load[width=width](stride)

    @always_inline("nodebug")
    fn strided_store[
        dtype: DType,
        T: Intable, //,
        width: Int = 1,
    ](
        self: UnsafePointerV2[mut=True, Scalar[dtype], **_],
        val: SIMD[dtype, width],
        stride: T,
    ):
        """Performs a strided store of the SIMD vector.

        Parameters:
            dtype: DType of `val`, the SIMD value to store.
            T: The Intable type of the stride.
            width: The SIMD width.

        Args:
            val: The SIMD value to store.
            stride: The stride between stores.
        """
        return self._as_v1().strided_store[width=width](val, stride)

    @always_inline("nodebug")
    fn gather[
        dtype: DType, //,
        *,
        width: Int = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: UnsafePointerV2[Scalar[dtype], **_],
        offset: SIMD[_, width],
        mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](fill=True),
        default: SIMD[dtype, width] = 0,
    ) -> SIMD[dtype, width]:
        """Gathers a SIMD vector from offsets of the current pointer.

        This method loads from memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector,
        or takes from the `default` SIMD vector, depending on the values of
        the `mask` SIMD vector.

        If a mask element is `True`, the respective result element is given
        by the current pointer and the `offset` SIMD vector; otherwise, the
        result element is taken from the `default` SIMD vector.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

        Parameters:
            dtype: DType of the return SIMD.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to gather from.
            mask: The SIMD vector of boolean values, indicating for each
                element whether to load from memory or to take from the
                `default` SIMD vector.
            default: The SIMD vector providing default values to be taken
                where the `mask` SIMD vector is `False`.

        Returns:
            The SIMD vector containing the gathered values.
        """
        return self._as_v1().gather[width=width, alignment=alignment](
            offset, mask, default
        )

    @always_inline("nodebug")
    fn scatter[
        dtype: DType, //,
        *,
        width: Int = 1,
        alignment: Int = align_of[dtype](),
    ](
        self: UnsafePointerV2[mut=True, Scalar[dtype], **_],
        offset: SIMD[_, width],
        val: SIMD[dtype, width],
        mask: SIMD[DType.bool, width] = SIMD[DType.bool, width](fill=True),
    ):
        """Scatters a SIMD vector into offsets of the current pointer.

        This method stores at memory addresses calculated by appropriately
        shifting the current pointer according to the `offset` SIMD vector,
        depending on the values of the `mask` SIMD vector.

        If a mask element is `True`, the respective element in the `val` SIMD
        vector is stored at the memory address defined by the current pointer
        and the `offset` SIMD vector; otherwise, no action is taken for that
        element in `val`.

        If the same offset is targeted multiple times, the values are stored
        in the order they appear in the `val` SIMD vector, from the first to
        the last element.

        Constraints:
            The offset type must be an integral type.
            The alignment must be a power of two integer value.

        Parameters:
            dtype: DType of `value`, the result SIMD buffer.
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The SIMD vector of offsets to scatter into.
            val: The SIMD vector containing the values to be scattered.
            mask: The SIMD vector of boolean values, indicating for each
                element whether to store at memory or not.
        """
        return self._as_v1().scatter[width=width, alignment=alignment](
            offset, val, mask
        )

    @always_inline
    fn free(
        self: UnsafePointerV2[
            mut=True,
            type,
            origin=_,
            address_space = AddressSpace.GENERIC,
        ]
    ):
        """Free the memory referenced by the pointer."""
        self._as_v1().free()

    @always_inline("builtin")
    fn bitcast[
        T: AnyType
    ](self) -> UnsafePointerV2[T, address_space=address_space, origin=origin,]:
        """Bitcasts an UnsafePointer to a different type.

        Parameters:
            T: The target type.

        Returns:
            A new UnsafePointer object with the specified type and the same
            address, mutability, and origin as the original UnsafePointer.
        """
        return self._as_v1().bitcast[T]()

    alias _OriginCastType[
        target_mut: Bool, target_origin: Origin[target_mut]
    ] = UnsafePointerV2[
        type,
        address_space=address_space,
        origin=target_origin,
    ]

    @always_inline("nodebug")
    fn mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[
        target_mut, Origin[target_mut].cast_from[origin]
    ]:
        """Changes the mutability of a pointer.

        This is a safe way to change the mutability of a pointer with an
        unbounded mutability. This function will emit a compile time error if
        you try to cast an immutable pointer to mutable.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.
        """
        return self._as_v1().mut_cast[target_mut]()

    @always_inline("builtin")
    fn unsafe_mut_cast[
        target_mut: Bool
    ](self) -> Self._OriginCastType[
        target_mut, Origin[target_mut].cast_from[origin]
    ]:
        """Changes the mutability of a pointer.

        Parameters:
            target_mut: Mutability of the destination pointer.

        Returns:
            A pointer with the same type, origin and address space as the
            original pointer, but with the newly specified mutability.

        If you are unconditionally casting the mutability to `False`, use
        `as_immutable` instead.
        If you are casting to mutable or a parameterized mutability, prefer
        using the safe `mut_cast` method instead.

        Safety:
            Casting the mutability of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior. Consider restricting
            types to their proper mutability at the function signature level.
            For example, taking an `UnsafeMutPointer[T, **_]` as an
            argument over an unbound `UnsafePointer[T, **_]` is preferred.
        """
        return self._as_v1().unsafe_mut_cast[target_mut]()

    @always_inline("builtin")
    fn unsafe_origin_cast[
        target_origin: Origin[mut]
    ](self) -> Self._OriginCastType[mut, target_origin]:
        """Changes the origin of a pointer.

        Parameters:
            target_origin: Origin of the destination pointer.

        Returns:
            A pointer with the same type, mutability and address space as the
            original pointer, but with the newly specified origin.

        If you are unconditionally casting the origin to an `AnyOrigin`, use
        `as_any_origin` instead.

        Safety:
            Casting the origin of a pointer is inherently very unsafe.
            Improper usage can lead to undefined behavior or unexpected variable
            destruction. Considering parameterizing the origin at the function
            level to avoid unnecessary casts.
        """
        return self._as_v1().unsafe_origin_cast[target_origin]()

    @always_inline("builtin")
    fn as_immutable(
        self,
    ) -> Self._OriginCastType[False, ImmutableOrigin.cast_from[origin]]:
        """Changes the mutability of a pointer to immutable.

        Unlike `unsafe_mut_cast`, this function is always safe to use as casting
        from (im)mutable to immutable is always safe.

        Returns:
            A pointer with the mutability set to immutable.
        """
        return self._as_v1().as_immutable()

    @doc_private
    fn as_any_origin(
        self: UnsafePointerV2[type, **_],
        out result: type_of(self)._OriginCastType[False, ImmutableAnyOrigin],
    ):
        constrained[
            False,
            (
                "An UnsafePointer with unbound mutability cannot be cast to"
                " 'AnyOrigin'. Consider using `as_immutable` first, or binding"
                " the mutability explicitly before calling this function."
            ),
        ]()
        result = abort[type_of(result)]()

    @always_inline("builtin")
    fn as_any_origin(
        self: UnsafePointerV2[mut=False, type, **_],
    ) -> UnsafePointerV2[
        type,
        ImmutableAnyOrigin,
        address_space=address_space,
    ]:
        """Casts the origin of an immutable pointer to `ImmutableAnyOrigin`.

        Returns:
            A pointer with the origin set to `ImmutableAnyOrigin`.

        It is usually preferred to maintain concrete origin values instead of
        using `ImmutableAnyOrigin`. However, if it is needed, keep in mind that
        `ImmutableAnyOrigin` can alias any memory value, so Mojo's ASAP
        destruction will not apply during the lifetime of the pointer.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = UnsafePointerV2[
                type,
                ImmutableAnyOrigin,
                address_space=address_space,
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn as_any_origin(
        self: UnsafePointerV2[mut=True, type, **_],
    ) -> UnsafePointerV2[type, MutableAnyOrigin, address_space=address_space,]:
        """Casts the origin of a mutable pointer to `MutableAnyOrigin`.

        Returns:
            A pointer with the origin set to `MutableAnyOrigin`.

        This requires the pointer to already be mutable as casting mutability
        is inherently very unsafe.

        It is usually preferred to maintain concrete origin values instead of
        using `MutableAnyOrigin`. However, if it is needed, keep in mind that
        `MutableAnyOrigin` can alias any memory value, so Mojo's ASAP
        destruction will not apply during the lifetime of the pointer.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = UnsafePointerV2[
                type,
                MutableAnyOrigin,
                address_space=address_space,
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn address_space_cast[
        target_address_space: AddressSpace = Self.address_space,
    ](self) -> UnsafePointerV2[
        type,
        origin,
        address_space=target_address_space,
    ]:
        """Casts an UnsafePointer to a different address space.

        Parameters:
            target_address_space: The address space of the result.

        Returns:
            A new UnsafePointer object with the same type and the same address,
            as the original UnsafePointer and the new address space.
        """
        return self._as_v1().address_space_cast[target_address_space]()

    @always_inline
    fn destroy_pointee(
        self: UnsafePointerV2[
            mut=True, type, address_space = AddressSpace.GENERIC, **_
        ]
    ):
        """Destroy the pointed-to value.

        The pointer must not be null, and the pointer memory location is assumed
        to contain a valid initialized instance of `type`.  This is equivalent to
        `_ = self.take_pointee()` but doesn't require `Movable` and is
        more efficient because it doesn't invoke `__moveinit__`.

        """
        _ = __get_address_as_owned_value(self.address)

    @always_inline
    fn take_pointee[
        T: Movable, //,
    ](
        self: UnsafePointerV2[
            mut=True, T, address_space = AddressSpace.GENERIC, **_
        ]
    ) -> T:
        """Move the value at the pointer out, leaving it uninitialized.

        The pointer must not be null, and the pointer memory location is assumed
        to contain a valid initialized instance of `T`.

        This performs a _consuming_ move, ending the origin of the value stored
        in this pointer memory location. Subsequent reads of this pointer are
        not valid. If a new valid value is stored using `init_pointee_move()`, then
        reading from this pointer becomes valid again.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Returns:
            The value at the pointer.
        """
        return __get_address_as_owned_value(self.address)

    @always_inline
    fn init_pointee_move[
        T: Movable, //,
    ](
        self: UnsafePointerV2[
            mut=True, T, address_space = AddressSpace.GENERIC, **_
        ],
        var value: T,
    ):
        """Emplace a new value into the pointer location, moving from `value`.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        When compared to `init_pointee_copy`, this avoids an extra copy on
        the caller side when the value is an `owned` rvalue.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            value: The value to emplace.
        """
        __get_address_as_uninit_lvalue(self.address) = value^

    @always_inline
    fn init_pointee_copy[
        T: Copyable, //,
    ](
        self: UnsafePointerV2[
            mut=True, T, address_space = AddressSpace.GENERIC, **_
        ],
        value: T,
    ):
        """Emplace a copy of `value` into the pointer location.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        When compared to `init_pointee_move`, this avoids an extra move on
        the callee side when the value must be copied.

        Parameters:
            T: The type the pointer points to, which must be `Copyable`.

        Args:
            value: The value to emplace.
        """
        __get_address_as_uninit_lvalue(self.address) = value.copy()

    @always_inline
    fn init_pointee_move_from[
        T: Movable, //,
    ](
        self: UnsafePointerV2[
            mut=True, T, address_space = AddressSpace.GENERIC, **_
        ],
        src: UnsafePointerV2[
            mut=True, T, address_space = AddressSpace.GENERIC, **_
        ],
    ):
        """Moves the value `src` points to into the memory location pointed to
        by `self`.

        The `self` pointer memory location is assumed to contain uninitialized
        data prior to this assignment, and consequently the current contents of
        this pointer are not destructed before writing the value from the `src`
        pointer.

        Ownership of the value is logically transferred from `src` into `self`'s
        pointer location.

        After this call, the `src` pointee value should be treated as
        uninitialized data. Subsequent reads of or destructor calls on the `src`
        pointee value are invalid, unless and until a new valid value has been
        moved into the `src` pointer's memory location using an
        `init_pointee_*()` operation.

        This transfers the value out of `src` and into `self` using at most one
        `__moveinit__()` call.

        ### Example

        ```mojo
        var a_ptr = UnsafePointer.alloc[String](1)
        var b_ptr = UnsafePointer.alloc[String](2)

        # Initialize A pointee
        a_ptr.init_pointee_move("foo")

        # Perform the move
        b_ptr.init_pointee_move_from(a_ptr)

        # Clean up
        b_ptr.destroy_pointee()
        a_ptr.free()
        b_ptr.free()
        ```

        ### Safety

        * `self` and `src` must be non-null
        * `src` must contain a valid, initialized instance of `T`
        * The pointee contents of `self` should be uninitialized. If `self` was
          previously written with a valid value, that value will be be
          overwritten and its destructor will NOT be run.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            src: Source pointer that the value will be moved from.
        """
        __get_address_as_uninit_lvalue(
            self.address
        ) = __get_address_as_owned_value(src.address)
