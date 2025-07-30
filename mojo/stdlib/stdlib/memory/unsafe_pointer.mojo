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
"""Implement a generic unsafe pointer type.

These APIs are imported automatically, just like builtins.
"""

from sys import alignof, is_gpu, is_nvidia_gpu, sizeof
from sys.intrinsics import (
    gather,
    scatter,
    strided_load,
    strided_store,
)

from builtin.simd import _simd_construction_checks
from memory.memory import _free, _malloc

from python import PythonObject

# ===----------------------------------------------------------------------=== #
# UnsafePointer
# ===----------------------------------------------------------------------=== #


@always_inline
fn _default_alignment[type: AnyType]() -> Int:
    return alignof[type]() if is_gpu() else 1


@always_inline
fn _default_alignment[dtype: DType, width: Int = 1]() -> Int:
    return _default_alignment[Scalar[dtype]]()


@always_inline
fn _default_invariant[mut: Bool]() -> Bool:
    return is_gpu() and mut == False


alias _must_be_mut_err = "UnsafePointer must be mutable for this operation"


@register_passable("trivial")
struct UnsafePointer[
    type: AnyType,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    alignment: Int = _default_alignment[type](),
    mut: Bool = True,
    origin: Origin[mut] = Origin[mut].cast_from[MutableAnyOrigin],
](
    ImplicitlyBoolable,
    Copyable,
    Movable,
    ExplicitlyCopyable,
    Stringable,
    Writable,
    Intable,
    Comparable,
    Defaultable,
):
    """UnsafePointer[T] represents an indirect reference to one or more values of
    type T consecutively in memory, and can refer to uninitialized memory.

    Because it supports referring to uninitialized memory, it provides unsafe
    methods for initializing and destroying instances of T, as well as methods
    for accessing the values once they are initialized.

    For more information see [Unsafe
    pointers](/mojo/manual/pointers/unsafe-pointers) in the Mojo Manual. For a
    comparison with other pointer types, see [Intro to
    pointers](/mojo/manual/pointers/).

    Parameters:
        type: The type the pointer points to.
        address_space: The address space associated with the UnsafePointer allocated memory.
        alignment: The minimum alignment of this pointer known statically.
        mut: Whether the origin is mutable.
        origin: The origin of the memory being addressed.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    # Fields
    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        type,
        `, `,
        address_space._value.value,
        `>`,
    ]
    """The underlying pointer type."""

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
    fn __init__(out self, *, ref [origin, address_space._value.value]to: type):
        """Constructs a Pointer from a reference to a value.

        Args:
            to: The value to construct a pointer to.
        """
        self = Self(__mlir_op.`lit.ref.to_pointer`(__get_mvalue_as_litref(to)))

    @always_inline("builtin")
    @implicit
    fn __init__(
        out self, other: UnsafePointer[type, address_space=address_space, **_]
    ):
        """Exclusivity parameter cast a pointer.

        Args:
            other: Pointer to cast.
        """
        self.address = __mlir_op.`pop.pointer.bitcast`[_type = Self._mlir_type](
            other.address
        )

    fn __init__(
        out self: UnsafePointer[type, mut=mut, origin=origin],
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

    @always_inline
    fn copy(self) -> Self:
        """Copy an existing pointer.

        Returns:
            A copy of the value.
        """
        return self

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline
    fn alloc(
        count: Int,
    ) -> UnsafePointer[
        type,
        address_space = AddressSpace.GENERIC,
        alignment=alignment,
        # This is a newly allocated pointer, so should not alias anything
        # already existing.
        origin = MutableOrigin.empty,
    ]:
        """Allocate an array with specified or default alignment.

        Args:
            count: The number of elements in the array.

        Returns:
            The pointer to the newly allocated array.
        """
        alias sizeof_t = sizeof[type]()
        constrained[sizeof_t > 0, "size must be greater than zero"]()
        return _malloc[type, alignment=alignment](sizeof_t * count)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(self) -> ref [origin, address_space] type:
        """Return a reference to the underlying data.

        Returns:
            A reference to the value.
        """

        # We're unsafe, so we can have unsafe things.
        alias _ref_type = Pointer[type, origin, address_space]
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.from_pointer`[_type = _ref_type._mlir_type](
                self.address
            )
        )

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
        return __mlir_op.`pop.offset`(self.address, index(idx))

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
        return (self + offset)[]

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

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
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
        other_type: __type_of(
            UnsafePointer[
                type,
                address_space=address_space,
                alignment=_,
                mut=_,
                origin=_,
            ]
        ),
    ](self) -> UnsafePointer[
        type=type,
        mut = mut & other_type.origin.mut,
        origin = __origin_of(origin, other_type.origin),
        address_space=address_space,
        alignment = min(alignment, other_type.alignment),
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
        return Int(self) != 0

    @always_inline
    fn __as_bool__(self) -> Bool:
        """Return true if the pointer is non-null.

        Returns:
            Whether the pointer is null.
        """
        return self.__bool__()

    @always_inline
    fn __int__(self) -> Int:
        """Returns the pointer address as an integer.

        Returns:
          The address of the pointer as an Int.
        """
        return __mlir_op.`pop.pointer_to_index`(self.address)

    @no_inline
    fn __str__(self) -> String:
        """Gets a string representation of the pointer.

        Returns:
            The string representation of the pointer.
        """
        return hex(Int(self))

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this pointer address to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        # TODO: Avoid intermediate String allocation.
        writer.write(String(self))

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

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
        return __mlir_op.`pop.noalias_pointer_cast`(self.address)

    @always_inline("nodebug")
    fn load[
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = _default_alignment[dtype, width](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[mut](),
    ](self: UnsafePointer[Scalar[dtype], **_]) -> SIMD[dtype, width]:
        """Loads the value the pointer points to.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.
            invariant: Whether the memory is load invariant.

        Returns:
            The loaded value.
        """
        _simd_construction_checks[dtype, width]()
        constrained[
            alignment > 0, "alignment must be a positive integer value"
        ]()
        constrained[
            not volatile or volatile ^ invariant,
            "both volatile and invariant cannot be set at the same time",
        ]()

        @parameter
        if is_nvidia_gpu() and sizeof[dtype]() == 1 and alignment == 1:
            # LLVM lowering to PTX incorrectly vectorizes loads for 1-byte types
            # regardless of the alignment that is passed. This causes issues if
            # this method is called on an unaligned pointer.
            # TODO #37823 We can make this smarter when we add an `aligned`
            # trait to the pointer class.
            var v = SIMD[dtype, width]()

            # intentionally don't unroll, otherwise the compiler vectorizes
            for i in range(width):
                v[i] = __mlir_op.`pop.load`[
                    alignment = alignment.value,
                    isVolatile = volatile.value,
                    isInvariant = invariant.value,
                ]((self + i).address)
            return v

        var address = self.bitcast[SIMD[dtype, width]]().address

        return __mlir_op.`pop.load`[
            alignment = alignment.value,
            isVolatile = volatile.value,
            isInvariant = invariant.value,
        ](address)

    @always_inline("nodebug")
    fn load[
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = _default_alignment[dtype, width](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[mut](),
    ](self: UnsafePointer[Scalar[dtype], **_], offset: Scalar) -> SIMD[
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
        constrained[offset.dtype.is_integral(), "offset must be integer"]()
        return self.offset(Int(offset)).load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
        ]()

    @always_inline("nodebug")
    fn load[
        I: Indexer,
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = _default_alignment[dtype, width](),
        volatile: Bool = False,
        invariant: Bool = _default_invariant[mut](),
    ](self: UnsafePointer[Scalar[dtype], **_], offset: I) -> SIMD[dtype, width]:
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
        return self.offset(offset).load[
            width=width,
            alignment=alignment,
            volatile=volatile,
            invariant=invariant,
        ]()

    @always_inline("nodebug")
    fn store[
        I: Indexer,
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = _default_alignment[dtype, width](),
        volatile: Bool = False,
    ](
        self: UnsafePointer[Scalar[dtype], **_],
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
        constrained[mut, _must_be_mut_err]()
        self.offset(offset).store[alignment=alignment, volatile=volatile](val)

    @always_inline("nodebug")
    fn store[
        dtype: DType,
        offset_type: DType, //,
        width: Int = 1,
        *,
        alignment: Int = _default_alignment[dtype, width](),
        volatile: Bool = False,
    ](
        self: UnsafePointer[Scalar[dtype], **_],
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
        constrained[mut, _must_be_mut_err]()
        constrained[offset_type.is_integral(), "offset must be integer"]()
        self.offset(Int(offset))._store[alignment=alignment, volatile=volatile](
            val
        )

    @always_inline("nodebug")
    fn store[
        dtype: DType, //,
        width: Int = 1,
        *,
        alignment: Int = _default_alignment[dtype, width](),
        volatile: Bool = False,
    ](self: UnsafePointer[Scalar[dtype], **_], val: SIMD[dtype, width]):
        """Stores a single element value.

        Constraints:
            The width and alignment must be positive integer values.

        Parameters:
            dtype: The data type of SIMD vector elements.
            width: The size of the SIMD vector.
            alignment: The minimal alignment of the address.
            volatile: Whether the operation is volatile or not.

        Args:
            val: The value to store.
        """
        constrained[mut, _must_be_mut_err]()
        self._store[alignment=alignment, volatile=volatile](val)

    @always_inline("nodebug")
    fn _store[
        dtype: DType,
        width: Int,
        *,
        alignment: Int = _default_alignment[dtype, width](),
        volatile: Bool = False,
    ](self: UnsafePointer[Scalar[dtype], **_], val: SIMD[dtype, width]):
        constrained[mut, _must_be_mut_err]()
        constrained[width > 0, "width must be a positive integer value"]()
        constrained[
            alignment > 0, "alignment must be a positive integer value"
        ]()

        __mlir_op.`pop.store`[
            alignment = alignment.value, isVolatile = volatile.value
        ](val, self.bitcast[SIMD[dtype, width]]().address)

    @always_inline("nodebug")
    fn strided_load[
        dtype: DType, T: Intable, //, width: Int
    ](self: UnsafePointer[Scalar[dtype], **_], stride: T) -> SIMD[dtype, width]:
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
        return strided_load(self, Int(stride), SIMD[DType.bool, width](True))

    @always_inline("nodebug")
    fn strided_store[
        dtype: DType,
        T: Intable, //,
        width: Int = 1,
    ](
        self: UnsafePointer[Scalar[dtype], **_],
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
        constrained[mut, _must_be_mut_err]()
        strided_store(val, self, Int(stride), True)

    @always_inline("nodebug")
    fn gather[
        dtype: DType, //,
        *,
        width: Int = 1,
        alignment: Int = _default_alignment[dtype, width](),
    ](
        self: UnsafePointer[Scalar[dtype], **_],
        offset: SIMD[_, width],
        mask: SIMD[DType.bool, width] = True,
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
        constrained[
            offset.dtype.is_integral(),
            "offset type must be an integral type",
        ]()
        constrained[
            alignment.is_power_of_two(),
            "alignment must be a power of two integer value",
        ]()

        var base = offset.cast[DType.index]().fma(sizeof[dtype](), Int(self))
        return gather(base, mask, default, alignment)

    @always_inline("nodebug")
    fn scatter[
        dtype: DType, //,
        *,
        width: Int = 1,
        alignment: Int = _default_alignment[dtype, width](),
    ](
        self: UnsafePointer[Scalar[dtype], **_],
        offset: SIMD[_, width],
        val: SIMD[dtype, width],
        mask: SIMD[DType.bool, width] = True,
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
        constrained[mut, _must_be_mut_err]()
        constrained[
            offset.dtype.is_integral(),
            "offset type must be an integral type",
        ]()
        constrained[
            alignment.is_power_of_two(),
            "alignment must be a power of two integer value",
        ]()

        var base = offset.cast[DType.index]().fma(sizeof[dtype](), Int(self))
        scatter(val, base, mask, alignment)

    @always_inline
    fn free(self: UnsafePointer[_, address_space = AddressSpace.GENERIC, **_]):
        """Free the memory referenced by the pointer."""
        _free(self)

    @always_inline("builtin")
    fn bitcast[
        T: AnyType = Self.type,
    ](self) -> UnsafePointer[
        T,
        address_space=address_space,
        alignment=alignment,
        mut=mut,
        origin=origin,
    ]:
        """Bitcasts a UnsafePointer to a different type.

        Parameters:
            T: The target type.

        Returns:
            A new UnsafePointer object with the specified type and the same address,
            as the original UnsafePointer.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = UnsafePointer[
                T, address_space=address_space, alignment=alignment
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn static_alignment_cast[
        alignment: Int = Self.alignment
    ](self) -> UnsafePointer[
        type,
        address_space=address_space,
        alignment=alignment,
        mut=mut,
        origin=origin,
    ]:
        """Changes the `alignment` of an `UnsafePointer`.

        The static alignment of an UnsafePointer must be greater
        or equal to the actual alignment of the runtime pointer
        value. Casting an UnsafePointer to a static alignment greater
        than its runtime alignment may cause undefined behavior".

        This only changes the compile-time alignment encoded in the type of
        this pointer. This does not change the alignment of the pointer address
        at runtime.


        Parameters:
            alignment: Alignment of the destination pointer.

        Returns:
            A new UnsafePointer object with the same type, address_space, and address,
            as the original UnsafePointer, and the new specified alignment.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = UnsafePointer[
                type, address_space=address_space, alignment=alignment
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn origin_cast[
        mut: Bool = Self.mut,
        origin: Origin[mut] = Origin[mut].cast_from[Self.origin],
    ](self) -> UnsafePointer[
        type,
        address_space=address_space,
        alignment=alignment,
        mut=mut,
        origin=origin,
    ]:
        """Changes the origin or mutability of a pointer.

        Parameters:
            mut: Whether the origin is mutable.
            origin: Origin of the destination pointer.

        Returns:
            A new UnsafePointer object with the same type and the same address,
            as the original UnsafePointer and the new specified mutability and origin.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = UnsafePointer[
                type, address_space=address_space, alignment=alignment
            ]._mlir_type,
        ](self.address)

    @always_inline("builtin")
    fn address_space_cast[
        address_space: AddressSpace = Self.address_space,
    ](self) -> UnsafePointer[
        type,
        address_space=address_space,
        alignment=alignment,
        mut=mut,
        origin=origin,
    ]:
        """Casts an UnsafePointer to a different address space.

        Parameters:
            address_space: The address space of the result.

        Returns:
            A new UnsafePointer object with the same type and the same address,
            as the original UnsafePointer and the new address space.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type = UnsafePointer[
                type, address_space=address_space, alignment=alignment
            ]._mlir_type,
        ](self.address)

    @always_inline
    fn destroy_pointee(
        self: UnsafePointer[type, address_space = AddressSpace.GENERIC, **_]
    ):
        """Destroy the pointed-to value.

        The pointer must not be null, and the pointer memory location is assumed
        to contain a valid initialized instance of `type`.  This is equivalent to
        `_ = self.take_pointee()` but doesn't require `Movable` and is
        more efficient because it doesn't invoke `__moveinit__`.

        """
        constrained[mut, _must_be_mut_err]()
        _ = __get_address_as_owned_value(self.address)

    @always_inline
    fn take_pointee[
        T: Movable, //,
    ](self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_]) -> T:
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
        constrained[mut, _must_be_mut_err]()
        return __get_address_as_owned_value(self.address)

    # TODO: Allow overloading on more specific traits
    @always_inline
    fn init_pointee_move[
        T: Movable, //,
    ](
        self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
        owned value: T,
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
        constrained[mut, _must_be_mut_err]()
        __get_address_as_uninit_lvalue(self.address) = value^

    @always_inline
    fn init_pointee_copy[
        T: Copyable, //,
    ](
        self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
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
        constrained[mut, _must_be_mut_err]()
        __get_address_as_uninit_lvalue(self.address) = value

    @always_inline
    fn init_pointee_explicit_copy[
        T: ExplicitlyCopyable, //
    ](
        self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
        value: T,
    ):
        """Emplace a copy of `value` into this pointer location.

        The pointer memory location is assumed to contain uninitialized data,
        and consequently the current contents of this pointer are not destructed
        before writing `value`. Similarly, ownership of `value` is logically
        transferred into the pointer location.

        When compared to `init_pointee_move`, this avoids an extra move on
        the callee side when the value must be copied.

        Parameters:
            T: The type the pointer points to, which must be
               `ExplicitlyCopyable`.

        Args:
            value: The value to emplace.
        """
        constrained[mut, _must_be_mut_err]()
        __get_address_as_uninit_lvalue(self.address) = value.copy()

    @always_inline
    fn move_pointee_into[
        T: Movable, //,
    ](
        self: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
        dst: UnsafePointer[T, address_space = AddressSpace.GENERIC, **_],
    ):
        """Moves the value `self` points to into the memory location pointed to by
        `dst`.

        This performs a consuming move (using `__moveinit__()`) out of the
        memory location pointed to by `self`. Subsequent reads of this
        pointer are not valid unless and until a new, valid value has been
        moved into this pointer's memory location using `init_pointee_move()`.

        This transfers the value out of `self` and into `dest` using at most one
        `__moveinit__()` call.

        **Safety:**

        * `self` must be non-null
        * `self` must contain a valid, initialized instance of `T`
        * `dst` must not be null
        * The contents of `dst` should be uninitialized. If `dst` was
            previously written with a valid value, that value will be be
            overwritten and its destructor will NOT be run.

        Parameters:
            T: The type the pointer points to, which must be `Movable`.

        Args:
            dst: Destination pointer that the value will be moved into.
        """
        constrained[mut, _must_be_mut_err]()
        __get_address_as_uninit_lvalue(
            dst.address
        ) = __get_address_as_owned_value(self.address)


alias OpaquePointer = UnsafePointer[NoneType]
"""An opaque pointer, equivalent to the C `void*` type."""
