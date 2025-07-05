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
"""Implements the VariadicList and VariadicPack types.

These are Mojo built-ins, so you don't need to import them.
"""

from memory import Pointer

# ===-----------------------------------------------------------------------===#
# VariadicList / VariadicListMem
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _VariadicListIter[type: AnyTrivialRegType](Copyable, Iterator, Movable):
    """Const Iterator for VariadicList.

    Parameters:
        type: The type of the elements in the list.
    """

    alias Element = type

    var index: Int
    var src: VariadicList[type]

    fn __next__(mut self) -> type:
        self.index += 1
        return self.src[self.index - 1]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src) - self.index


@register_passable("trivial")
struct VariadicList[type: AnyTrivialRegType](Sized):
    """A utility class to access variadic function arguments. Provides a "list"
    view of the function argument so that the size of the argument list and each
    individual argument can be accessed.

    Parameters:
        type: The type of the elements in the list.
    """

    alias _mlir_type = __mlir_type[`!kgen.variadic<`, type, `>`]
    var value: Self._mlir_type
    """The underlying storage for the variadic list."""

    alias IterType = _VariadicListIter[type]

    @always_inline
    @implicit
    fn __init__(out self, *value: type):
        """Constructs a VariadicList from a variadic list of arguments.

        Args:
            value: The variadic argument list to construct the variadic list
              with.
        """
        self = value

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, value: Self._mlir_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        self.value = value

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """

        return __mlir_op.`pop.variadic.size`(self.value)

    @always_inline
    fn __getitem__[I: Indexer](self, idx: I) -> type:
        """Gets a single element on the variadic list.

        Args:
            idx: The index of the element to access on the list.

        Parameters:
            I: A type that can be used as an index.

        Returns:
            The element on the list corresponding to the given index.
        """
        return __mlir_op.`pop.variadic.get`(self.value, index(idx))

    @always_inline
    fn __iter__(self) -> Self.IterType:
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return Self.IterType(0, self)


@fieldwise_init
struct _VariadicListMemIter[
    elt_is_mutable: Bool, //,
    elt_type: AnyType,
    elt_origin: Origin[elt_is_mutable],
    list_origin: ImmutableOrigin,
    is_owned: Bool,
]:
    """Iterator for VariadicListMem.

    Parameters:
        elt_is_mutable: Whether the elements in the list are mutable.
        elt_type: The type of the elements in the list.
        elt_origin: The origin of the elements.
        list_origin: The origin of the VariadicListMem.
        is_owned: Whether the elements are owned by the list because they are
                  passed as an 'var' argument.
    """

    alias variadic_list_type = VariadicListMem[
        elt_type, elt_origin._mlir_origin, is_owned
    ]

    var index: Int
    var src: Pointer[
        Self.variadic_list_type,
        list_origin,
    ]

    fn __init__(
        out self, index: Int, ref [list_origin]list: Self.variadic_list_type
    ):
        self.index = index
        self.src = Pointer(to=list)

    fn __next_ref__(mut self) -> ref [elt_origin] elt_type:
        self.index += 1
        return rebind[Self.variadic_list_type.reference_type](
            Pointer(to=self.src[][self.index - 1])
        )[]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


struct VariadicListMem[
    elt_is_mutable: Bool, //,
    element_type: AnyType,
    origin: Origin[elt_is_mutable],
    is_owned: Bool,
](Sized):
    """A utility class to access variadic function arguments of memory-only
    types that may have ownership. It exposes references to the elements in a
    way that can be enumerated.  Each element may be accessed with `elt[]`.

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        mut or owned argument.
        element_type: The type of the elements in the list.
        origin: The origin of the underlying elements.
        is_owned: Whether the elements are owned by the list because they are
                  passed as an 'var' argument.
    """

    alias reference_type = Pointer[element_type, origin]
    alias _mlir_ref_type = Self.reference_type._mlir_type
    alias _mlir_type = __mlir_type[`!kgen.variadic<`, Self._mlir_ref_type, `>`]

    var value: Self._mlir_type
    """The underlying storage, a variadic list of references to elements of the
    given type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    # Provide support for read-only variadic arguments.
    @doc_private
    @always_inline
    @implicit
    fn __init__(out self, value: Self._mlir_type):
        """Constructs a VariadicList from a variadic argument type.

        Args:
            value: The variadic argument to construct the list with.
        """
        self.value = value

    @always_inline
    fn __moveinit__(out self, owned existing: Self):
        """Moves constructor.

        Args:
          existing: The existing VariadicListMem.
        """
        self.value = existing.value

    @always_inline
    fn __del__(owned self):
        """Destructor that releases elements if owned."""

        # Destroy each element if this variadic has owned elements, destroy
        # them.  We destroy in backwards order to match how arguments are
        # normally torn down when CheckLifetimes is left to its own devices.
        @parameter
        if is_owned:
            for i in reversed(range(len(self))):
                UnsafePointer(to=self[i]).destroy_pointee()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __len__(self) -> Int:
        """Gets the size of the list.

        Returns:
            The number of elements on the variadic list.
        """
        return __mlir_op.`pop.variadic.size`(self.value)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__(
        self, idx: Int
    ) -> ref [
        # cast mutability of self to match the mutability of the element,
        # since that is what we want to use in the ultimate reference and
        # the union overall doesn't matter.
        Origin[elt_is_mutable].cast_from[__origin_of(origin, self)]
    ] element_type:
        """Gets a single element on the variadic list.

        Args:
            idx: The index of the element to access on the list.

        Returns:
            A low-level pointer to the element on the list corresponding to the
            given index.
        """
        return __get_litref_as_mvalue(
            __mlir_op.`pop.variadic.get`(self.value, idx.value)
        )

    fn __iter__(
        self,
        out result: _VariadicListMemIter[
            element_type, origin, __origin_of(self), is_owned
        ],
    ):
        """Iterate over the list.

        Returns:
            An iterator to the start of the list.
        """
        return __type_of(result)(0, self)


# ===-----------------------------------------------------------------------===#
# VariadicPack
# ===-----------------------------------------------------------------------===#


alias _AnyTypeMetaType = __type_of(AnyType)


@register_passable
struct VariadicPack[
    elt_is_mutable: Bool, //,
    is_owned: Bool,
    origin: Origin[elt_is_mutable],
    element_trait: _AnyTypeMetaType,
    *element_types: element_trait,
](Sized):
    """A utility class to access variadic pack  arguments and provide an API for
    doing things with them.

    Parameters:
        elt_is_mutable: True if the elements of the list are mutable for an
                        mut or owned argument pack.
        is_owned: Whether the elements are owned by the pack. If so, the pack
                  will release the elements when it is destroyed.
        origin: The origin of the underlying elements.
        element_trait: The trait that each element of the pack conforms to.
        element_types: The list of types held by the argument pack.
    """

    alias _mlir_type = __mlir_type[
        `!lit.ref.pack<:variadic<`,
        element_trait,
        `> `,
        element_types,
        `, `,
        origin._mlir_origin,
        `>`,
    ]

    var _value: Self._mlir_type

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_private
    @always_inline("nodebug")
    # This disables nested origin exclusivity checking because it is taking a
    # raw variadic pack which can have nested origins in it (which this does not
    # dereference).
    @__unsafe_disable_nested_origin_exclusivity
    fn __init__(out self, value: Self._mlir_type):
        """Constructs a VariadicPack from the internal representation.

        Args:
            value: The argument to construct the pack with.
        """
        self._value = value

    @always_inline("nodebug")
    fn __del__(owned self):
        """Destructor that releases elements if owned."""

        @parameter
        if is_owned:

            @parameter
            for i in reversed(range(Self.__len__())):
                UnsafePointer(to=self[i]).destroy_pointee()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    @staticmethod
    fn __len__() -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """

        @parameter
        fn variadic_size(
            x: __mlir_type[`!kgen.variadic<`, element_trait, `>`]
        ) -> Int:
            return __mlir_op.`pop.variadic.size`(x)

        alias result = variadic_size(element_types)
        return result

    @always_inline
    fn __len__(self) -> Int:
        """Return the VariadicPack length.

        Returns:
            The number of elements in the variadic pack.
        """
        return Self.__len__()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__[
        index: Int
    ](self) -> ref [Self.origin] element_types[index.value]:
        """Return a reference to an element of the pack.

        Parameters:
            index: The element of the pack to return.

        Returns:
            A reference to the element.  The Pointer's mutability follows the
            mutability of the pack argument convention.
        """
        litref_elt = __mlir_op.`lit.ref.pack.extract`[index = index.value](
            self._value
        )
        return __get_litref_as_mvalue(litref_elt)

    # ===-------------------------------------------------------------------===#
    # C Pack Utilities
    # ===-------------------------------------------------------------------===#

    alias _kgen_element_types = rebind[
        __mlir_type.`!kgen.variadic<!kgen.type>`
    ](Self.element_types)
    """This is the element_types list lowered to `variadic<type>` type for kgen.
    """
    alias _variadic_pointer_types = __mlir_attr[
        `#kgen.param.expr<variadic_ptr_map, `,
        Self._kgen_element_types,
        `, 0: index>: !kgen.variadic<!kgen.type>`,
    ]
    """Use variadic_ptr_map to construct the type list of the !kgen.pack that
    the !lit.ref.pack will lower to.  It exposes the pointers introduced by the
    references.
    """
    alias _kgen_pack_with_pointer_type = __mlir_type[
        `!kgen.pack<:variadic<type> `, Self._variadic_pointer_types, `>`
    ]
    """This is the !kgen.pack type with pointer elements."""

    @doc_private
    @always_inline("nodebug")
    fn get_as_kgen_pack(self) -> Self._kgen_pack_with_pointer_type:
        """This rebinds `in_pack` to the equivalent `!kgen.pack` with kgen
        pointers."""
        return rebind[Self._kgen_pack_with_pointer_type](self._value)

    alias _variadic_with_pointers_removed = __mlir_attr[
        `#kgen.param.expr<variadic_ptrremove_map, `,
        Self._variadic_pointer_types,
        `>: !kgen.variadic<!kgen.type>`,
    ]
    alias _loaded_kgen_pack_type = __mlir_type[
        `!kgen.pack<:variadic<type> `, Self._variadic_with_pointers_removed, `>`
    ]
    """This is the `!kgen.pack` type that happens if one loads all the elements
    of the pack.
    """

    # Returns all the elements in a kgen.pack.
    # Useful for FFI, such as calling printf. Otherwise, avoid this if possible.
    @doc_private
    @always_inline("nodebug")
    fn get_loaded_kgen_pack(self) -> Self._loaded_kgen_pack_type:
        """This returns the stored KGEN pack after loading all of the elements.
        """
        return __mlir_op.`kgen.pack.load`(self.get_as_kgen_pack())
