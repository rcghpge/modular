# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap, Tup
from .mobicc import AttrPtr, Builder, ModulePtr, GraphPtr, SymbolPtr, TuplePtr
from .module import Module

from tensor import TensorShape


# XXX abstraction boundary feels pretty weird here
#     it's really unclear what functions should go here vs in module
struct Graph:
    var g: GraphPtr
    # TODO: yuck. this should be able to be a Pointer[Module]
    #       but pointers to nontrivial Mojo types aren't supported yet.
    #       for now we pass the Module object in to any methods that need it.
    var m: ModulePtr
    var builder: Builder

    fn __init__(
        inout self, g: GraphPtr, m: ModulePtr, borrowed builder: Builder
    ):
        self.g = g
        self.m = m
        self.builder = builder

    # TODO: Add __str__/__repr__.

    fn __getitem__(inout self, pos: UInt32) raises -> SymbolPtr:
        return self.builder.get_input(self.g, pos)

    fn values(inout self, *values: SymbolPtr) raises -> Tup:
        var t = self.builder.new_tuple()
        for i in range(len(values)):
            var v = values[i]
            self.builder.add_symbol(t, v)
        return Tup(t, len(values), self.builder)

    fn op(
        inout self,
        name: StringRef,
        inputs: Tup,
        out_types: Arity,
    ) raises -> Tup:
        var loc = self.builder.unknown_loc(self.m)
        let attrs = self.builder.new_attr_map()
        let outputs = self.builder.new_op(
            self.g, loc, name, inputs.t, out_types.a, attrs
        )
        return Tup(outputs, out_types.l, self.builder)

    fn op(
        inout self,
        name: StringRef,
        inputs: Tup,
        out_types: Arity,
        attrs: AttrMap,
    ) raises -> Tup:
        var loc = self.builder.unknown_loc(self.m)
        let outputs = self.builder.new_op(
            self.g, loc, name, inputs.t, out_types.a, attrs.m
        )
        return Tup(outputs, out_types.l, self.builder)

    fn constant[
        dtype: DType
    ](
        inout self,
        inout m: Module,
        shape: TensorShape,
        value: DTypePointer[dtype],
    ) raises -> Tup:
        return self.constant[dtype](m, Tensor[dtype](value, shape))

    fn constant[
        dtype: DType
    ](inout self, inout m: Module, value: Tensor[dtype]) raises -> Tup:
        debug_assert(m.m == self.m, "Module pointer mismatch")
        let type = m.type_of(value)
        return self.op(
            "mo.constant",
            self.values(),
            m.types(type),
            m.attrs(m.tensor_attr("const", type, value)),
        )

    fn constant[
        dtype: DType = DType.float32
    ](inout self, inout m: Module, value: Float32) raises -> Tup:
        debug_assert(m.m == self.m, "Module pointer mismatch")
        constrained[dtype == DType.float32]()  # TODO: add more float types
        return self.op(
            "mo.constant",
            self.values(),
            m.types(m.f32()),
            m.attrs(m.f32_attr("const", m.f32(), value)),
        )

    fn constant[
        dtype: DType = DType.uint64
    ](inout self, inout m: Module, value: Int) raises -> Tup:
        debug_assert(m.m == self.m, "Module pointer mismatch")
        let arity: Arity
        let attr: AttrPtr

        if dtype == DType.uint64:
            arity = m.types(m.i64())
            attr = m.i64_attr("const", m.i64(), value)
        elif dtype == DType.uint32:
            arity = m.types(m.i32())
            attr = m.i32_attr("const", m.i32(), value)
        else:
            # These are unused, but the compiler thinks that they might be
            # uninitialized if we don't set them
            arity = m.types(m.i64())
            attr = m.i64_attr("XXX", m.i64(), value)

            raise Error("Unimplemented")
        return self.op("mo.constant", self.values(), arity, m.attrs(attr))
