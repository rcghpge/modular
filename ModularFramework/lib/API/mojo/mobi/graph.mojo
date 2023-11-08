# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap, Tup
from .mobicc import AttrPtr, GraphPtr, SymbolPtr, TuplePtr
from .module import Module
from .symbol import Symbol
from .type import Type

from memory.unsafe import Pointer
from tensor import TensorShape


# XXX abstraction boundary feels pretty weird here
#     it's really unclear what functions should go here vs in module
@value
struct Graph:
    # TODO(#23665): This should be an owning pointer.
    var g: GraphPtr
    var m: Module

    fn __init__(inout self, g: GraphPtr, m: Module):
        self.g = g
        self.m = m

    fn __getitem__(self, pos: UInt32) raises -> Symbol:
        return Symbol(mobicc.get_input(self.g, pos), self)

    fn values(self, *values: Symbol) raises -> Tup:
        var t = mobicc.new_tuple()
        for i in range(len(values)):
            let val = __get_address_as_lvalue(values[i])
            mobicc.add_symbol(t, val.s)
        return Tup(t, len(values), self)

    fn op(
        inout self,
        name: StringRef,
        input: Symbol,
        out_type: Type,
    ) raises -> Tup:
        return self.op(name, Tup(input), Arity(out_type))

    fn op(
        inout self,
        name: StringRef,
        input: Symbol,
        out_types: Arity,
    ) raises -> Tup:
        return self.op(name, Tup(input), out_types)

    fn op(
        inout self,
        name: StringRef,
        inputs: Tup,
        out_types: Arity,
    ) raises -> Tup:
        let loc = self.m.unknown_loc()
        let attrs = mobicc.new_attr_map()
        let outputs = mobicc.new_op(
            self.g, loc, name, inputs.t, out_types.a, attrs
        )
        return Tup(outputs, out_types.l, self)

    fn op(
        inout self,
        name: StringRef,
        inputs: Tup,
        out_types: Arity,
        attrs: AttrMap,
    ) raises -> Tup:
        let loc = self.m.unknown_loc()
        let outputs = mobicc.new_op(
            self.g, loc, name, inputs.t, out_types.a, attrs.m
        )
        return Tup(outputs, out_types.l, self)

    fn constant[
        dtype: DType
    ](
        inout self,
        shape: TensorShape,
        value: DTypePointer[dtype],
    ) raises -> Tup:
        return self.constant[dtype](Tensor[dtype](value, shape))

    fn constant[
        dtype: DType
    ](inout self, value: Tensor[dtype]) raises -> Symbol:
        let m = self.m
        let type = m.type_of(value)
        return self.op(
            "mo.constant",
            self.values(),
            m.types(type),
            m.attrs(m.tensor_attr("const", type, value)),
        )[0]

    fn constant[
        dtype: DType = DType.float32
    ](inout self, value: Float32) raises -> Symbol:
        constrained[dtype == DType.float32]()  # TODO: add more float types
        let m = self.m
        return self.op(
            "mo.constant",
            self.values(),
            m.types(m.f32()),
            m.attrs(m.f32_attr("const", m.f32(), value)),
        )[0]

    fn constant[
        dtype: DType = DType.uint64
    ](inout self, value: Int) raises -> Symbol:
        let m = self.m
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
        return self.op("mo.constant", self.values(), arity, m.attrs(attr))[0]

    fn output(inout self, outs: Tup) raises:
        _ = self.op("mo.output", outs, self.m.types())
