# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap
from .mobicc import AttrPtr, GraphPtr, SymbolPtr, TuplePtr, TypePtr
from .module import Module
from .symbol import Symbol, Tup
from .type import Type
from .motype import MOTensor, dyn
from .motype import Type as MOType

from memory.unsafe import Pointer
from tensor import TensorShape, TensorSpec


@value
struct Graph:
    var g: GraphPtr
    var m: Module

    fn __init__(inout self, g: GraphPtr):
        self.g = g
        self.m = Module(mobicc.get_graph_module(self.g))

    fn __getitem__(self, pos: UInt32) -> Symbol:
        return Symbol(mobicc.get_input(self.g, pos))

    fn op(inout self, name: StringRef, input: Symbol, out_type: Type) -> Tup:
        return self.op(name, Tup(input), Arity(out_type))

    fn op[
        T: MOType
    ](inout self, name: StringRef, input: Symbol, out_type: T) -> Tup:
        return self.op(name, Tup(input), Arity(Type(out_type.to_mlir(self.m))))

    fn op(inout self, name: StringRef, input: Symbol, out_types: Arity) -> Tup:
        return self.op(name, Tup(input), out_types)

    fn op(inout self, name: StringRef, inputs: Tup, out_types: Arity) -> Tup:
        return self.op(
            name, inputs, out_types, AttrMap(mobicc.new_attr_map(), 0)
        )

    fn op[
        T: MOType
    ](inout self, name: StringRef, inputs: Tup, out_type: T) -> Tup:
        return self.op(
            name,
            inputs,
            Arity(Type(out_type.to_mlir(self.m))),
            AttrMap(mobicc.new_attr_map(), 0),
        )

    fn op[
        T: MOType
    ](
        inout self,
        name: StringRef,
        inputs: Tup,
        out_type: T,
        attrs: AttrMap,
    ) -> Tup:
        return self.op(
            name, inputs, Arity(Type(out_type.to_mlir(self.m))), attrs
        )

    fn op[
        T: MOType
    ](inout self, name: StringRef, out_type: T, attrs: AttrMap,) -> Tup:
        return self.op(name, Arity(Type(out_type.to_mlir(self.m))), attrs)

    fn op(
        inout self,
        name: StringRef,
        out_types: Arity,
        attrs: AttrMap,
    ) -> Tup:
        return self.op(name, Tup(), out_types, attrs)

    fn op(
        inout self,
        name: StringRef,
        inputs: Tup,
        out_types: Arity,
        attrs: AttrMap,
    ) -> Tup:
        let loc = self.m.unknown_loc()
        let outputs = mobicc.new_op(
            self.g, loc, name, inputs.t, out_types.a, attrs.m
        )
        return Tup(outputs)

    # TODO: Separate DType/Tensor/TensorSpec adapters from Graph.

    fn constant[
        dtype: DType
    ](inout self, shape: TensorShape, value: DTypePointer[dtype]) raises -> Tup:
        """Takes ownership of `value` to avoid a copy."""
        return self.constant[dtype](Tensor[dtype](value, shape))

    fn constant[
        dtype: DType
    ](inout self, owned value: Tensor[dtype]) raises -> Symbol:
        """Takes ownership of `value` to avoid a copy."""
        let m = self.m
        let type = m.type_of(value)
        return self.op(
            "mo.constant",
            m.types(type),
            m.attrs(m.tensor_attr("value", type, value)),
        )[0]

    fn constant[
        dtype: DType
    ](inout self, value: SIMD[dtype, 1]) raises -> Symbol:
        let m = self.m

        var val_copy = value
        let t = mobicc.type_from_spec(m.m, TensorSpec(dtype, TensorShape()))
        let value_ptr = DTypePointer[dtype].address_of(val_copy).bitcast[
            DType.invalid
        ]()
        let attr = mobicc.tensor_attr(
            m.m, "value", value_ptr, t, is_owned=False
        )

        return self.op("mo.constant", Tup(), m.types(t), m.attrs(attr))[0]

    fn constant(inout self, value: Bool) raises -> Symbol:
        let v: SIMD[DType.bool, 1] = value
        return self.constant(v)

    fn constant(inout self, value: Int) raises -> Symbol:
        let v: Int64 = value
        return self.constant(v)

    fn constant(inout self, value: FloatLiteral) raises -> Symbol:
        let v: Float32 = value
        return self.constant(v)

    fn constant(inout self, value: Int, dtype: DType) raises -> Symbol:
        let m = self.m

        var val_copy = value
        let t = mobicc.type_from_spec(m.m, TensorSpec(dtype, TensorShape()))
        let value_ptr = Pointer[Int].address_of(val_copy).bitcast[AnyRegType]()
        let attr = mobicc.tensor_attr(
            m.m, "value", value_ptr, t, is_owned=False
        )

        return self.op("mo.constant", Tup(), m.types(t), m.attrs(attr))[0]

    fn i64_constant(inout self, t: TypePtr, *values: Int64) raises -> Symbol:
        # TODO: This isn't great, can't overload to Int32
        # TODO: Fix the checks once we have Mojo structured types
        if not (
            mobicc.type_is_si_tensor(t) and (mobicc.tensor_dtype_width(t) == 64)
        ):
            raise "Type must be an i64 tensor"
        return self.op(
            "mo.constant",
            self.m.types(t),
            self.m.attrs(self.m.i64_attr("value", t, values)),
        )[0]

    fn f32_constant(inout self, t: TypePtr, *values: Float32) raises -> Symbol:
        # TODO: This isn't great, can't overload to Float64
        # TODO: Fix the checks once we have Mojo structured types
        if not mobicc.type_is_float_tensor(t):
            raise "Type must be an f32 tensor"
        return self.op(
            "mo.constant",
            self.m.types(t),
            self.m.attrs(self.m.f32_attr("value", t, values)),
        )[0]

    fn output(inout self, outs: Tup) raises:
        _ = self.op("mo.output", outs, self.m.types())

    fn arange[
        dtype: DType
    ](
        inout self,
        start: SIMD[dtype, 1],
        stop: SIMD[dtype, 1],
        step: SIMD[dtype, 1],
    ) raises -> Symbol:
        return self.op(
            "mo.range",
            (
                self.constant[dtype](start),
                self.constant[dtype](stop),
                self.constant[dtype](step),
            ),
            MOTensor(dtype, dyn()),
        )[0]

    fn arange[dtype: DType](inout self, stop: SIMD[dtype, 1]) raises -> Symbol:
        return self.arange[dtype](0, stop, 1)

    fn arange[
        dtype: DType
    ](inout self, start: SIMD[dtype, 1], stop: SIMD[dtype, 1]) raises -> Symbol:
        return self.arange[dtype](start, stop, 1)
