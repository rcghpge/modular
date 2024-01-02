# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import AttrMap
from .mobicc import AttrPtr, GraphPtr, SymbolPtr, TuplePtr, TypePtr
from .module import Module
from .symbol import Symbol, Tup
from .motype import dyn, MOType, MOTensor, Arity

from memory.unsafe import Pointer
from tensor import TensorShape, TensorSpec


@value
struct Graph:
    var g: GraphPtr
    var m: Module

    fn __init__(inout self, g: GraphPtr):
        self.g = g
        self.m = Module(mobicc.get_graph_module(g))

    fn __getitem__(self, pos: UInt32) -> Symbol:
        return Symbol(mobicc.get_input(self.g, pos))

    fn nvop(inout self, name: StringRef, inputs: Tup) -> Tup:
        return self.nvop(
            name,
            inputs,
            Arity(mobicc.new_arity()),
            AttrMap(mobicc.new_attr_map(), 0),
        )

    fn nvop(inout self, name: StringRef, inputs: Tup, out_types: Arity) -> Tup:
        return self.nvop(
            name, inputs, out_types, AttrMap(mobicc.new_attr_map(), 0)
        )

    fn nvop(
        inout self, name: StringRef, out_types: Arity, attrs: AttrMap
    ) -> Tup:
        return self.nvop(name, Tup(), out_types, attrs)

    fn nvop(
        inout self,
        name: StringRef,
        inputs: Tup,
        out_types: Arity,
        attrs: AttrMap,
    ) -> Tup:
        let loc = self.m.unknown_loc()
        let outputs = mobicc.new_op(
            self.g, loc, name, inputs.t, out_types.to_mlir(self.m), attrs.m
        )
        return Tup(outputs)

    # Shorthand helpers for single-valued ops.
    # TODO: s/T/Arity/ when MOTensor can be autoboxed inside an Arity.

    fn op[
        T: MOType
    ](inout self, name: StringRef, inputs: Tup, out_type: T) -> Symbol:
        return self.nvop(
            name,
            inputs,
            Arity(out_type.to_mlir(self.m)),
            AttrMap(mobicc.new_attr_map(), 0),
        )[0]

    fn op[
        T: MOType
    ](inout self, name: StringRef, out_type: T, attrs: AttrMap,) -> Symbol:
        return self.nvop(name, Arity(out_type.to_mlir(self.m)), attrs)[0]

    fn op[
        T: MOType
    ](
        inout self,
        name: StringRef,
        inputs: Tup,
        out_type: T,
        attrs: AttrMap,
    ) -> Symbol:
        return self.nvop(name, inputs, Arity(out_type.to_mlir(self.m)), attrs)[
            0
        ]

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
        return self.nvop(
            "mo.constant",
            Arity(type),
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

        return self.nvop("mo.constant", Tup(), Arity(t), m.attrs(attr))[0]

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

        return self.nvop("mo.constant", Tup(), Arity(t), m.attrs(attr))[0]

    fn i64_constant(inout self, t: TypePtr, *values: Int64) raises -> Symbol:
        # TODO: Fix the checks once we have Mojo structured types
        if not (
            mobicc.type_is_si_tensor(t) and (mobicc.tensor_dtype_width(t) == 64)
        ):
            raise "Type must be an i64 tensor"
        return self.nvop(
            "mo.constant",
            Arity(t),
            self.m.attrs(self.m.i64_attr("value", t, values)),
        )[0]

    fn i64_constant(inout self, t: TypePtr, value: Int) raises -> Symbol:
        # This special overload prevents the DynamicVector overload from
        # stealing precedence due to unpredictable resolution order around
        # implicit casting.
        return self.i64_constant(t, Int64(value))

    fn i64_constant(
        inout self, t: TypePtr, values: DynamicVector[Int64]
    ) raises -> Symbol:
        # TODO: This isn't great, can't overload to Int32
        # TODO: Fix the checks once we have Mojo structured types
        if not (
            mobicc.type_is_si_tensor(t) and (mobicc.tensor_dtype_width(t) == 64)
        ):
            raise "Type must be an i64 tensor"
        return self.nvop(
            "mo.constant",
            Arity(t),
            self.m.attrs(self.m.i64_attr("value", t, values)),
        )[0]

    fn f32_constant(inout self, t: TypePtr, *values: Float32) raises -> Symbol:
        # TODO: Fix the checks once we have Mojo structured types
        if not mobicc.type_is_float_tensor(t):
            raise "Type must be an f32 tensor"
        return self.nvop(
            "mo.constant",
            Arity(t),
            self.m.attrs(self.m.f32_attr("value", t, values)),
        )[0]

    fn output(inout self) raises:
        _ = self.nvop("mo.output", Tup())

    fn output(inout self, outs: Tup) raises:
        _ = self.nvop("mo.output", outs)

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
        )

    fn arange[dtype: DType](inout self, stop: SIMD[dtype, 1]) raises -> Symbol:
        return self.arange[dtype](0, stop, 1)

    fn arange[
        dtype: DType
    ](inout self, start: SIMD[dtype, 1], stop: SIMD[dtype, 1]) raises -> Symbol:
        return self.arange[dtype](start, stop, 1)
