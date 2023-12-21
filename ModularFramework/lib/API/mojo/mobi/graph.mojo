# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap, Tup
from .mobicc import AttrPtr, GraphPtr, SymbolPtr, TuplePtr, TypePtr
from .module import Module
from .symbol import Symbol
from .type import Type

from memory.unsafe import Pointer
from tensor import TensorShape, TensorSpec


# TODO: Try to drop these special annotations.
# TODO: Move all factories into Module (or vice-versa), it's confusing
# if they're in two places.
@value
@register_passable
struct Graph:
    var g: GraphPtr
    var m: Module

    fn __init__(g: GraphPtr, m: Module) -> Self:
        return Self {g: g, m: m}

    fn __getitem__(self, pos: UInt32) -> Symbol:
        return Symbol(mobicc.get_input(self.g, pos), self)

    fn values(self, *values: Symbol) -> Tup:
        var t = mobicc.new_tuple()
        for s in values:
            mobicc.add_symbol(t, s.s)
        return Tup(t, len(values), self)

    fn op(
        inout self,
        name: StringRef,
        input: Symbol,
        out_type: Type,
    ) -> Tup:
        return self.op(name, Tup(input), Arity(out_type))

    fn op(
        inout self,
        name: StringRef,
        input: Symbol,
        out_types: Arity,
    ) -> Tup:
        return self.op(name, Tup(input), out_types)

    fn op(
        inout self,
        name: StringRef,
        inputs: Tup,
        out_types: Arity,
    ) -> Tup:
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
        out_spec: TensorSpec,
    ) -> Tup:
        return self.op(name, inputs, Arity(self.m.type(out_spec)))

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
        return Tup(outputs, out_types.l, self)

    # TODO: Separate DType/Tensor/TensorSpec adapters from Graph.

    fn constant[
        dtype: DType
    ](
        inout self,
        shape: TensorShape,
        value: DTypePointer[dtype],
    ) raises -> Tup:
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
            self.values(),
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

        return self.op("mo.constant", self.values(), m.types(t), m.attrs(attr))[
            0
        ]

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

        return self.op("mo.constant", self.values(), m.types(t), m.attrs(attr))[
            0
        ]

    fn i64_constant(inout self, t: TypePtr, *values: Int64) raises -> Symbol:
        # TODO: This isn't great, can't overload to Int32
        # TODO: Fix the checks once we have Mojo structured types
        if not (
            mobicc.type_is_si_tensor(t) and (mobicc.tensor_dtype_width(t) == 64)
        ):
            raise "Type must be an i64 tensor"
        return self.op(
            "mo.constant",
            self.values(),
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
            self.values(),
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
            TensorSpec(dtype, self.m.dyn().to_int()),
        )[0]

    fn arange[dtype: DType](inout self, stop: SIMD[dtype, 1]) raises -> Symbol:
        return self.arange[dtype](0, stop, 1)

    fn arange[
        dtype: DType
    ](inout self, start: SIMD[dtype, 1], stop: SIMD[dtype, 1]) raises -> Symbol:
        return self.arange[dtype](start, stop, 1)
