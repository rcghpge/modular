# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap, Tup
from .mobicc import Builder, ModulePtr, GraphPtr, SymbolPtr, TuplePtr


struct Graph:
    var g: GraphPtr
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
