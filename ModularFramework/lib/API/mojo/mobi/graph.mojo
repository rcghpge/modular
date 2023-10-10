# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap, Tup
from .mobicc import Builder, ModulePtr, GraphPtr, SymbolPtr
from utils.list import VariadicList


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
        let l = VariadicList(values)
        var t = self.builder.new_tuple()
        for i in range(len(l)):
            var v = l[i]
            self.builder.add_symbol(t, v)
        return Tup(t, len(l), self.builder)

    fn no_value(inout self) raises -> Tup:
        let t = self.builder.new_tuple()
        return Tup(t, 0, self.builder)

    fn op(
        inout self,
        name: StringRef,
        inputs: Tup,
        outTypes: Arity,
    ) raises -> Tup:
        var loc = self.builder.unknown_loc(self.m)
        let attrs = self.builder.new_attr_map()
        let outputs = self.builder.new_op(
            self.g, loc, name, inputs.t, outTypes.a, attrs
        )
        return Tup(outputs, outTypes.l, self.builder)

    fn op(
        inout self,
        name: StringRef,
        inputs: Tup,
        outTypes: Arity,
        attrs: AttrMap,
    ) raises -> Tup:
        var loc = self.builder.unknown_loc(self.m)
        let outputs = self.builder.new_op(
            self.g, loc, name, inputs.t, outTypes.a, attrs.m
        )
        return Tup(outputs, outTypes.l, self.builder)
