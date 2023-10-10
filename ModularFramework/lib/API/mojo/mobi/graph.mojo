# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, Tup
from .mobicc import Builder, ModulePtr, GraphPtr, SymbolPtr
from utils.list import VariadicList


struct _ArgsView:
    var g: GraphPtr
    var builder: Builder

    fn __init__(inout self, g: GraphPtr, builder: Builder):
        self.g = g
        self.builder = builder

    fn __getitem__(inout self, pos: UInt32) raises -> SymbolPtr:
        return self.builder.get_graph_input(self.g, pos)


struct Graph:
    var g: GraphPtr
    var m: ModulePtr
    var builder: Builder

    var inputs: _ArgsView

    fn __init__(
        inout self, g: GraphPtr, m: ModulePtr, borrowed builder: Builder
    ):
        self.g = g
        self.m = m
        # TODO: Any concerns about copying dlhandles like this?
        self.builder = builder
        self.inputs = _ArgsView(g, builder)

    # TODO: Add __str__/__repr__.

    fn values(inout self, *values: SymbolPtr) raises -> Tup:
        let l = VariadicList(values)
        var t = self.builder.new_tuple()
        for i in range(len(l)):
            var v = l[i]
            self.builder.ext_tuple(t, v)
        return Tup(t, len(l))

    fn no_value(inout self) raises -> Tup:
        let t = self.builder.new_tuple()
        return Tup(t, 0)

    fn input(
        inout self,
        pos: UInt32,
    ) raises -> SymbolPtr:
        return self.builder.get_graph_input(self.g, pos)

    fn op(
        inout self,
        name: StringRef,
        inputs: Tup,
        outTypes: Arity,
    ) raises -> Tup:
        var loc = self.builder.unknown_loc(self.m)
        let outputs = self.builder.new_op(
            self.g, loc, name, inputs.t, outTypes.a
        )
        return Tup(outputs, outTypes.l)
