# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .mobicc import (
    ModulePtr,
    Builder,
    TypePtr,
    GraphPtr,
    SymbolPtr,
    TuplePtr,
    LocPtr,
)
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

    fn values(inout self, *values: SymbolPtr) -> VariadicList[SymbolPtr]:
        return VariadicList(values)

    fn no_value(inout self) -> VariadicList[SymbolPtr]:
        return VariadicList[SymbolPtr]()

    fn input(
        inout self,
        pos: UInt32,
    ) raises -> SymbolPtr:
        return self.builder.get_graph_input(self.g, pos)

    fn _new_op(
        inout self,
        name: StringRef,
        inputs: VariadicList[SymbolPtr],
        outTypes: VariadicList[TypePtr],
    ) raises -> TuplePtr:
        var ins = self.builder.new_tuple()
        for i in range(len(inputs)):
            var t = inputs[i]
            self.builder.ext_tuple(ins, t)
        var outs = self.builder.new_arity()
        for i in range(len(outTypes)):
            var t = outTypes[i]
            self.builder.inc_arity(outs, t)
        var loc = self.builder.unknown_loc(self.m)
        # TODO: Why crash when trying to return VariadicList[SymbolPtr]?
        return self.builder.new_op(self.g, loc, name, ins, outs)

    fn void_op(
        inout self,
        name: StringRef,
        inputs: VariadicList[SymbolPtr],
        outTypes: VariadicList[TypePtr],
    ) raises:
        _ = self._new_op(name, inputs, outTypes)

    fn single_op(
        inout self,
        name: StringRef,
        inputs: VariadicList[SymbolPtr],
        outTypes: VariadicList[TypePtr],
    ) raises -> SymbolPtr:
        var retTuple = self._new_op(name, inputs, outTypes)
        return self.builder.get_tuple_item(retTuple, 0)
