# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .mobicc import ModulePtr, Builder, TypePtr, GraphPtr, SymbolPtr, TuplePtr
from utils.list import VariadicList


struct Module:
    var m: ModulePtr
    var builder: Builder

    fn __init__(inout self) raises:
        self.builder = Builder()
        self.m = self.builder.new_module()

    fn debug_print(inout self) raises:
        self.builder.debug_print_module(self.m)

    # TODO: Add __str__/__repr__.

    fn i32_scalar(inout self) raises -> TypePtr:
        return self.builder.si32_scalar(self.m)

    fn scalar[dtype: DType](inout self) raises -> TypePtr:
        if dtype == DType.int32:
            return self.i32_scalar()
        # TODO: Just pass it down generically.
        raise "add more dtypes"

    fn types(inout self, *values: TypePtr) raises -> VariadicList[TypePtr]:
        return VariadicList(values)

    fn void(inout self) raises -> VariadicList[TypePtr]:
        return VariadicList[TypePtr]()

    fn graph(
        inout self,
        name: StringRef,
        inTypes: VariadicList[TypePtr],
        outTypes: VariadicList[TypePtr],
    ) raises -> Graph:
        var ins = self.builder.new_arity()
        for i in range(len(inTypes)):
            var t = inTypes[i]
            self.builder.inc_arity(ins, t)
        var outs = self.builder.new_arity()
        for i in range(len(outTypes)):
            var t = outTypes[i]
            self.builder.inc_arity(outs, t)
        var unknown = self.builder.unknown_loc(self.m)
        return Graph(
            self.builder.new_graph(self.m, unknown, name, ins, outs),
            self.m,
            self.builder,
        )
