# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity
from .mobicc import ModulePtr, Builder, TypePtr, GraphPtr
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

    fn types(inout self, *types: TypePtr) raises -> Arity:
        let l = VariadicList(types)
        var a = self.builder.new_arity()
        for i in range(len(l)):
            var t = l[i]
            self.builder.inc_arity(a, t)
        return Arity(a, len(l))

    fn void(inout self) raises -> Arity:
        let a = self.builder.new_arity()
        return Arity(a, 0)

    fn graph(
        inout self,
        name: StringRef,
        inTypes: Arity,
        outTypes: Arity,
    ) raises -> Graph:
        var unknown = self.builder.unknown_loc(self.m)
        let g = self.builder.new_graph(
            self.m, unknown, name, inTypes.a, outTypes.a
        )
        return Graph(g, self.m, self.builder)
