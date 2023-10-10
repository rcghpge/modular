# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap
from .mobicc import AttrPtr, AttrMapPtr, ModulePtr, Builder, TypePtr, GraphPtr
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

    fn i32_scalar_attr(
        inout self, name: StringRef, value: Int32
    ) raises -> AttrPtr:
        var valueCopy = value
        let dataPtr = Pointer[Int32].address_of(valueCopy).bitcast[AnyType]()
        return self.builder.tensor_attr(
            self.m, name, dataPtr, self.i32_scalar()
        )

    fn attrs(inout self, *attrs: AttrPtr) raises -> AttrMap:
        let l = VariadicList(attrs)
        var m = self.builder.new_attr_map()
        for i in range(len(l)):
            var a = l[i]
            self.builder.add_attr(m, a)
        return AttrMap(m, len(l))

    fn i32_scalar(inout self) raises -> TypePtr:
        return self.builder.si32_scalar(self.m)

    fn types(inout self, *types: TypePtr) raises -> Arity:
        let l = VariadicList(types)
        var a = self.builder.new_arity()
        for i in range(len(l)):
            var t = l[i]
            self.builder.add_type(a, t)
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
