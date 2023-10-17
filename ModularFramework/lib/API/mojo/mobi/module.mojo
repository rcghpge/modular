# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap
from .mobicc import AttrPtr, AttrMapPtr, ModulePtr, Builder, TypePtr, GraphPtr
from tensor import Tensor, TensorSpec
from utils.list import VariadicList

# TODO: DynamicVectors all need a _del_old() call.


struct Module:
    var m: ModulePtr
    var builder: Builder

    fn __init__(inout self) raises:
        self.builder = Builder()
        self.m = self.builder.new_module()

    fn __str__(inout self) raises -> String:
        return self.builder.module_to_string(self.m)

    fn debug_print_execute(inout self, arg: Tensor[DType.float32]) raises:
        var rtArgs = self.builder.new_rt_tuple()
        var shape = DynamicVector[Int64]()
        for i in range(arg.shape().rank()):
            shape.push_back(arg.dim(i))
        self.builder.add_f32_tensor(
            rtArgs, arg.data().bitcast[DType.invalid](), shape.data, arg.rank()
        )
        var rtOuts = self.builder.new_rt_tuple()
        let execCode = self.builder.execute_module(self.m, rtArgs, rtOuts)
        if execCode != 0:
            raise Error("Execution error")
        self.builder.debug_print_tensors(rtOuts)

    fn debug_print_execute(
        inout self, fileName: StringRef, spec: TensorSpec
    ) raises:
        var rtArgs = self.builder.new_rt_tuple()
        var shape = DynamicVector[Int64]()
        for i in range(spec.rank()):
            shape.push_back(spec[i])
        self.builder.add_f32_tensor_from_file(
            rtArgs, fileName, shape.data, spec.rank()
        )
        var rtOuts = self.builder.new_rt_tuple()
        let execCode = self.builder.execute_module(self.m, rtArgs, rtOuts)
        if execCode != 0:
            raise Error("Execution error")
        self.builder.debug_print_tensors(rtOuts)

    fn i32_attr(
        inout self, name: StringRef, t: TypePtr, *elements: Int32
    ) raises -> AttrPtr:
        let l = VariadicList(elements)
        var data = DynamicVector[Int32]()
        for i in range(len(l)):
            data.push_back(l[i])
        return self.builder.tensor_attr(
            self.m, name, data.data.bitcast[AnyType](), t
        )

    fn i64_attr(
        inout self, name: StringRef, t: TypePtr, *elements: Int64
    ) raises -> AttrPtr:
        let l = VariadicList(elements)
        var data = DynamicVector[Int64]()
        for i in range(len(l)):
            data.push_back(l[i])
        return self.builder.tensor_attr(
            self.m, name, data.data.bitcast[AnyType](), t
        )

    fn f32_attr(
        inout self, name: StringRef, t: TypePtr, *elements: Float32
    ) raises -> AttrPtr:
        let l = VariadicList(elements)
        var data = DynamicVector[Float32]()
        for i in range(len(l)):
            data.push_back(l[i])
        return self.builder.tensor_attr(
            self.m, name, data.data.bitcast[AnyType](), t
        )

    fn f32_attr(
        inout self, name: StringRef, t: TypePtr, value: Tensor[DType.float32]
    ) raises -> AttrPtr:
        return self.builder.tensor_attr(
            self.m, name, value.data().bitcast[DType.invalid](), t
        )

    fn f32_attr(
        inout self, name: StringRef, t: TypePtr, fileName: StringRef
    ) raises -> AttrPtr:
        return self.builder.tensor_attr_from_file(self.m, name, fileName, t)

    fn attrs(inout self, *attrs: AttrPtr) raises -> AttrMap:
        let l = VariadicList(attrs)
        var m = self.builder.new_attr_map()
        for i in range(len(l)):
            var a = l[i]
            self.builder.add_attr(m, a)
        return AttrMap(m, len(l))

    # TODO: Support TensorSpec overload.

    fn dyn(inout self) raises -> Int64:
        return self.builder.dynamic_dim(self.m)

    fn i32(inout self, *dims: Int64) raises -> TypePtr:
        let l = VariadicList(dims)
        var shape = DynamicVector[Int64]()
        for i in range(len(l)):
            shape.push_back(l[i])
        return self.builder.i32(self.m, shape.data, shape.__len__())

    fn i64(inout self, *dims: Int64) raises -> TypePtr:
        let l = VariadicList(dims)
        var shape = DynamicVector[Int64]()
        for i in range(len(l)):
            shape.push_back(l[i])
        return self.builder.i64(self.m, shape.data, shape.__len__())

    fn f32(inout self, *dims: Int64) raises -> TypePtr:
        let l = VariadicList(dims)
        var shape = DynamicVector[Int64]()
        for i in range(len(l)):
            shape.push_back(l[i])
        return self.builder.f32(self.m, shape.data, shape.__len__())

    fn bool(inout self, *dims: Int64) raises -> TypePtr:
        let l = VariadicList(dims)
        var shape = DynamicVector[Int64]()
        for i in range(len(l)):
            shape.push_back(l[i])
        return self.builder.bool(self.m, shape.data, shape.__len__())

    fn types(inout self, *types: TypePtr) raises -> Arity:
        let l = VariadicList(types)
        var a = self.builder.new_arity()
        for i in range(len(l)):
            var t = l[i]
            self.builder.add_type(a, t)
        return Arity(a, len(l))

    fn graph(
        inout self,
        name: StringRef,
        in_types: Arity,
        out_types: Arity,
    ) raises -> Graph:
        var unknown = self.builder.unknown_loc(self.m)
        let g = self.builder.new_graph(
            self.m, unknown, name, in_types.a, out_types.a
        )
        return Graph(g, self.m, self.builder)
