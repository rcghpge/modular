# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap
from .mobicc import AttrPtr, AttrMapPtr, ModulePtr, Builder, TypePtr, GraphPtr
from tensor import Tensor, TensorSpec, TensorShape

# TODO: DynamicVectors all need a _del_old() call.


fn _dynamic_vector(shape: TensorShape) -> DynamicVector[Int64]:
    var v = DynamicVector[Int64]()
    for i in range(shape.rank()):
        v.push_back(shape[i])
    return v


fn _dynamic_vector(shape: VariadicList[Int64]) -> DynamicVector[Int64]:
    var v = DynamicVector[Int64]()
    for i in range(len(shape)):
        v.push_back(shape[i])
    return v


struct Module:
    var m: ModulePtr
    var builder: Builder

    fn __init__(inout self) raises:
        self.builder = Builder()
        self.m = self.builder.new_module()

    fn __init__(inout self, borrowed m: ModulePtr, borrowed builder: Builder):
        self.m = m
        self.builder = builder

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
        var data = DynamicVector[Int32]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return self.builder.tensor_attr(
            self.m, name, data.data.bitcast[AnyType](), t
        )

    fn i64_attr(
        inout self, name: StringRef, t: TypePtr, *elements: Int64
    ) raises -> AttrPtr:
        var data = DynamicVector[Int64]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return self.builder.tensor_attr(
            self.m, name, data.data.bitcast[AnyType](), t
        )

    fn f32_attr(
        inout self, name: StringRef, t: TypePtr, *elements: Float32
    ) raises -> AttrPtr:
        var data = DynamicVector[Float32]()
        for i in range(len(elements)):
            data.push_back(elements[i])
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

    fn tensor_attr[
        dtype: DType
    ](
        inout self, name: StringRef, t: TypePtr, value: Tensor[dtype]
    ) raises -> AttrPtr:
        return self.builder.tensor_attr(
            self.m, name, value.data().bitcast[DType.invalid](), t
        )

    fn type_of[dtype: DType](inout self, t: Tensor[dtype]) raises -> TypePtr:
        if dtype == DType.float32:
            return self.f32(t.shape())
        elif dtype == DType.uint32:
            return self.i32(t.shape())
        elif dtype == DType.uint32:
            return self.i64(t.shape())
        else:
            raise Error("not implemented")
        return self.f32()

    fn attrs(inout self, *attrs: AttrPtr) raises -> AttrMap:
        var m = self.builder.new_attr_map()
        for i in range(len(attrs)):
            var a = attrs[i]
            self.builder.add_attr(m, a)
        return AttrMap(m, len(attrs))

    # TODO: Support TensorSpec overload.

    fn dyn(inout self) raises -> Int64:
        return self.builder.dynamic_dim(self.m)

    fn i32(inout self, *dims: Int64) raises -> TypePtr:
        let shape = _dynamic_vector(dims)
        return self.builder.i32(self.m, shape.data, len(shape))

    fn i32(inout self, dims: TensorShape) raises -> TypePtr:
        let shape = _dynamic_vector(dims)
        return self.builder.i32(self.m, shape.data, len(shape))

    fn i64(inout self, *dims: Int64) raises -> TypePtr:
        let shape = _dynamic_vector(dims)
        return self.builder.i64(self.m, shape.data, len(shape))

    fn i64(inout self, dims: TensorShape) raises -> TypePtr:
        let shape = _dynamic_vector(dims)
        return self.builder.i64(self.m, shape.data, len(shape))

    fn f32(inout self, *dims: Int64) raises -> TypePtr:
        let shape = _dynamic_vector(dims)
        return self.builder.f32(self.m, shape.data, len(shape))

    fn f32(inout self, dims: TensorShape) raises -> TypePtr:
        let shape = _dynamic_vector(dims)
        return self.builder.f32(self.m, shape.data, len(shape))

    fn bool(inout self, *dims: Int64) raises -> TypePtr:
        let shape = _dynamic_vector(dims)
        return self.builder.bool(self.m, shape.data, len(shape))

    fn types(inout self, *types: TypePtr) raises -> Arity:
        var a = self.builder.new_arity()
        for i in range(len(types)):
            var t = types[i]
            self.builder.add_type(a, t)
        return Arity(a, len(types))

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
