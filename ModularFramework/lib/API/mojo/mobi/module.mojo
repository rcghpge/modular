# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .misc import Arity, AttrMap
from .mobicc import (
    AttrPtr,
    AttrMapPtr,
    LocPtr,
    ModulePtr,
    TypePtr,
    GraphPtr,
)
from .type import Type
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


@value
@register_passable
struct Module:
    var m: ModulePtr

    fn __init__() raises -> Self:
        return Self(mobicc.new_module())

    fn __init__(m: ModulePtr) -> Self:
        return Self {m: m}

    fn __str__(self) -> String:
        return mobicc.module_to_string(self.m)

    fn debug_print_execute(self, inout arg: Tensor[DType.float32]) raises:
        var rtArgs = mobicc.new_rt_tuple()
        mobicc.add_f32_tensor(rtArgs, arg)
        var rtOuts = mobicc.new_rt_tuple()
        let execCode = mobicc.execute_module(self.m, rtArgs, rtOuts)
        if execCode != 0:
            raise Error("Execution error")
        mobicc.debug_print_tensors(rtOuts)

    fn debug_print_execute(self, file_name: StringRef, spec: TensorSpec) raises:
        var rtArgs = mobicc.new_rt_tuple()
        var shape = DynamicVector[Int64]()
        for i in range(spec.rank()):
            shape.push_back(spec[i])
        mobicc.add_f32_tensor_from_file(
            rtArgs, file_name, shape.data, spec.rank()
        )
        var rtOuts = mobicc.new_rt_tuple()
        let execCode = mobicc.execute_module(self.m, rtArgs, rtOuts)
        if execCode != 0:
            raise Error("Execution error")
        mobicc.debug_print_tensors(rtOuts)

    fn unknown_loc(self) -> LocPtr:
        return mobicc.unknown_loc(self.m)

    fn i32_attr(self, name: StringRef, t: TypePtr, *elements: Int32) -> AttrPtr:
        var data = DynamicVector[Int32]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return mobicc.tensor_attr(self.m, name, data.data.bitcast[AnyType](), t)

    fn i64_attr(self, name: StringRef, t: TypePtr, *elements: Int64) -> AttrPtr:
        var data = DynamicVector[Int64]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return mobicc.tensor_attr(self.m, name, data.data.bitcast[AnyType](), t)

    fn f32_attr(
        self, name: StringRef, t: TypePtr, *elements: Float32
    ) -> AttrPtr:
        var data = DynamicVector[Float32]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return mobicc.tensor_attr(self.m, name, data.data.bitcast[AnyType](), t)

    fn f32_attr(
        self, name: StringRef, t: TypePtr, value: Tensor[DType.float32]
    ) -> AttrPtr:
        return mobicc.tensor_attr(
            self.m, name, value.data().bitcast[DType.invalid](), t
        )

    fn f32_attr(
        self, name: StringRef, t: TypePtr, file_name: StringRef
    ) -> AttrPtr:
        return mobicc.tensor_attr_from_file(self.m, name, file_name, t)

    fn tensor_attr[
        dtype: DType
    ](self, name: StringRef, t: TypePtr, value: Tensor[dtype]) -> AttrPtr:
        return mobicc.tensor_attr(
            self.m, name, value.data().bitcast[DType.invalid](), t
        )

    fn type_of[dtype: DType](self, t: Tensor[dtype]) raises -> TypePtr:
        if dtype == DType.float32:
            return self.f32(t.shape())
        elif dtype == DType.uint32:
            return self.i32(t.shape())
        elif dtype == DType.uint32:
            return self.i64(t.shape())
        raise "not implemented"

    fn attrs(self, *attrs: AttrPtr) -> AttrMap:
        var m = mobicc.new_attr_map()
        for i in range(len(attrs)):
            mobicc.add_attr(m, attrs[i])
        return AttrMap(m, len(attrs))

    # TODO: Support TensorSpec overload.

    fn dyn(self) -> Int64:
        return mobicc.dynamic_dim(self.m)

    fn i32(self, *dims: Int64) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.si(self.m, 32, shape.data, len(shape))

    fn i32(self, dims: TensorShape) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.si(self.m, 32, shape.data, len(shape))

    fn i64(self, *dims: Int64) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.si(self.m, 64, shape.data, len(shape))

    fn i64(self, dims: TensorShape) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.si(self.m, 64, shape.data, len(shape))

    fn f32(self, *dims: Int64) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.f32(self.m, shape.data, len(shape))

    fn f32(self, dims: TensorShape) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.f32(self.m, shape.data, len(shape))

    fn bool(self, *dims: Int64) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.bool(self.m, shape.data, len(shape))

    fn types(self, *types: TypePtr) -> Arity:
        var a = mobicc.new_arity()
        for i in range(len(types)):
            mobicc.add_type(a, types[i])
        return Arity(a, len(types))

    fn type(inout self, spec: TensorSpec) -> Type:
        return Type(mobicc.type_from_spec(self.m, spec))

    fn graph(
        inout self,
        name: StringRef,
        in_types: Arity,
        out_types: Arity,
    ) -> Graph:
        let unknown = mobicc.unknown_loc(self.m)
        let g = mobicc.new_graph(self.m, unknown, name, in_types.a, out_types.a)
        return Graph(g, self)
