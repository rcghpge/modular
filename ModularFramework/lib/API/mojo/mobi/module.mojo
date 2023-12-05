# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import Tensor, TensorSpec, TensorShape

# FIXME(#26590): Migrate to DynamicVector.
from collections.vector import _OldDynamicVector as DynamicVector

from .misc import Arity, AttrMap, Symbol
from .mobicc import (
    AttrPtr,
    AttrMapPtr,
    LocPtr,
    ModulePtr,
    SymbolPtr,
    TypePtr,
    GraphPtr,
)
from .type import Type


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

    fn unknown_loc(self) -> LocPtr:
        return mobicc.unknown_loc(self.m)

    fn i32_attr(self, name: StringRef, t: TypePtr, *elements: Int32) -> AttrPtr:
        return self.i32_attr(name, t, elements)

    fn i32_attr(
        self, name: StringRef, t: TypePtr, elements: VariadicList[Int32]
    ) -> AttrPtr:
        var data = DynamicVector[Int32]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return mobicc.tensor_attr(
            self.m, name, data.data.bitcast[AnyRegType](), t, is_owned=False
        )

    fn i64_attr(self, name: StringRef, t: TypePtr, *elements: Int64) -> AttrPtr:
        return self.i64_attr(name, t, elements)

    fn i64_attr(
        self, name: StringRef, t: TypePtr, elements: VariadicList[Int64]
    ) -> AttrPtr:
        var data = DynamicVector[Int64]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return mobicc.tensor_attr(
            self.m, name, data.data.bitcast[AnyRegType](), t, is_owned=False
        )

    fn f32_attr(
        self, name: StringRef, t: TypePtr, *elements: Float32
    ) -> AttrPtr:
        var data = DynamicVector[Float32]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return mobicc.tensor_attr(
            self.m, name, data.data.bitcast[AnyRegType](), t, is_owned=False
        )

    fn f32_attr(
        self, name: StringRef, t: TypePtr, file_name: StringRef
    ) -> AttrPtr:
        return mobicc.tensor_attr_from_file(self.m, name, file_name, t)

    fn string_attr(self, name: StringRef, value: StringRef) -> AttrPtr:
        return mobicc.string_attr(self.m, name, value)

    fn tensor_attr[
        dtype: DType
    ](self, name: StringRef, t: TypePtr, owned value: Tensor[dtype]) -> AttrPtr:
        # Intentionally takes ownership of `value` to avoid extra copies.
        return mobicc.tensor_attr(
            self.m,
            name,
            value._steal_ptr().bitcast[DType.invalid](),
            t,
            is_owned=True,
        )

    fn type_of[dtype: DType](self, t: Tensor[dtype]) raises -> TypePtr:
        return mobicc.type_from_spec(self.m, t.spec())

    fn attrs(self, *attrs: AttrPtr) -> AttrMap:
        var m = mobicc.new_attr_map()
        for i in range(len(attrs)):
            mobicc.add_attr(m, attrs[i])
        return AttrMap(m, len(attrs))

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

    fn arg_reduction_type(self, input: TypePtr, axis: Int64) -> Arity:
        var a = mobicc.new_arity()
        mobicc.add_type(a, mobicc.arg_reduction_type(self.m, input, axis))
        return Arity(a, 1)

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

    fn verify(self) raises:
        if not mobicc.module_verify(self.m):
            raise "Module did not verify"

    fn save_to_file(self, path: Path) raises:
        # TODO: Add an overload that takes FileHandle.
        if not mobicc.module_to_bytecode(self.m, path.__str__()):
            raise "Error writing to file"

    fn save_readable_ir_to_file(self, path: Path) raises:
        # TODO: Add an overload that takes FileHandle.
        with open(path.__str__(), "w") as f:
            f.write(self.__str__())
