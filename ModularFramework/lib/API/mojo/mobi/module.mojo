# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import Tensor, TensorSpec, TensorShape

from .attr import AttrMap
from .type import Arity
from .mobicc import AttrPtr, LocPtr, ModulePtr, TypePtr
from .symbol import Symbol
from .type import MOType


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
        return Self(mobicc.module_new())

    fn __str__(self) -> String:
        return mobicc.module_to_string(self.m)

    fn loc_new_unknown(self) -> LocPtr:
        return mobicc.loc_new_unknown(self.m)

    fn i32_attr(self, name: StringRef, t: TypePtr, *elements: Int32) -> AttrPtr:
        return self.i32_attr(name, t, elements)

    fn i32_attr(
        self, name: StringRef, t: TypePtr, elements: VariadicList[Int32]
    ) -> AttrPtr:
        var data = DynamicVector[Int32]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return mobicc.attr_new_tensor(self.m, name, data, t, False)

    fn i64_attr(self, name: StringRef, t: TypePtr, *elements: Int64) -> AttrPtr:
        return self.i64_attr(name, t, elements)

    fn i64_attr(
        self, name: StringRef, t: TypePtr, elements: VariadicList[Int64]
    ) -> AttrPtr:
        var data = DynamicVector[Int64]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return self.i64_attr(name, t, data)

    fn i64_attr(self, name: StringRef, t: TypePtr, element: Int) -> AttrPtr:
        # This special overload prevents the DynamicVector overload from
        # stealing precedence due to unpredictable resolution order around
        # implicit casting.
        return self.i64_attr(name, t, Int64(element))

    fn i64_attr(
        self, name: StringRef, t: TypePtr, elements: DynamicVector[Int64]
    ) -> AttrPtr:
        return mobicc.attr_new_tensor(self.m, name, elements, t, False)

    fn f32_attr(
        self, name: StringRef, t: TypePtr, *elements: Float32
    ) -> AttrPtr:
        return self.f32_attr(name, t, elements)

    fn f32_attr(
        self, name: StringRef, t: TypePtr, elements: VariadicList[Float32]
    ) -> AttrPtr:
        var data = DynamicVector[Float32]()
        for i in range(len(elements)):
            data.push_back(elements[i])
        return mobicc.attr_new_tensor(self.m, name, data, t, False)

    fn f32_attr(
        self, name: StringRef, t: TypePtr, file_name: StringRef
    ) -> AttrPtr:
        return mobicc.attr_new_tensor_from_file(self.m, name, file_name, t)

    fn attr_new_string(self, name: StringRef, value: StringRef) -> AttrPtr:
        return mobicc.attr_new_string(self.m, name, value)

    fn tensor_attr[
        dtype: DType
    ](self, name: StringRef, values: DynamicVector[SIMD[dtype, 1]]) -> AttrPtr:
        let t = MOTensor(dtype, len(values)).to_mlir(self)
        return mobicc.attr_new_tensor(
            self.m,
            name,
            values,
            t,
            is_owned=False,
        )

    fn tensor_attr[
        dtype: DType
    ](self, name: StringRef, owned value: Tensor[dtype]) -> AttrPtr:
        # Intentionally takes ownership of `value` to avoid extra copies.
        let t = MOTensor(value.spec()).to_mlir(self)
        return mobicc.attr_new_tensor(
            self.m,
            name,
            value._steal_ptr().bitcast[DType.invalid](),
            t,
            is_owned=True,
        )

    fn attrs(self, *attrs: AttrPtr) -> AttrMap:
        var m = mobicc.attrmap_new()
        for i in range(len(attrs)):
            mobicc.attrmap_add_attr(m, attrs[i])
        return AttrMap(m, len(attrs))

    fn dyn(self) -> Int64:
        return mobicc.dim_type_new_dynamic()

    fn i32(self, *dims: Int64) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.si(self.m, 32, shape, len(shape))

    fn i32(self, dims: TensorShape) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.si(self.m, 32, shape, len(shape))

    fn i64(self, *dims: Int64) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.si(self.m, 64, shape, len(shape))

    fn i64(self, dims: TensorShape) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.si(self.m, 64, shape, len(shape))

    fn f32(self, *dims: Int64) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.f32(self.m, shape, len(shape))

    fn f32(self, dims: TensorShape) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.f32(self.m, shape, len(shape))

    fn bool(self, *dims: Int64) -> TypePtr:
        let shape = _dynamic_vector(dims)
        return mobicc.bool(self.m, shape, len(shape))

    fn graph(
        inout self,
        name: StringRef,
        in_types: Arity,
        out_types: Arity,
    ) -> Graph:
        let unknown = mobicc.loc_new_unknown(self.m)
        let g = mobicc.graph_new(self.m, unknown, name, in_types.a, out_types.a)
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
