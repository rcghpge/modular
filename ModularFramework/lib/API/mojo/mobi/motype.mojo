# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from .module import Module
from .mobicc import TypePtr


# TODO: Don't use magic value, return a proper type.
fn dyn() -> Int64:
    return mobicc.dynamic_dim()


trait Type:
    fn to_mlir(self, m: Module) -> TypePtr:
        ...

    fn to_string(self, m: Module) -> String:
        ...


@value
struct ElementType(Type):
    var dtype: DType

    fn to_mlir(self, m: Module) -> TypePtr:
        return mobicc.new_dtype(m.m, self.dtype)

    fn to_string(self, m: Module) -> String:
        return mobicc.type_to_string(self.to_mlir(m))


@value
struct MOTensor(Type):
    var dtype: ElementType
    # TODO: This is insufficient. We need a vector of types.
    # To be able to wrap the shape and dims as a type, we need a wrapper over
    # MLIR TypedAttrs first, because MO stores shapes and dims as attributes
    # rather than types.
    var dims: DynamicVector[Int64]
    var ranked: Bool

    fn __init__(inout self, dtype: ElementType, *dims: Int64):
        self.dtype = dtype
        self.dims = DynamicVector[Int64]()
        for d in dims:
            self.dims.append(d)
        self.ranked = True

    fn __init__(inout self, dtype: ElementType, dims: VariadicList[Int64]):
        self.dtype = dtype
        self.dims = DynamicVector[Int64]()
        for d in dims:
            self.dims.append(d)
        self.ranked = True

    fn __init__(inout self, dtype: ElementType):
        self.dtype = dtype
        self.dims = DynamicVector[Int64]()
        self.ranked = True

    fn __init__(inout self, dtype: ElementType, ranked: Bool):
        self.dtype = dtype
        self.dims = DynamicVector[Int64]()
        self.ranked = ranked

    fn __init__(inout self, dtype: ElementType, dims: DynamicVector[Int64]):
        self.dtype = dtype
        self.dims = dims
        self.ranked = True

    fn to_mlir(self, m: Module) -> TypePtr:
        return mobicc.new_tensor_type(
            m.m, self.dtype.to_mlir(m), self.dims, self.ranked
        )

    fn to_string(self, m: Module) -> String:
        return mobicc.type_to_string(self.to_mlir(m))
