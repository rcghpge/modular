# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from .module import Module
from .mobicc import ArityPtr, TypePtr


# TODO: Don't use magic value, return a proper type.
fn dyn() -> Int64:
    return mobicc.dynamic_dim()


trait MOType:
    fn to_mlir(self, m: Module) -> TypePtr:
        ...

    fn to_string(self, m: Module) -> String:
        ...


@value
struct ElementType(MOType):
    var dtype: DType

    fn to_mlir(self, m: Module) -> TypePtr:
        return mobicc.new_dtype(m.m, self.dtype)

    fn to_string(self, m: Module) -> String:
        return mobicc.type_to_string(self.to_mlir(m))


@value
struct MOTensor(MOType):
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

    fn __init__(inout self, dtype: ElementType, dim: Int):
        # This special overload prevents the DynamicVector overload from
        # stealing precedence due to unpredictable resolution order around
        # implicit casting.
        self.__init__(dtype, Int64(dim))

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

    fn rank(self) -> Int:
        return len(self.dims)

    # TODO: Add shape arithmetics here, or in a separate Shape type.

    fn dim(self, pos: Int) raises -> Int64:
        if not self.ranked:
            raise "Cannot get dim of unranked type"
        var i = pos
        if i < 0:
            i = i + self.rank()
        return self.dims[i]

    fn num_elements(self) -> Int64:
        var n: Int64 = 1
        for i in range(self.rank()):
            if self.dims[i] == dyn():
                return dyn()
            n *= self.dims[i]
        return n

    fn is_static(self) -> Bool:
        if not self.ranked:
            return False
        for i in range(self.rank()):
            if self.dims[i] == dyn():
                return False
        return True

    fn __eq__(self, other: MOTensor) -> Bool:
        if self.dtype.dtype != other.dtype.dtype:
            return False
        if self.ranked != other.ranked:
            return False
        if self.ranked:
            if self.rank() != other.rank():
                return False
            for i in range(self.rank()):
                if self.dims[i] != other.dims[i]:
                    return False
        return True


@value
struct Arity:
    var a: ArityPtr

    fn __init__(inout self, *types: TypePtr):
        self.a = mobicc.new_arity()
        for t in types:
            mobicc.add_type(self.a, t)

    # TODO: Add MOType constructors once we have a way to store trait pointers.

    fn __len__(self) -> Int:
        return mobicc.arity_size(self.a)

    fn to_mlir(self, m: Module) -> ArityPtr:
        return self.a
