# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max

from .symbol import Symbol
from .graph import Graph
from .type import Type
from .misc import Tup

from tensor import Tensor, TensorShape, TensorSpec


# ===----------------------------------------------------------------------=== #
#  File local utilities
# ===----------------------------------------------------------------------=== #


fn _fill_vector[
    type: AnyRegType
](size: Int, value: type) -> DynamicVector[type]:
    var filled = DynamicVector[type]()
    for _ in range(size):
        filled.push_back(value)

    return filled


fn _dynamic_vector[T: AnyRegType](list: VariadicList[T]) -> DynamicVector[T]:
    var vec = DynamicVector[T]()
    for value in list:
        vec.push_back(value)
    return vec


def _has_dynamic_dims(shape: TensorShape) -> Bool:
    for i in range(shape.rank()):
        if shape[i] < 0:
            return True
    return False


fn _tensor[
    N: Int
](shape: TensorShape, data: StaticIntTuple[N]) -> Tensor[DType.int64]:
    let ptr = DTypePointer[DType.int64].alloc(N)
    for i in range(N):
        ptr.store(i, data[i])
    return Tensor(ptr, shape)


# ===----------------------------------------------------------------------=== #
#  Range fns
# ===----------------------------------------------------------------------=== #


def arange(stop: Symbol) -> Symbol:
    var g = stop.g
    let dtype = stop.type().unwrap_tensor().dtype()
    return arange(g.constant(0, dtype), stop, g.constant(1, dtype))


def arange(start: Symbol, stop: Symbol, step: Symbol) -> Symbol:
    # Use stop to get the dtype since it's supplied in all cases
    var g = stop.g
    let dtype = stop.type().unwrap_tensor().dtype()
    return g.op(
        "mo.range", (start, stop, step), TensorSpec(dtype, g.m.dyn().to_int())
    )[0]


# ===----------------------------------------------------------------------=== #
#  fns that take the bits from over here and put them over there
# ===----------------------------------------------------------------------=== #


def cast(v: Symbol, dtype: DType) -> Symbol:
    var g = v.g
    let out_spec = TensorSpec(dtype, v.type().unwrap_tensor().shape())
    return g.op("mo.cast", v, out_spec)[0]


def shape_of(v: Symbol) -> Symbol:
    var g = v.g
    let spec = v.type().unwrap_tensor().spec()
    return g.op("mo.shape_of", v, TensorSpec(DType.int64, spec.rank()))[0]


def concat[axis: Int](*values: Symbol) -> Symbol:
    return concat[axis](values)


def concat[axis: Int](values: VariadicList[Symbol]) -> Symbol:
    return concat[axis](_dynamic_vector(values))


def concat[axis: Int](values: DynamicVector[Symbol]) -> Symbol:
    if not len(values):
        raise "must concat at least 1 value"
    let v0 = values[0]
    var g = values[0].g

    let spec = v0.type().unwrap_tensor().spec()
    let rank = spec.rank()
    let positive_axis = axis + spec.rank() if axis < 0 else axis
    if positive_axis < 0 or positive_axis >= spec.rank():
        raise (
            "concat axis out of bounds: axis="
            + String(positive_axis)
            + ", rank="
            + String(spec.rank())
        )

    var concat_dim = 0
    let dyn = g.m.dyn().to_int()
    for i in range(len(values)):
        let vspec = values[i].type().unwrap_tensor().spec()
        if vspec.rank() != rank:
            raise "all concat values must have same rank"
        if concat_dim == dyn or vspec[axis] == dyn:
            concat_dim = dyn
        else:
            concat_dim += vspec[axis]

    var concat_args = DynamicVector[Symbol]()
    concat_args.push_back(g.constant(positive_axis))
    for i in range(len(values)):
        concat_args.push_back(values[i])

    var dims = DynamicVector[Int](rank)
    for i in range(rank):
        dims.push_back(spec.shape[i])
    dims[positive_axis] = concat_dim

    let result = g.op(
        "mo.concat",
        concat_args,
        TensorSpec(spec.dtype(), TensorShape(dims)),
    )[0]

    concat_args._del_old()

    return result


def stack[axis: Int](values: DynamicVector[Symbol]) -> Symbol:
    # axis treated like an unsqueeze index
    var unsqueezed = DynamicVector[Symbol]()
    for i in range(len(values)):
        unsqueezed.push_back(unsqueeze(values[i], axis))
    return concat[axis](unsqueezed)


def stack[axis: Int](*values: Symbol) -> Symbol:
    return stack[axis](_dynamic_vector(values))


def unsqueeze(v: Symbol, axis: Int) -> Symbol:
    var g = v.g
    let spec = v.type().unwrap_tensor().spec()
    # Negative values add an extra 1, as -1 adds a new dim at the _end_.
    if axis < 0:
        axis += spec.rank() + 1
    if axis < 0 or axis > spec.rank():
        raise (
            "unsqueeze axis out of bounds: axis="
            + String(axis)
            + ", rank="
            + String(spec.rank())
        )

    let new_shape = g.op(
        "mo.unsqueeze_shape",
        (shape_of(v), g.constant(Tensor[DType.int64](TensorShape(1), axis))),
        TensorSpec(DType.int64, spec.rank() + 1),
    )[0]

    var dims = DynamicVector[Int]()
    for i in range(spec.rank()):
        if i == axis:
            dims.push_back(1)
        dims.push_back(spec[i])
    if axis == spec.rank():
        dims.push_back(1)

    return g.op(
        "mo.reshape",
        (v, new_shape),
        TensorSpec(spec.dtype(), TensorShape(dims)),
    )[0]


def reshape(v: Symbol, shape: TensorShape) -> Symbol:
    var g = v.g

    var symbols = DynamicVector[Symbol]()
    let shape_data = DTypePointer[DType.int64].alloc(shape.rank())
    var reshape_dims = DynamicVector[Int]()

    for i in range(shape.rank()):
        let dim = shape[i]
        symbols.push_back(g.constant(dim))
        reshape_dims.push_back(g.m.dyn().to_int() if dim == -1 else dim)
        shape_data.store(i, dim)

    let shape_ = g.constant(Tensor(shape_data, TensorShape(shape.rank())))

    let dtype = v.type().unwrap_tensor().dtype()
    return g.op("mo.reshape", (v, shape_), TensorSpec(dtype, reshape_dims))[0]


def reshape(v: Symbol, *dims: Int) -> Symbol:
    return reshape(v, TensorShape(dims))


def reshape(v: Symbol, *dims: Symbol) -> Symbol:
    var g = v.g
    let rank = len(dims)
    let shape = stack[axis=0](_dynamic_vector(dims))

    var reshape_dims = DynamicVector[Int]()
    for _ in range(rank):
        reshape_dims.push_back(g.m.dyn().to_int())

    let dtype = v.type().unwrap_tensor().dtype()
    return g.op("mo.reshape", (v, shape), TensorSpec(dtype, reshape_dims))[0]


fn reshape(v: Symbol, shape: Symbol, out_shape: TensorShape) raises -> Symbol:
    var g = v.g
    let dtype = v.type().unwrap_tensor().dtype()
    let result = g.op("mo.reshape", (v, shape), TensorSpec(dtype, out_shape))[0]
    return result


fn reshape(rank: Int, v: Symbol, shape: Symbol) raises -> Symbol:
    return reshape(
        v,
        shape,
        TensorShape(_fill_vector(rank, v.g.m.dyn().to_int())),
    )


def split[
    axis: Int, N: Int, split_sizes: StaticIntTuple[N]
](x: Symbol) -> Tuple[Symbol, Symbol]:
    var g = x.g
    let sizes = g.constant(_tensor[N](TensorShape(N), split_sizes))

    let spec = x.type().unwrap_tensor().spec()
    let out_shape = _fill_vector(spec.rank(), g.m.dyn().to_int())
    let split_ty = g.m.type(TensorSpec(spec.dtype(), out_shape)).type_ptr()

    let p = g.op(
        "mo.split",
        Tup(x, sizes, g.constant(axis)),
        g.m.types(split_ty, split_ty),
    )
    return p[0], p[1]


def transpose_matrix(matrix: Symbol) -> Symbol:
    return transpose(matrix, -1, -2)


def transpose(input: Symbol, x: Int, y: Int) -> Symbol:
    var g = input.g
    let spec = input.type().unwrap_tensor().spec()
    if spec.rank() < 2:
        raise "transpose input must have rank >= 2"
    if x < 0:
        x += spec.rank()
    if y < 0:
        y += spec.rank()
    if x < 0 or x >= spec.rank() or y < 0 or y >= spec.rank():
        raise "transpose dim outside range"

    var dims = DynamicVector[Int]()
    let ptr = DTypePointer[DType.int64].alloc(spec.rank())
    for i in range(spec.rank()):
        dims.push_back(spec[i])
        ptr.store(i, i)

    dims[x] = spec[y]
    dims[y] = spec[x]
    ptr.store(x, y)
    ptr.store(y, x)

    let transpose_indices = g.constant(
        Tensor[DType.int64](ptr, TensorShape(spec.rank()))
    )

    return g.op(
        "mo.transpose",
        (input, transpose_indices),
        TensorSpec(spec.dtype(), TensorShape(dims)),
    )[0]


# ===----------------------------------------------------------------------=== #
#  Index fns
# ===----------------------------------------------------------------------=== #


def dim(v: Symbol, dim: Int) -> Symbol:
    return reshape(shape_of(v)[dim : dim + 1], TensorShape())


def index[axis: Int = 0](v: Symbol, idx: Int) -> Symbol:
    """Index a tensor at a specific dim on a single axis.
    The output is a tensor omitting the dim `axis` and returning only the
    part of the tensor at at the specified index of the `axis` dimension."""
    var g = v.g
    let spec = v.type().unwrap_tensor().spec()

    var slice_dims = DynamicVector[Int]()
    var indexed_dims = DynamicVector[Int]()
    let start = DTypePointer[DType.int64].alloc(spec.rank())
    let stop = DTypePointer[DType.int64].alloc(spec.rank())
    let step = DTypePointer[DType.int64].alloc(spec.rank())
    for i in range(spec.rank()):
        if i == axis:
            slice_dims.push_back(1)
            start.store(i, idx)
            stop.store(i, idx + 1)
        else:
            slice_dims.push_back(spec[i])
            indexed_dims.push_back(spec[i])
            start.store(i, 0)
            stop.store(i, 0x7FFFFFFFFFFFFFFF)
        step.store(i, 1)

    let slice = g.op(
        "mo.slice",
        (
            v,
            g.constant[DType.int64](Tensor(start, TensorShape(spec.rank()))),
            g.constant[DType.int64](Tensor(stop, TensorShape(spec.rank()))),
            g.constant[DType.int64](Tensor(step, TensorShape(spec.rank()))),
        ),
        TensorSpec(spec.dtype(), TensorShape(slice_dims)),
    )[0]

    return reshape(slice, TensorShape(indexed_dims))


def gather(input: Symbol, indices: Symbol, axis: Int) -> Symbol:
    var g = input.g
    let input_spec = input.type().unwrap_tensor().spec()
    let indices_spec = indices.type().unwrap_tensor().spec()
    if axis < 0:
        axis += input_spec.rank()
    if axis < 0 or axis >= input_spec.rank():
        raise (
            "gather axis out of bounds: axis="
            + String(axis)
            + ", rank="
            + String(input_spec.rank())
        )

    var dims = DynamicVector[Int]()
    for i in range(input_spec.rank()):
        if i == axis:
            for j in range(indices_spec.rank()):
                dims.push_back(indices_spec[j])
        else:
            dims.push_back(input_spec[i])

    return g.op(
        "mo.gather",
        (input, indices, g.constant(axis)),
        TensorSpec(input_spec.dtype(), TensorShape(dims)),
    )[0]


# ===----------------------------------------------------------------------=== #
#  Broadcast helpers
# ===----------------------------------------------------------------------=== #


def matmul_broadcast(lhs: Symbol, rhs: Symbol) -> (Symbol, Symbol):
    var g = lhs.g
    let lhs_spec = lhs.type().unwrap_tensor().spec()
    let rhs_spec = rhs.type().unwrap_tensor().spec()

    let lhs_rank = lhs_spec.rank()
    let rhs_rank = rhs_spec.rank()

    let broadcast_rank = max(lhs_rank, rhs_rank)
    let lhs_shape = shape_of(lhs)
    let rhs_shape = shape_of(rhs)

    let lhs_broadcast_dims = lhs_shape[: lhs_rank - 2]
    let lhs_matrix_dims = lhs_shape[lhs_rank - 2 : lhs_rank]

    let rhs_broadcast_dims = rhs_shape[: rhs_rank - 2]
    let rhs_matrix_dims = rhs_shape[rhs_rank - 2 : rhs_rank]

    let broadcast_dims_shape = g.op(
        "mo.broadcast_shape",
        (lhs_broadcast_dims, rhs_broadcast_dims),
        TensorSpec(DType.int64, broadcast_rank - 2),
    )[0]

    var lhs_final_dims = DynamicVector[Int]()
    var rhs_final_dims = DynamicVector[Int]()
    for _ in range(broadcast_rank - 2):
        lhs_final_dims.push_back(g.m.dyn().to_int())
        rhs_final_dims.push_back(g.m.dyn().to_int())
    lhs_final_dims.push_back(lhs_spec[-2])
    lhs_final_dims.push_back(lhs_spec[-1])
    rhs_final_dims.push_back(rhs_spec[-2])
    rhs_final_dims.push_back(rhs_spec[-1])

    let lhs_broadcast_shape = concat[axis=0](
        broadcast_dims_shape, lhs_matrix_dims
    )

    let broadcast_lhs = g.op(
        "mo.broadcast_to",
        (lhs, lhs_broadcast_shape),
        TensorSpec(lhs_spec.dtype(), lhs_final_dims),
    )[0]

    let rhs_broadcast_shape = concat[axis=0](
        broadcast_dims_shape, rhs_matrix_dims
    )

    let broadcast_rhs = g.op(
        "mo.broadcast_to",
        (rhs, rhs_broadcast_shape),
        TensorSpec(rhs_spec.dtype(), rhs_final_dims),
    )[0]

    return (broadcast_lhs, broadcast_rhs)


def elementwise_broadcast(lhs: Symbol, rhs: Symbol) -> (Symbol, Symbol):
    var g = lhs.g
    let lhs_spec = lhs.type().unwrap_tensor().spec()
    let rhs_spec = rhs.type().unwrap_tensor().spec()

    if lhs_spec == rhs_spec and not _has_dynamic_dims(lhs_spec.shape):
        return (lhs, rhs)

    let lhs_rank = lhs_spec.rank()
    let rhs_rank = rhs_spec.rank()

    let broadcast_rank = max(lhs_rank, rhs_rank)
    let lhs_shape = shape_of(lhs)
    let rhs_shape = shape_of(rhs)
    let broadcast_shape = g.op(
        "mo.broadcast_shape",
        (lhs_shape, rhs_shape),
        TensorSpec(DType.int64, broadcast_rank),
    )[0]

    let broadcast_dims = _fill_vector(broadcast_rank, g.m.dyn().to_int())

    let broadcast_lhs = g.op(
        "mo.broadcast_to",
        (lhs, broadcast_shape),
        TensorSpec(lhs_spec.dtype(), broadcast_dims),
    )[0]
    let broadcast_rhs = g.op(
        "mo.broadcast_to",
        (rhs, broadcast_shape),
        TensorSpec(rhs_spec.dtype(), broadcast_dims),
    )[0]
    return (broadcast_lhs, broadcast_rhs)


# ===----------------------------------------------------------------------=== #
#  Binary tensor ops
# ===----------------------------------------------------------------------=== #


def add(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.g
    let p = elementwise_broadcast(lhs, rhs)
    return g.op("mo.add", p, p.get[0, Symbol]().type())[0]


def sub(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.g
    let p = elementwise_broadcast(lhs, rhs)
    return g.op("mo.sub", p, p.get[0, Symbol]().type())[0]


def mul(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.g
    let p = elementwise_broadcast(lhs, rhs)
    return g.op("mo.mul", p, p.get[0, Symbol]().type())[0]


def div(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.g
    # Cast to satisfy `mo.div`'s SameOperandsAndResultTypes OpTrait.
    let cast_rhs = cast(rhs, lhs.type().unwrap_tensor().dtype())
    let p = elementwise_broadcast(lhs, cast_rhs)
    return g.op("mo.div", p, p.get[0, Symbol]().type())[0]


def pow(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.g
    let p = elementwise_broadcast(lhs, rhs)
    return g.op("mo.pow", p, p.get[0, Symbol]().type())[0]


# ===----------------------------------------------------------------------=== #
#  Unary tensor ops
# ===----------------------------------------------------------------------=== #


def rsqrt(v: Symbol) -> Symbol:
    var g = v.g
    return g.op(
        "mo.rsqrt",
        cast(v, DType.float32),  # rsqrt expects a float
        TensorSpec(DType.float32, v.type().unwrap_tensor().shape()),
    )[0]


def softmax(v: Symbol) -> Symbol:
    """Computes softmax on the last axis."""
    var g = v.g
    return g.op("mo.softmax", v, v.type())[0]


def silu(v: Symbol) -> Symbol:
    """See https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html ."""
    return mul(v, sigmoid(v))


def sigmoid(v: Symbol) -> Symbol:
    var g = v.g
    return g.op("mo.sigmoid", v, v.type())[0]


def sin(v: Symbol) -> Symbol:
    var g = v.g
    return g.op("mo.sin", v, v.type())[0]


def cos(v: Symbol) -> Symbol:
    var g = v.g
    return g.op("mo.cos", v, v.type())[0]


# ===----------------------------------------------------------------------=== #
#  Unary tensor reduction ops
# ===----------------------------------------------------------------------=== #


def mean(v: Symbol, dim: Int) -> Symbol:
    # Mean along axis `dim`, leaving a singleton dimension in its place.
    # Similar to v.mean(dim, keepdim=True) in torch.
    var g = v.g
    let spec = v.type().unwrap_tensor().spec()
    if dim < 0:
        dim += spec.rank()
    if dim < 0 or dim >= spec.rank():
        raise "dim out of range"

    var dims = DynamicVector[Int]()
    var reshape_dims = DynamicVector[Int]()
    for i in range(spec.rank()):
        if i == dim:
            dims.push_back(1)  # mean sets reduce dim to 1
        else:
            dims.push_back(spec[i])
            reshape_dims.push_back(spec[i])

    return g.op(
        "mo.mean",
        (v, g.constant(dim)),
        TensorSpec(spec.dtype(), TensorShape(dims)),
    )[0]


# ===----------------------------------------------------------------------=== #
#  Matmul
# ===----------------------------------------------------------------------=== #


def batch_matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.g
    let broadcast_lhs: Symbol
    let broadcast_rhs: Symbol
    broadcast_lhs, broadcast_rhs = matmul_broadcast(lhs, rhs)

    let lhs_spec = broadcast_lhs.type().unwrap_tensor().spec()
    let rhs_spec = broadcast_rhs.type().unwrap_tensor().spec()
    var dims = DynamicVector[Int]()
    for i in range(lhs_spec.rank() - 1):
        dims.push_back(lhs_spec[i])
    dims.push_back(rhs_spec[-1])
    let out_spec = TensorSpec(lhs_spec.dtype(), dims)

    return g.op("mo.batch_matmul", (broadcast_lhs, broadcast_rhs), out_spec)[0]


def matmul_by_matrix(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.g
    let lhs_spec = lhs.type().unwrap_tensor().spec()
    let rhs_spec = rhs.type().unwrap_tensor().spec()
    if rhs_spec.rank() != 2:
        raise "rhs must be a matrix"

    let reshape_shape = stack[axis=0](
        g.constant(-1),
        dim(lhs, 1),
    )
    let reshape_dims = TensorShape(g.m.dyn().to_int(), rhs_spec[-1])

    let final_shape = concat[axis=0](
        shape_of(lhs)[: lhs_spec.rank() - 1],
        reshape(dim(rhs, 1), TensorShape(1)),
    )

    var final_dims = DynamicVector[Int]()
    for i in range(lhs_spec.rank() - 1):
        final_dims.push_back(lhs_spec[i])
    final_dims.push_back(rhs_spec[-1])

    let matmul_out = g.op(
        "mo.matmul",
        (reshape(lhs, reshape_shape, reshape_dims), rhs),
        TensorSpec(lhs_spec.dtype(), g.m.dyn().to_int(), rhs_spec[-1]),
    )[0]

    return reshape(matmul_out, final_shape, final_dims)


def matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    let rhs_spec = rhs.type().unwrap_tensor().spec()
    if rhs_spec.rank() > 2:
        return batch_matmul(lhs, rhs)
    else:
        return matmul_by_matrix(lhs, rhs)
