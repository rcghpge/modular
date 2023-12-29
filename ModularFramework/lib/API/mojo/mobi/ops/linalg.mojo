# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max

from mobi.motype import ElementType, MOTensor, dyn

from tensor import TensorSpec, TensorShape


def arange(stop: Symbol) -> Symbol:
    var g = stop.graph()
    let dtype = stop.type().unwrap_tensor().dtype()
    return arange(g.constant(0, dtype), stop, g.constant(1, dtype))


def arange(start: Symbol, stop: Symbol) -> Symbol:
    var g = stop.graph()
    let dtype = stop.type().unwrap_tensor().dtype()
    return arange(start, stop, g.constant(1, dtype))


def arange(start: Int, stop: Symbol, step: Int) -> Symbol:
    # We can define this more generally, but hack it in a bit for now.
    # The more general one is a bit awkward since it should be parameterized
    # on dtype (unlike the others), and start/step should take a variant
    # of SIMD[dtype, 1] | Symbol with conditional logic depending on the variant.
    var g = stop.graph()
    let dtype = stop.type().unwrap_tensor().dtype()
    return arange(g.constant(start, dtype), stop, g.constant(step, dtype))


def arange(start: Symbol, stop: Symbol, step: Symbol) -> Symbol:
    # Use stop to get the dtype since it's supplied in all cases
    var g = stop.graph()
    let dtype = stop.type().unwrap_tensor().dtype()
    return g.op("mo.range", (start, stop, step), MOTensor(dtype, dyn()))[0]


def outer(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Outer product of two vectors. This function does not broadcast."""
    return reshape(lhs, -1, 1) * reshape(rhs, 1, -1)


def batch_matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.graph()
    let broadcast_lhs: Symbol
    let broadcast_rhs: Symbol
    broadcast_lhs, broadcast_rhs = matmul_broadcast(lhs, rhs)

    let lhs_spec = broadcast_lhs.type().unwrap_tensor().spec()
    let rhs_spec = broadcast_rhs.type().unwrap_tensor().spec()
    var dims = DynamicVector[Int64]()
    for i in range(lhs_spec.rank() - 1):
        dims.push_back(lhs_spec[i])
    dims.push_back(rhs_spec[-1])
    let out_spec = MOTensor(lhs_spec.dtype(), dims)

    return g.op("mo.batch_matmul", (broadcast_lhs, broadcast_rhs), out_spec)[0]


def matmul_by_matrix(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.graph()
    let lhs_spec = lhs.type().unwrap_tensor().spec()
    let rhs_spec = rhs.type().unwrap_tensor().spec()
    if rhs_spec.rank() != 2:
        raise "rhs must be a matrix"

    let reshape_shape = stack[axis=0](
        g.constant(-1),
        dim(lhs, -1),
    )
    let reshape_dims = TensorShape(g.m.dyn().to_int(), lhs_spec[-1])

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
        MOTensor(lhs_spec.dtype(), dyn(), Int64(rhs_spec[-1])),
    )[0]

    return reshape(matmul_out, final_shape, final_dims)


def matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    let rhs_spec = rhs.type().unwrap_tensor().spec()
    if rhs_spec.rank() > 2:
        return batch_matmul(lhs, rhs)
    else:
        return matmul_by_matrix(lhs, rhs)


def matmul_broadcast(lhs: Symbol, rhs: Symbol) -> (Symbol, Symbol):
    var g = lhs.graph()
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
        MOTensor(DType.int64, Int64(broadcast_rank - 2)),
    )[0]

    var lhs_final_dims = DynamicVector[Int64]()
    var rhs_final_dims = DynamicVector[Int64]()
    for _ in range(broadcast_rank - 2):
        lhs_final_dims.push_back(dyn())
        rhs_final_dims.push_back(dyn())
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
        MOTensor(lhs_spec.dtype(), lhs_final_dims),
    )[0]

    let rhs_broadcast_shape = concat[axis=0](
        broadcast_dims_shape, rhs_matrix_dims
    )

    let broadcast_rhs = g.op(
        "mo.broadcast_to",
        (rhs, rhs_broadcast_shape),
        MOTensor(rhs_spec.dtype(), rhs_final_dims),
    )[0]

    return (broadcast_lhs, broadcast_rhs)
