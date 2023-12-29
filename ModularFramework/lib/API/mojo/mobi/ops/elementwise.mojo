# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max

from mobi.motype import ElementType, MOTensor, dyn

from tensor import TensorSpec, TensorShape


def _has_dynamic_dims(shape: TensorShape) -> Bool:
    for i in range(shape.rank()):
        if shape[i] < 0:
            return True
    return False


def rsqrt(v: Symbol) -> Symbol:
    let f32_v = cast(v, DType.float32)  # TODO: add missing rsqrt coverage.
    return v.g.op("mo.rsqrt", f32_v, f32_v.tensor_type())[0]


def softmax(v: Symbol) -> Symbol:
    var g = v.g
    return g.op("mo.softmax", v, v.type())[0]


def silu(v: Symbol) -> Symbol:
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
        MOTensor(DType.int64, Int64(broadcast_rank)),
    )[0]

    # Implement basic numpy-style broadcast, but with dynamic dims allowed
    # 1. the smaller shape is filled with 1s from the left
    # 2. if either dim is 1 it is promoted to the other value
    # 3. if either dim is dynamic, we can't predict, so output dynamic
    var broadcast_dims = DynamicVector[Int64]()
    let larger = lhs_spec if lhs_spec.rank() > rhs_spec.rank() else rhs_spec
    let smaller = rhs_spec if lhs_spec.rank() > rhs_spec.rank() else lhs_spec
    let offset = larger.rank() - smaller.rank()
    for i in range(offset):
        broadcast_dims.push_back(larger[i])
    for i in range(offset, broadcast_rank):
        let d1 = larger[i]
        let d2 = smaller[i - offset]
        broadcast_dims.push_back(
            d1 if d1 == d2 or d2 == 1 else (d2 if d1 == 1 else dyn())
        )

    let broadcast_lhs = g.op(
        "mo.broadcast_to",
        (lhs, broadcast_shape),
        MOTensor(lhs_spec.dtype(), broadcast_dims),
    )[0]
    let broadcast_rhs = g.op(
        "mo.broadcast_to",
        (rhs, broadcast_shape),
        MOTensor(rhs_spec.dtype(), broadcast_dims),
    )[0]
    return (broadcast_lhs, broadcast_rhs)
