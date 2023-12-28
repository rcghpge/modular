# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorSpec, TensorShape


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


fn arg_max(v: Symbol, axis: Int64) raises -> Symbol:
    var g = v.g
    let m = g.m
    # TODO: Replace with Mojo logic once we have properly shadowed types.
    return g.op(
        "mo.arg_max",
        g.values(v, g.constant(axis)),
        m.arg_reduction_type(v.type().type_ptr(), axis),
    )[0]
