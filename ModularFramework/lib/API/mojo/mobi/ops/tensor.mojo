# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorSpec, TensorShape

from mobi.symbol import Tup
from mobi.motype import *


# TODO: Add checks or extend to unranked support, where static shapes assumed.


fn _fill_vector[
    type: CollectionElement
](size: Int, value: type) -> DynamicVector[type]:
    var filled = DynamicVector[type]()
    for _ in range(size):
        filled.push_back(value)

    return filled


fn _tensor[
    N: Int
](shape: TensorShape, data: StaticIntTuple[N]) -> Tensor[DType.int64]:
    let ptr = DTypePointer[DType.int64].alloc(N)
    for i in range(N):
        ptr.store(i, data[i])
    return Tensor(ptr, shape)


def dim(v: Symbol, dim: Int) -> Symbol:
    if dim < 0:
        dim += v.tensor_type().rank()
    return reshape(shape_of(v)[dim : dim + 1], TensorShape())


def index[axis: Int = 0](v: Symbol, idx: Int) -> Symbol:
    """Index a tensor at a specific dim on a single axis.
    The output is a tensor omitting the dim `axis` and returning only the
    part of the tensor at at the specified index of the `axis` dimension."""
    var g = v.graph()
    let v_type = v.tensor_type()
    let rank = v_type.rank()

    var slice_dims = DynamicVector[Int64]()
    var start = DynamicVector[Int64]()
    var stop = DynamicVector[Int64]()
    var step = DynamicVector[Int64]()
    for i in range(rank):
        if i == axis:
            slice_dims.append(1)
            start.append(idx)
            stop.append(idx + 1)
        else:
            slice_dims.append(v_type.dims[i])
            start.append(0)
            stop.append(slice(None, None, None).end)
        step.append(1)

    let slice = g.op(
        "mo.slice",
        (
            v,
            g.i64_constant(g.m.i64(rank), start),
            g.i64_constant(g.m.i64(rank), stop),
            g.i64_constant(g.m.i64(rank), step),
        ),
        MOTensor(v_type.dtype, slice_dims),
    )

    return squeeze(slice, axis)


def gather(input: Symbol, indices: Symbol, axis: Int) -> Symbol:
    var g = input.graph()
    let input_type = input.tensor_type()
    let indices_type = indices.tensor_type()

    if axis < 0:
        axis += input_type.rank()
    if axis < 0 or axis >= input_type.rank():
        raise (
            "gather axis out of bounds: axis="
            + String(axis)
            + ", rank="
            + String(input_type.rank())
        )

    var dims = DynamicVector[Int64]()
    for i in range(input_type.rank()):
        if i == axis:
            for j in range(indices_type.rank()):
                dims.push_back(indices_type.dims[j])
        else:
            dims.push_back(input_type.dims[i])

    return g.op(
        "mo.gather",
        (input, indices, g.constant(axis)),
        MOTensor(input_type.dtype, dims),
    )


@value
struct SliceSymbol:
    var start: Symbol
    var stop: Symbol
    var step: Symbol

    fn __init__(inout self, stop: Symbol) raises:
        var g = stop.graph()
        self.__init__(g.constant(0), stop)

    fn __init__(inout self, start: Symbol, stop: Symbol) raises:
        var g = stop.graph()
        self.__init__(start, stop, g.constant(1))

    fn __init__(inout self, start_stop: (Symbol, Symbol)) raises:
        self.__init__(start_stop.get[0, Symbol](), start_stop.get[1, Symbol]())

    fn __init__(inout self, start: Symbol, stop: Symbol, step: Symbol):
        self.start = start
        self.stop = stop
        self.step = step

    fn __init__(inout self, start_stop_step: (Symbol, Symbol, Symbol)):
        self.__init__(
            start_stop_step.get[0, Symbol](),
            start_stop_step.get[1, Symbol](),
            start_stop_step.get[2, Symbol](),
        )


def slice_(input: Symbol, borrowed *slices: SliceSymbol) -> Symbol:
    return slice_(input, slices)


def slice_(input: Symbol, slices: VariadicListMem[SliceSymbol]) -> Symbol:
    let g = input.graph()
    let input_type = input.tensor_type()

    var dims = DynamicVector[Int]()
    for slice in slices:
        dims.push_back(dyn().to_int())
    for i in range(len(slices), input_type.rank()):
        dims.push_back(input_type.dims[i].to_int())

    return slice_(input, dims, slices)


def slice_(
    input: Symbol, shape: TensorShape, borrowed *slices: SliceSymbol
) -> Symbol:
    return slice_(input, shape, slices)


def slice_(
    input: Symbol, shape: TensorShape, slices: VariadicListMem[SliceSymbol]
) -> Symbol:
    var g = input.graph()
    var starts = DynamicVector[Symbol]()
    var stops = DynamicVector[Symbol]()
    var steps = DynamicVector[Symbol]()

    let input_t = input.tensor_type()

    fn ensure_constant(value: Symbol) raises -> Symbol:
        let input_t = value.tensor_type()
        if input_t.num_elements() != 1:
            raise "slice indicies must be singular"
        return value if input_t.rank() == 0 else reshape(value, TensorShape())

    let slice_max_value = slice(None, None, None).end
    for dim in range(len(slices)):
        let slice: SliceSymbol = __get_address_as_lvalue(slices[dim])
        starts.push_back(ensure_constant(slice.start))
        stops.push_back(ensure_constant(slice.stop))
        steps.push_back(ensure_constant(slice.step))
    for dim in range(len(slices), input_t.rank()):
        starts.push_back(g.constant(0))
        stops.push_back(g.constant(slice_max_value))
        steps.push_back(g.constant(1))

    let start = stack[axis=0](starts)
    let stop = stack[axis=0](stops)
    let step = stack[axis=0](steps)

    var dims = DynamicVector[Int64]()
    for i in range(shape.rank()):
        dims.append(shape[i])

    return g.op(
        "mo.slice", (input, start, stop, step), MOTensor(input_t.dtype, dims)
    )


def cast(v: Symbol, dtype: ElementType) -> Symbol:
    var g = v.graph()
    var t = v.tensor_type()
    t.dtype = dtype
    return g.op("mo.cast", v, t)


def shape_of(v: Symbol) -> Symbol:
    var g = v.graph()
    return g.op("mo.shape_of", v, MOTensor(DType.int64, v.tensor_type().rank()))


def concat[axis: Int](*values: Symbol) -> Symbol:
    return concat[axis](values)


def concat[axis: Int](values: VariadicList[Symbol]) -> Symbol:
    var vec = DynamicVector[Symbol](len(values))
    for value in values:
        vec.append(value)
    return concat[axis](vec)


def concat[axis: Int](values: DynamicVector[Symbol]) -> Symbol:
    if not len(values):
        raise "must concat at least 1 value"
    let v0 = values[0]
    var g = values[0].graph()

    let v0_type = v0.tensor_type()
    let rank = v0_type.rank()
    let norm_axis = axis + v0_type.rank() if axis < 0 else axis
    if norm_axis < 0 or norm_axis >= v0_type.rank():
        raise (
            "concat axis out of bounds: axis="
            + String(norm_axis)
            + ", rank="
            + String(v0_type.rank())
        )

    var concat_dim: Int64 = 0
    for i in range(len(values)):
        let v_type = values[i].tensor_type()
        if v_type.rank() != rank:
            raise (
                "all concat values must have same rank: rank[0]="
                + String(rank)
                + ", rank["
                + String(i)
                + "]="
                + String(v_type.rank())
            )
        if concat_dim == dyn() or v_type.dims[norm_axis] == dyn():
            concat_dim = dyn()
        else:
            concat_dim += v_type.dims[norm_axis]

    var concat_args = DynamicVector[Symbol]()
    concat_args.push_back(g.constant(norm_axis))
    for i in range(len(values)):
        concat_args.push_back(values[i])

    var dims = DynamicVector[Int64](rank)
    for i in range(rank):
        dims.push_back(v0_type.dims[i])
    dims[norm_axis] = concat_dim

    return g.op("mo.concat", concat_args, MOTensor(v0_type.dtype, dims))


def stack[axis: Int](values: DynamicVector[Symbol]) -> Symbol:
    # axis treated like an unsqueeze index
    var unsqueezed = DynamicVector[Symbol]()
    for i in range(len(values)):
        unsqueezed.push_back(unsqueeze(values[i], axis))
    return concat[axis](unsqueezed)


def stack[axis: Int](*values: Symbol) -> Symbol:
    var vec = DynamicVector[Symbol](len(values))
    for value in values:
        vec.append(value)
    return stack[axis](vec)


def squeeze(v: Symbol, axis: Int) -> Symbol:
    var g = v.graph()
    let v_type = v.tensor_type()
    let rank = v_type.rank()
    # TODO: This should be a mo.select
    if axis < 0:
        axis += rank

    let new_shape = g.op(
        "mo.squeeze_shape",
        (shape_of(v), g.i64_constant(g.m.i64(1), axis)),
        MOTensor(DType.int64, rank - 1),
    )

    var squeezed_dims = DynamicVector[Int64]()
    for i in range(rank):
        if i != axis:
            squeezed_dims.push_back(v_type.dims[i])

    return g.op(
        "mo.reshape", (v, new_shape), MOTensor(v_type.dtype, squeezed_dims)
    )


def unsqueeze(v: Symbol, axis: Int) -> Symbol:
    var g = v.graph()
    let v_type = v.tensor_type()
    # Negative values add an extra 1, as -1 adds a new dim at the _end_.
    if axis < 0:
        axis += v_type.rank() + 1
    if axis < 0 or axis > v_type.rank():
        raise (
            "unsqueeze axis out of bounds: axis="
            + String(axis)
            + ", rank="
            + String(v_type.rank())
        )

    # TODO: Bug - passing v_type.rank() + 1 into a variadic Int64 corrupts it.
    let new_shape = g.op(
        "mo.unsqueeze_shape",
        (shape_of(v), g.constant(Tensor[DType.int64](TensorShape(1), axis))),
        MOTensor(DType.int64, v_type.rank() + 1),
    )

    var dims = DynamicVector[Int64]()
    for i in range(v_type.rank()):
        if i == axis:
            dims.push_back(1)
        dims.push_back(v_type.dims[i])
    if axis == v_type.rank():
        dims.push_back(1)

    return g.op("mo.reshape", (v, new_shape), MOTensor(v_type.dtype, dims))


def reshape(v: Symbol, shape: TensorShape) -> Symbol:
    var g = v.graph()

    var symbols = DynamicVector[Symbol]()
    let shape_data = DTypePointer[DType.int64].alloc(shape.rank())
    var reshape_dims = DynamicVector[Int64]()

    for i in range(shape.rank()):
        let dim = shape[i]
        symbols.push_back(g.constant(dim))
        reshape_dims.push_back(dyn() if dim == -1 else dim)
        shape_data.store(i, dim)

    let shape_ = g.constant(Tensor(shape_data, TensorShape(shape.rank())))

    return g.op(
        "mo.reshape", (v, shape_), MOTensor(v.tensor_type().dtype, reshape_dims)
    )


def reshape(v: Symbol, *dims: Int) -> Symbol:
    return reshape(v, TensorShape(dims))


def reshape(v: Symbol, *dims: Symbol) -> Symbol:
    var g = v.graph()
    for dim in dims:
        if dim.tensor_type().rank() > 1:
            raise "reshape: got multi-dimensional shape"

    if len(dims) == 1:
        # Special case: we're maybe passing in a dynamic reshape shape
        let dim = dims[0]
        let dim_t = dim.tensor_type()
        if dim_t.dims[0] > 1:  # only specialize for statically known rank > 1
            return reshape(dim_t.dims[0], v, dim)

    let rank = len(dims)
    var vec = DynamicVector[Symbol](rank)
    for dim in dims:
        vec.append(dim)
    let shape = stack[axis=0](vec)

    var reshape_dims = DynamicVector[Int64]()
    for _ in range(rank):
        reshape_dims.push_back(dyn())

    let dtype = v.tensor_type().dtype
    return g.op("mo.reshape", (v, shape), MOTensor(dtype, reshape_dims))


fn reshape(v: Symbol, shape: Symbol, out_shape: TensorShape) raises -> Symbol:
    var g = v.graph()
    var reshape_dims = DynamicVector[Int64]()
    for i in range(out_shape.rank()):
        reshape_dims.push_back(out_shape[i])
    let result = g.op(
        "mo.reshape",
        (v, shape),
        MOTensor(v.tensor_type().dtype.dtype, reshape_dims),
    )
    return result


fn reshape(
    v: Symbol, shape: Symbol, out_shape: DynamicVector[Int64]
) raises -> Symbol:
    var g = v.graph()
    let result = g.op(
        "mo.reshape", (v, shape), MOTensor(v.tensor_type().dtype, out_shape)
    )
    return result


fn reshape(v: Symbol, shape: Symbol, *out_shape: Int64) raises -> Symbol:
    var g = v.graph()
    let result = g.op(
        "mo.reshape", (v, shape), MOTensor(v.tensor_type().dtype, out_shape)
    )
    return result


fn reshape(rank: Int64, v: Symbol, shape: Symbol) raises -> Symbol:
    var g = v.graph()
    var reshape_dims = DynamicVector[Int64]()
    for _ in range(rank):
        reshape_dims.push_back(dyn())
    let result = g.op(
        "mo.reshape",
        (v, shape),
        MOTensor(v.tensor_type().dtype.dtype, reshape_dims),
    )
    return result


fn reshape_like(v: Symbol, like: Symbol) raises -> Symbol:
    return reshape(v, shape_of(like), like.tensor_type().dims)


def split[axis: Int](x: Symbol, split_sizes: (Int, Int)) -> (Symbol, Symbol):
    let tup = split[axis, N=2](
        x,
        StaticIntTuple[2](split_sizes.get[0, Int](), split_sizes.get[1, Int]()),
    )
    return (tup[0], tup[1])


def split[axis: Int, N: Int](x: Symbol, split_sizes: StaticIntTuple[N]) -> Tup:
    var g = x.graph()
    let sizes = g.constant(_tensor[N](TensorShape(N), split_sizes))

    let x_type = x.tensor_type()
    var out_dims = DynamicVector[Int64]()
    for i in range(x_type.rank()):
        out_dims.append(dyn())
    let split_type = MOTensor(x_type.dtype, out_dims).to_mlir(g.m)

    return g.nvop(
        "mo.split",
        (x, sizes, g.constant(axis)),
        Arity(split_type, split_type),
    )


def transpose_matrix(matrix: Symbol) -> Symbol:
    return transpose(matrix, -1, -2)


def transpose(input: Symbol, x: Int, y: Int) -> Symbol:
    var g = input.graph()
    let input_type = input.tensor_type()
    if input_type.rank() < 2:
        raise "transpose input must have rank >= 2"
    if x < 0:
        x += input_type.rank()
    if y < 0:
        y += input_type.rank()
    if x < 0 or x >= input_type.rank() or y < 0 or y >= input_type.rank():
        raise "transpose dim outside range"

    var dims = DynamicVector[Int64]()
    let ptr = DTypePointer[DType.int64].alloc(input_type.rank())
    for i in range(input_type.rank()):
        dims.push_back(input_type.dims[i])
        ptr.store(i, i)

    dims[x] = input_type.dims[y]
    dims[y] = input_type.dims[x]
    ptr.store(x, y)
    ptr.store(y, x)

    let transpose_indices = g.constant(
        Tensor[DType.int64](ptr, TensorShape(input_type.rank()))
    )

    return g.op(
        "mo.transpose",
        (input, transpose_indices),
        MOTensor(input_type.dtype, dims),
    )
