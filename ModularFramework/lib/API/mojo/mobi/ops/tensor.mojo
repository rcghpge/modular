# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorSpec, TensorShape

from mobi.misc import Tup


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
        dim += v.type().unwrap_tensor().spec().rank()
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
            stop.store(i, slice(None, None, None).end)
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

    return squeeze(slice, axis)


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


@value
struct SliceSymbol:
    var start: Symbol
    var stop: Symbol
    var step: Symbol

    fn __init__(inout self, stop: Symbol) raises:
        var g = stop.g
        self.__init__(g.constant(0), stop)

    fn __init__(inout self, start: Symbol, stop: Symbol) raises:
        var g = stop.g
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
    let spec = input.type().unwrap_tensor().spec()
    var dims = DynamicVector[Int]()

    for slice in slices:
        dims.push_back(input.g.m.dyn().to_int())
    for i in range(len(slices), spec.rank()):
        dims.push_back(spec[i])

    return slice_(input, dims, slices)


def slice_(
    input: Symbol, shape: TensorShape, borrowed *slices: SliceSymbol
) -> Symbol:
    return slice_(input, shape, slices)


def slice_(
    input: Symbol, shape: TensorShape, slices: VariadicListMem[SliceSymbol]
) -> Symbol:
    var starts = DynamicVector[Symbol]()
    var stops = DynamicVector[Symbol]()
    var steps = DynamicVector[Symbol]()

    var g = input.g
    let spec = input.type().unwrap_tensor().spec()

    fn ensure_constant(value: Symbol) raises -> Symbol:
        let spec = value.type().unwrap_tensor().spec()
        if spec.num_elements() != 1:
            raise "slice indicies must be singular"
        return value if spec.rank() == 0 else reshape(value, TensorShape())

    let slice_max_value = slice(None, None, None).end
    for dim in range(len(slices)):
        let slice: SliceSymbol = __get_address_as_lvalue(slices[dim])
        starts.push_back(ensure_constant(slice.start))
        stops.push_back(ensure_constant(slice.stop))
        steps.push_back(ensure_constant(slice.step))
    for dim in range(len(slices), spec.rank()):
        starts.push_back(g.constant(0))
        stops.push_back(g.constant(slice_max_value))
        steps.push_back(g.constant(1))

    let start = stack[axis=0](starts)
    let stop = stack[axis=0](stops)
    let step = stack[axis=0](steps)
    return g.op(
        "mo.slice",
        (input, start, stop, step),
        TensorSpec(spec.dtype(), shape),
    )[0]


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
    var vec = DynamicVector[Symbol](len(values))
    for value in values:
        vec.append(value)
    return concat[axis](vec)


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

    return g.op(
        "mo.concat",
        concat_args,
        TensorSpec(spec.dtype(), TensorShape(dims)),
    )[0]


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
    var g = v.g
    let spec = v.type().unwrap_tensor().spec()
    if axis < 0:
        axis += spec.rank()

    let new_shape = g.op(
        "mo.squeeze_shape",
        (
            shape_of(v),
            g.constant(Tensor[DType.int64](TensorShape(1), axis)),
        ),
        TensorSpec(DType.int64, spec.rank() - 1),
    )[0]

    var dims = DynamicVector[Int]()
    for i in range(spec.rank()):
        if i != axis:
            dims.push_back(spec[i])

    return g.op(
        "mo.reshape",
        (v, new_shape),
        TensorSpec(spec.dtype(), TensorShape(dims)),
    )[0]


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
    for dim in dims:
        if dim.type().unwrap_tensor().spec().rank() > 1:
            raise "reshape: got multi-dimensional shape"

    if len(dims) == 1:
        # Special case: we're maybe passing in a dynamic reshape shape
        let shape = dims[0]
        let shape_spec = shape.type().unwrap_tensor().spec()
        if shape_spec[0] > 1:  # only specialize for statically known rank > 1
            return reshape(shape_spec[0], v, shape)

    let rank = len(dims)
    var vec = DynamicVector[Symbol](len(dims))
    for dim in dims:
        vec.append(dim)
    let shape = stack[axis=0](vec)

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


fn reshape_like(v: Symbol, like: Symbol) raises -> Symbol:
    let shape = like.type().unwrap_tensor().spec().shape
    return reshape(v, shape_of(like), shape)


def split[axis: Int](x: Symbol, split_sizes: (Int, Int)) -> (Symbol, Symbol):
    let tup = split[axis, N=2](
        x,
        StaticIntTuple[2](split_sizes.get[0, Int](), split_sizes.get[1, Int]()),
    )
    return (tup[0], tup[1])


def split[axis: Int, N: Int](x: Symbol, split_sizes: StaticIntTuple[N]) -> Tup:
    var g = x.g
    let sizes = g.constant(_tensor[N](TensorShape(N), split_sizes))

    let spec = x.type().unwrap_tensor().spec()
    let out_shape = _fill_vector(spec.rank(), g.m.dyn().to_int())
    let split_ty = g.m.type(TensorSpec(spec.dtype(), out_shape)).type_ptr()

    return g.op(
        "mo.split",
        Tup(x, sizes, g.constant(axis)),
        g.m.types(split_ty, split_ty),
    )


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
