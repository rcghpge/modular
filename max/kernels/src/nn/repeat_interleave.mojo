# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements `repeat_interleave`, which repeats each tensor element a specified number of times along an axis."""

from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from layout import Coord, Idx, TileTensor, coord_to_index_list, row_major

from std.utils import IndexList


def _collapse_dims_around_axis(
    shape: IndexList, axis: Int, out result: IndexList[3]
) raises:
    if axis >= shape.size:
        raise Error("axis larger than provided shape")

    comptime if shape.size == 0:
        return IndexList[3](1, 1, 1)

    var split = shape[axis]

    var before = 1
    for i in range(axis):
        before *= shape[i]

    var after = 1
    for i in range(axis + 1, shape.size):
        after *= shape[i]

    return IndexList[3](before, split, after)


@inline(.always)
def _repeat_interleave_cpu[
    dtype: DType,
    type_repeats: DType,
](
    input: TileTensor[dtype, ...],
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """
    Fill `output` by repeating values from `input` along `axis` based on the
    values in `repeats` buffer.

    This is intended to implement the same functionality as torch.repeat:
    https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html

    Args:
        input: The input buffer.
        repeats: The number of repetitions each element in input.
        axis: The axis along which to repeat values.
        output: The output buffer.
        ctx: Device context.
    """
    # comptime assert (is_row_major[input.rank](input.layout)) and (
    #     is_row_major[output.rank](output.layout)
    # )
    comptime assert input.flat_rank == output.flat_rank
    comptime assert repeats.flat_rank == 1

    # Compute the shape of the input and result buffers.
    # These are the shapes of the buffers we will be working on.
    var collapsed_input_shape = _collapse_dims_around_axis(
        coord_to_index_list(input.layout.shape_coord()), axis
    )
    var collapsed_output_shape = _collapse_dims_around_axis(
        coord_to_index_list(output.layout.shape_coord()), axis
    )

    assert collapsed_output_shape[0] == collapsed_input_shape[0]
    assert collapsed_output_shape[2] == collapsed_input_shape[2]

    var collapsed_input = TileTensor(
        input.ptr,
        row_major(Coord(collapsed_input_shape)),
    )

    var collapsed_output = TileTensor(
        output.ptr,
        row_major(Coord(collapsed_output_shape)),
    )

    var input_repeat_dim = collapsed_input_shape[1]
    var output_repeat_dim = collapsed_output_shape[1]
    var repeat_stride = Int(repeats.dim[0]() > 1)

    # Mapping from offsets in the input tensor to offsets in the output tensor
    # along the repeat axis.
    var offset_mapping = List[Int](unsafe_uninit_length=output_repeat_dim)

    var repeat_offset = 0
    var output_offset = 0
    for input_offset in range(input_repeat_dim):
        var repeat_val = Int(repeats[repeat_offset])
        if repeat_val < 0:
            raise Error("all repeat values must be non-negative")

        for _ in range(repeat_val):
            offset_mapping[output_offset] = input_offset
            output_offset += 1

        repeat_offset += repeat_stride

    # `offset_mapping` is a `List` (not register-passable), so it cannot be
    # captured directly by the unified closure. Capture a `Span` view instead;
    # the `Span` carries the list's origin, keeping it alive through the
    # closure's last use. Everything else is captured by `mut`, matching the
    # original implicit-capture behavior.
    var offset_mapping_view = Span(offset_mapping)

    @inline(.always)
    def func[
        width: Int, alignment: Int = 1
    ](idx: Coord) {var offset_mapping_view, mut}:
        var output_index = rebind[IndexList[3]](coord_to_index_list(idx))
        var input_index = output_index
        input_index[1] = offset_mapping_view[output_index[1]]

        var input_idx = collapsed_input.layout(Coord(input_index))
        var input_value = collapsed_input.raw_load[width=width](input_idx)

        var output_idx = collapsed_output.layout(Coord(output_index))
        collapsed_output.raw_store(output_idx, input_value)

    elementwise[simd_width_of[output.dtype]()](
        func, collapsed_output.layout.shape_coord(), ctx
    )


def _repeat_interleave_gpu[
    dtype: DType,
    type_repeats: DType,
    target: StaticString,
](
    input: TileTensor[dtype, ...],
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Fills `output` on a GPU for the scalar-repeats case only.

    The GPU path requires a uniform repeat count: `repeats` is a size-1
    buffer whose single value is read on the host (it is a constant built
    from the Python scalar). Per-element repeats stay on the CPU path,
    which builds the output->input offset map from the repeats values.

    With a uniform count the output->input mapping is pure arithmetic --
    every input element expands into `repeat_count` consecutive output
    slots -- so the elementwise body decodes the source index with a
    division instead of a data-dependent gather. The decode uses the
    collapsed [outer, axis, inner] views over the same memory as the CPU
    path, with the extents read on the host and captured by value (the
    device-side layout round-trips misplace indices).
    """
    comptime assert input.flat_rank == output.flat_rank
    comptime assert repeats.flat_rank == 1

    var input_dims = coord_to_index_list(input.layout.shape_coord())
    var output_dims = coord_to_index_list(output.layout.shape_coord())
    var in_axis = Int(input_dims[axis])
    var out_axis = Int(output_dims[axis])

    if repeats.dim[0]() != 1:
        raise Error(
            "repeat_interleave GPU path requires scalar repeats; per-element"
            " repeats are CPU-only"
        )

    # Derive the repeat count from the shapes instead of reading the
    # device-resident repeats buffer on the host (a raw host read of
    # device memory crashes). The shape function already guaranteed
    # out_axis == in_axis * repeats, so the division is exact.
    if in_axis == 0:
        raise Error("repeat_interleave: input axis size must be positive")
    if out_axis % in_axis != 0:
        raise Error(
            "repeat_interleave: output axis size must be a multiple of the"
            " input axis size for scalar repeats"
        )
    var repeat_count = out_axis // in_axis
    if repeat_count < 0:
        raise Error("all repeat values must be non-negative")

    var collapsed_input_shape = _collapse_dims_around_axis(input_dims, axis)
    var collapsed_output_shape = _collapse_dims_around_axis(output_dims, axis)

    assert collapsed_output_shape[0] == collapsed_input_shape[0]
    assert collapsed_output_shape[2] == collapsed_input_shape[2]

    var collapsed_input = TileTensor(
        input.ptr,
        row_major(Coord(collapsed_input_shape)),
    )
    var collapsed_output = TileTensor(
        output.ptr,
        row_major(Coord(collapsed_output_shape)),
    )

    var num_elements = output.num_elements()
    if num_elements == 0:
        return

    @inline(.always)
    def func[width: Int, alignment: Int = 1](idx: Coord) {var}:
        var output_index = rebind[IndexList[3]](coord_to_index_list(idx))
        var input_index = output_index
        input_index[1] = output_index[1] // repeat_count

        var input_idx = collapsed_input.layout(Coord(input_index))
        var input_value = collapsed_input.raw_load[width=width](input_idx)

        var output_idx = collapsed_output.layout(Coord(output_index))
        collapsed_output.raw_store(output_idx, input_value)

    elementwise[simd_width_of[output.dtype](), target=target](
        func, collapsed_output.layout.shape_coord(), ctx
    )


@inline(.always)
def repeat_interleave[
    dtype: DType,
    type_repeats: DType,
    //,
    target: StaticString = "cpu",
](
    input: TileTensor[dtype, ...],
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """Fills `output` by repeating `input` along `axis`, on `target`.

    The CPU path handles both scalar and per-element `repeats`; the GPU
    path is scalar-repeats only (see `_repeat_interleave_gpu`).
    """
    comptime if is_cpu[target]():
        _repeat_interleave_cpu(input, repeats, axis, output, ctx)
    else:
        _repeat_interleave_gpu[dtype, type_repeats, target](
            input, repeats, axis, output, ctx
        )


@inline(.always)
def repeat_interleave_shape[
    type_repeats: DType,
](
    input: TileTensor,
    repeats: TileTensor[type_repeats, ...],
    axis: Int,
) raises -> IndexList[input.rank]:
    """
    Computes the output shape of `repeat_interleave` for the given `input`,
    `repeats`, and `axis`.

    The returned `IndexList` matches `input`'s rank with the size along
    `axis` replaced by the summed `repeats` values, or by `input[axis]`
    multiplied by the single repeat value when `repeats` is a size-1
    broadcast.

    Args:
        input: The input tensor whose shape is being transformed.
        repeats: A one-dimensional integral tensor of per-element repeat
            counts, either size 1 or equal to `input.dim(axis)`.
        axis: The axis along which elements are repeated.

    Returns:
        An `IndexList` matching `input.rank` with the `axis` dimension
        updated to the total repeated size.
    """
    comptime assert type_repeats.is_integral()
    comptime assert repeats.flat_rank == 1

    var repeats_size = repeats.dim[0]()
    if repeats_size != 1 and repeats_size != Scalar[repeats.linear_idx_type](
        Int(input.dim(axis))
    ):
        raise Error(
            "repeat_interleave: repeats must be size 1 or equal to "
            "the size of input[axis]"
        )

    var total_repeats = 0
    comptime assert repeats_size.dtype.is_integral()
    for i in range(repeats_size):
        total_repeats += Int(repeats[i])

    var result = coord_to_index_list(input.layout.shape_coord())

    # If the repeats is size 1, the repeat is treated as a broadcast
    if repeats_size == 1:
        result[axis] *= total_repeats
    else:
        result[axis] = total_repeats

    return rebind[IndexList[input.rank]](result)
