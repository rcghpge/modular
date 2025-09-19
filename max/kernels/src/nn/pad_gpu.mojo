# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from gpu import block_dim, block_idx, grid_dim, thread_idx
from gpu.host import DeviceContext
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor

from utils.index import IndexList, StaticTuple

# ===-----------------------------------------------------------------------===#
# pad GPU
# ===-----------------------------------------------------------------------===#


fn _fill_gpu_kernel[
    dtype: DType
](dst: UnsafePointer[Scalar[dtype]], value: Scalar[dtype], count: Int):
    var tid = thread_idx.x + block_idx.x * block_dim.x
    if tid < UInt(count):
        dst[tid] = value


fn _copy_gpu_kernel[
    dtype: DType
](
    dst: UnsafePointer[Scalar[dtype]],
    src: UnsafePointer[Scalar[dtype]],
    count: Int,
):
    var tid = thread_idx.x + block_idx.x * block_dim.x
    if tid < UInt(count):
        dst[tid] = src[tid]


fn _fill_strides_indexlist[
    rank: Int,
](
    input_shape: IndexList[rank],
    strides: LayoutTensor[mut=True, DType.int, Layout(rank)],
):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    constrained[rank > 0]()
    strides[rank - 1] = 1

    @parameter
    for idx in range(rank - 1):
        alias axis = rank - idx - 2
        var next_axis_stride = strides[axis + 1]
        var next_axis_dim = input_shape[axis + 1]
        var curr_axis_stride = next_axis_stride * next_axis_dim
        strides[axis] = curr_axis_stride


fn _fill_gpu[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype]],
    value: Scalar[dtype],
    count: Int,
    ctx: DeviceContext,
) raises:
    alias block_dim = 256
    ctx.enqueue_function[_fill_gpu_kernel[dtype]](
        ptr,
        value,
        count,
        grid_dim=((count + block_dim) // block_dim),
        block_dim=(block_dim),
    )


fn _memcpy_gpu[
    dtype: DType
](
    dst: UnsafePointer[Scalar[dtype]],
    src: UnsafePointer[Scalar[dtype]],
    count: Int,
    ctx: DeviceContext,
) raises:
    alias block_dim = 256
    ctx.enqueue_function[_copy_gpu_kernel[dtype]](
        dst,
        src,
        count,
        grid_dim=((count + block_dim) // block_dim),
        block_dim=(block_dim),
    )


@register_passable("trivial")
struct _AxisParams[rank: Int, dtype: DType, paddings_type: DType](
    ImplicitlyCopyable & Movable
):
    var pre_pad: Int
    var post_pad: Int
    var non_pad: Int

    var output_offset: Int
    var input_offset: Int
    var pad_with_constant: Bool
    var is_within_padding: Bool
    var next_pad_with_constant: Bool
    var output_shape: IndexList[rank]

    """
    output_offset: The offset at which output data starts.
    input_offset: The offset at which input data starts.
    pad_with_constant: whether to always pad remaining region with constant.
    """

    @always_inline
    fn __init__(
        out self,
        axis: Int,
        paddings: UnsafePointer[Scalar[paddings_type]],
        output_shape: IndexList[rank],
    ):
        var axis_dim = output_shape[axis]
        var pre_pad = Int(paddings[2 * axis])
        var post_pad = Int(paddings[2 * axis + 1])
        var non_pad = axis_dim - pre_pad - post_pad

        self.pre_pad = pre_pad
        self.post_pad = post_pad
        self.non_pad = non_pad
        self.output_offset = 0
        self.input_offset = 0
        self.pad_with_constant = False
        self.is_within_padding = False
        self.next_pad_with_constant = False
        self.output_shape = output_shape

    @always_inline
    fn init_offsets(
        mut self,
        output_offset: Int,
        input_offset: Int,
        pad_with_constant: Bool,
    ):
        self.output_offset = output_offset
        self.input_offset = input_offset
        self.pad_with_constant = pad_with_constant

    @always_inline
    fn pre_check(mut self, i: Int):
        self.is_within_padding = (i < self.pre_pad) or (
            self.pre_pad + self.non_pad <= i
        )
        self.next_pad_with_constant = (
            self.pad_with_constant or self.is_within_padding
        )

    @always_inline
    fn post_check(mut self, output_axis_stride: Int, input_axis_stride: Int):
        if not self.is_within_padding:
            self.input_offset += input_axis_stride
        self.output_offset += output_axis_stride

    @always_inline
    fn base(
        mut self,
        output: UnsafePointer[Scalar[dtype]],
        input: UnsafePointer[Scalar[dtype]],
        constant: Scalar[dtype],
        axis_dim: Int,
        ctx: DeviceContext,
    ) raises:
        var pre_pad_start_ptr = output.offset(self.output_offset)

        # setting values
        if self.pad_with_constant:
            _fill_gpu(pre_pad_start_ptr, constant, axis_dim, ctx)
        else:
            var non_pad_start_ptr = pre_pad_start_ptr.offset(self.pre_pad)
            var post_pad_start_ptr = non_pad_start_ptr.offset(self.non_pad)
            var input_start_ptr = input.offset(self.input_offset)
            _fill_gpu(pre_pad_start_ptr, constant, self.pre_pad, ctx)
            _memcpy_gpu(non_pad_start_ptr, input_start_ptr, self.non_pad, ctx)
            _fill_gpu(post_pad_start_ptr, constant, self.post_pad, ctx)


@always_inline
fn _pad_constant_axis[
    rank: Int, dtype: DType, paddings_type: DType, axis: Int
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    constant: Scalar[dtype],
    output_shape: IndexList[rank],
    output_strides: UnsafePointer[Scalar[DType.int]],
    input_strides: UnsafePointer[Scalar[DType.int]],
    var axis_params: StaticTuple[_AxisParams[rank, dtype, paddings_type], rank],
    ctx: DeviceContext,
) raises:
    @parameter
    if axis == (rank - 1):
        # print(product(self.output_shape, rank))
        axis_params[axis].base(output, input, constant, output_shape[axis], ctx)

    else:
        var output_axis_stride = Int(output_strides[axis])
        var input_axis_stride = Int(input_strides[axis])
        for i in range(output_shape[axis]):
            axis_params[axis].pre_check(i)

            axis_params[axis + 1].init_offsets(
                axis_params[axis].output_offset,
                axis_params[axis].input_offset,
                axis_params[axis].next_pad_with_constant,
            )
            _pad_constant_axis[rank, dtype, paddings_type, axis + 1](
                output,
                input,
                constant,
                output_shape,
                output_strides,
                input_strides,
                axis_params,
                ctx,
            )

            axis_params[axis].post_check(output_axis_stride, input_axis_stride)


fn _pad_constant_impl[
    rank: Int, dtype: DType, paddings_type: DType
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    paddings: UnsafePointer[Scalar[paddings_type]],
    constant: Scalar[dtype],
    output_shape: IndexList[rank],
    output_strides: UnsafePointer[Scalar[DType.int]],
    input_strides: UnsafePointer[Scalar[DType.int]],
    ctx: DeviceContext,
) raises:
    """
    Fill axis ∈ [axis, rank) in `output` with values from `input`, and edges
    padded with `constant` based on `paddings`.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: The (before, after) padding sizes for each axis.
        constant: the constant to pad output with.
        output_shape: the dynamic shape of the tensor pointed to by output buffer
        output_strides: the stride at each output axis.
        input_strides: the stride at each input axis.
        ctx: Device context for participating GPU.
    """

    # allocate 'rank' axis-data vector, only use the ones in range[axis,rank)
    var axis_params = StaticTuple[
        _AxisParams[rank, dtype, paddings_type], rank
    ]()

    @parameter
    for r in range(rank):
        axis_params[r] = _AxisParams[rank, dtype, paddings_type](
            r, paddings, output_shape
        )

    # CRITICAL: should be setting output_offset=0, input_offset=0, and
    # pad_with_constant=False for axis=0 in padding. However, this is
    # already addressed in the constructor of _AxisParams.

    # axis_params[0].init_offsets(output_offset, input_offset, pad_with_constant)
    _pad_constant_axis[rank, dtype, paddings_type, 0](
        output,
        input,
        constant,
        output_shape,
        output_strides,
        input_strides,
        axis_params,
        ctx,
    )


fn pad_constant[
    rank: Int, dtype: DType, padding_type: DType
](
    output: UnsafePointer[Scalar[dtype]],
    output_shape: IndexList[rank],
    input: UnsafePointer[Scalar[dtype]],
    input_shape: IndexList[rank],
    paddings: UnsafePointer[Scalar[padding_type]],
    # TODO: implement (before, after) variant
    constant: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    """
    Fill `output` with values from `input`, and edges padded with `constant`
    based on `paddings`.

    Args:
        output: The output buffer.
        output_shape: The output shape.
        input: The input buffer.
        input_shape: The input shape.
        paddings: Ordered (before, after) padding sizes for each axis.
        constant: The constant to pad output with.
        ctx: Device context for participating GPU.

    Example:
        ```mojo
        var input_shape = (X, Y, Z)
        var paddings = [x0, x1, y0, y1, z0, z1]

        out[x, y, z] =
          input[x - x0, y - y0, z - z0] if x ∈ [x0, x0 + X] &&
                                           y ∈ [y0, y0 + Y] &&
                                           z ∈ [z0, z0 + Z]
          else constant
        ```
    """
    var constant_cast = rebind[Scalar[dtype]](constant[0])

    @__copy_capture(constant_cast)
    @parameter
    fn pad_constant_wrapper(
        output: UnsafePointer[Scalar[dtype]],
        input: UnsafePointer[Scalar[dtype]],
        paddings: UnsafePointer[Scalar[padding_type]],
        output_shape: IndexList[rank],
        output_strides: UnsafePointer[Scalar[DType.int]],
        input_strides: UnsafePointer[Scalar[DType.int]],
        ctx: DeviceContext,
    ) raises:
        return _pad_constant_impl[rank, dtype](
            output,
            input,
            paddings,
            constant_cast,
            output_shape,
            output_strides,
            input_strides,
            ctx,
        )

    var input_strides_buf = LayoutTensor[
        DType.int, Layout(rank), MutableAnyOrigin
    ].stack_allocation()
    var output_strides_buf = LayoutTensor[
        DType.int, Layout(rank), MutableAnyOrigin
    ].stack_allocation()
    _fill_strides_indexlist[rank](input_shape, input_strides_buf)
    _fill_strides_indexlist[rank](output_shape, output_strides_buf)

    return pad_constant_wrapper(
        output,
        input,
        paddings,
        output_shape,
        output_strides_buf.ptr,
        input_strides_buf.ptr,
        ctx,
    )


fn get_padding_output_shape[
    rank: Int
](
    input_shape: IndexList[rank],
    paddings: LayoutTensor[DType.int, Layout(2 * rank)],
) -> IndexList[rank]:
    var output_shape = IndexList[rank]()
    for i in range(rank):
        var before = paddings[2 * i]
        var after = paddings[2 * i + 1]
        output_shape[i] = Int(before) + input_shape[i] + Int(after)
    return output_shape
