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


# ===-----------------------------------------------------------------------===#
# pad
# ===-----------------------------------------------------------------------===#


from layout import (
    LayoutTensor,
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)

# TODO Refactor -- we should decide on and put them into a more common file
from linalg.transpose import _fill_strides
from memory import memcpy

from utils import IndexList, StaticTuple


@always_inline
fn _fill[
    dtype: DType
](dst: UnsafePointer[Scalar[dtype]], value: Scalar[dtype], count: Int):
    alias layout = Layout.row_major(UNKNOWN_VALUE)
    _ = LayoutTensor[dtype, layout](
        dst, RuntimeLayout[layout].row_major(IndexList[1](count))
    ).fill(value)


# TODO: could this be deleted? maybe replaced with faster collapsed loop.
struct _NestedLoopIter[n_loops: Int]:
    """
    Helper iterable for padding functions meant to represent an n-level loop nest of
    the form:

    for i1 in range(lower_i1, upper_i1):
       for i2 in range(lower_i2, upper_i2):
           for i3 in range(lower_i3, upper_i3):
             .....
    """

    var cur: IndexList[n_loops]

    alias LoopBoundSpec = InlineArray[IndexList[2], n_loops]
    var loop_bounds: Self.LoopBoundSpec
    var early_stop: Bool

    fn __init__(out self, loop_bounds: Self.LoopBoundSpec):
        debug_assert(
            len(loop_bounds) == n_loops,
            (
                "Number of entries in loop_bounds doesn't match the number of"
                " loops specified"
            ),
        )

        # TODO: Should this function take an `owned loop_bounds` to avoid
        #   a copy in places where the caller already has an owned value?
        self.loop_bounds = loop_bounds.copy()

        self.cur = IndexList[n_loops]()
        self.early_stop = False

        for i in range(n_loops):
            var lb = self._lb_loop(i)
            var ub = self._ub_loop(i)

            self.cur[i] = lb

            var invalid_bound = lb >= ub
            self.early_stop = self.early_stop or invalid_bound

    fn _lb_loop(self, axis: Int) -> Int:
        return self.loop_bounds[axis][0]

    fn _ub_loop(self, axis: Int) -> Int:
        return self.loop_bounds[axis][1]

    fn __copyinit__(out self, other: Self):
        self.cur = other.cur
        self.loop_bounds = other.loop_bounds.copy()
        self.early_stop = other.early_stop

    fn __iter__(mut self) -> Self:
        return self

    fn __next__(mut self) -> IndexList[n_loops]:
        var cur = self.cur

        self.cur[len(self.cur) - 1] += 1

        for i in range(n_loops - 1, 0, -1):
            if self.cur[i] == self._ub_loop(i):
                self.cur[i] = self._lb_loop(i)
                self.cur[i - 1] += 1

        return cur

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        if self.cur[0] >= self._ub_loop(0) or self.early_stop:
            return 0
        else:
            return 1


fn pad_constant[
    output_layout: Layout,
    input_layout: Layout,
    dtype: DType,
    paddings_type: DType,
    constant_type: DType,
](
    output: LayoutTensor[
        mut=True,
        dtype,
        output_layout,
        address_space = AddressSpace.GENERIC, **_,
    ],
    input: LayoutTensor[
        dtype, input_layout, address_space = AddressSpace.GENERIC, **_
    ],
    paddings: UnsafePointer[Scalar[paddings_type]],
    constant: SIMD[constant_type, 1],
):
    """
    Fill `output` with values from `input`, and edges padded with `constant`
    based on `paddings`.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: Ordered (before, after) padding sizes for each axis.
        constant: The constant to pad output with.

    Example:
        var input_shape = (X, Y, Z)
        var paddings = [x0, x1, y0, y1, z0, z1]

        out[x, y, z] =
          input[x - x0, y - y0, z - z0] if x ∈ [x0, x0 + X] &&
                                           y ∈ [y0, y0 + Y] &&
                                           z ∈ [z0, z0 + Z]
          else constant
    """
    var constant_cast = rebind[Scalar[dtype]](constant[0])

    @__copy_capture(constant_cast)
    @parameter
    fn pad_constant_wrapper(
        output: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.GENERIC, **_
        ],
        input: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.GENERIC, **_
        ],
        paddings: UnsafePointer[Scalar[paddings_type]],
        output_shape: IndexList[output_layout.rank()],
        output_strides: UnsafePointer[Scalar[DType.index]],
        input_strides: UnsafePointer[Scalar[DType.index]],
    ):
        return _pad_constant_impl[output_layout.rank(), dtype, paddings_type](
            output,
            input,
            paddings,
            constant_cast,
            output_shape,
            output_strides,
            input_strides,
        )

    return _do_pad[
        output_layout,
        input_layout,
        dtype,
        paddings_type,
        pad_constant_wrapper,
    ](output, input, paddings)


fn pad_reflect[
    output_layout: Layout,
    input_layout: Layout,
    dtype: DType,
    paddings_type: DType,
](
    output: LayoutTensor[
        mut=True,
        dtype,
        output_layout,
        address_space = AddressSpace.GENERIC, **_,
    ],
    input: LayoutTensor[
        dtype, input_layout, address_space = AddressSpace.GENERIC, **_
    ],
    paddings: UnsafePointer[Scalar[paddings_type]],
):
    """
    Fill `output` with values from `input`, and edges padded with reflected
    values from the unpadded region.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: Ordered (before, after) padding sizes for each axis.

    Example:
        var input = [[1, 2],
                     [3, 4]]
        var paddings = [2, 2, 1, 0]

        Yields:
        output = [[2, 1, 2],
                  [4, 3, 4],
                  [2, 1, 2],
                  [4, 3, 4],
                  [2, 1, 2],
                  [4, 3, 4]]
    """

    @parameter
    fn pad_reflect_wrapper(
        output: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.GENERIC, **_
        ],
        input: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.GENERIC, **_
        ],
        paddings: UnsafePointer[Scalar[paddings_type]],
        output_shape: IndexList[output_layout.rank()],
        output_strides: UnsafePointer[Scalar[DType.index]],
        input_strides: UnsafePointer[Scalar[DType.index]],
    ):
        return _pad_reflect_impl[output_layout.rank(), dtype, paddings_type](
            output, input, paddings, output_shape, output_strides, input_strides
        )

    return _do_pad[
        output_layout,
        input_layout,
        dtype,
        paddings_type,
        pad_reflect_wrapper,
    ](output, input, paddings)


@always_inline
fn pad_shape[
    input_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: LayoutTensor[input_type, **_],
    paddings_buf: LayoutTensor[paddings_type, **_],
) raises -> IndexList[input_buf.rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_type: Type of the input tensor.
        paddings_type: Type of the padding tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_buf: The tensor to pad.
        paddings_buf: The paddings tensor, of shape (input_rank, 2).

    Returns:
        The output shape.
    """
    constrained[paddings_buf.rank == 1, "paddings_buf must be of rank 1"]()

    # TODO add runtime test once we support dynamic rank execution, currently
    # MLIR verifier of `MO::PadLike` prevents testing this with static rank.
    if Int(paddings_buf.dim[0]()) != 2 * input_buf.rank:
        raise Error("[pad] paddings shape must be (2 * input_rank)")

    # compute and return the output shape
    var output_shape = IndexList[input_buf.rank]()

    @parameter
    for axis in range(input_buf.rank):
        var pre_pad = Int(paddings_buf[2 * axis])
        var post_pad = Int(paddings_buf[2 * axis + 1])
        output_shape[axis] = pre_pad + Int(input_buf.dim[axis]()) + post_pad

    return output_shape


fn _do_pad[
    output_layout: Layout,
    input_layout: Layout,
    dtype: DType,
    paddings_type: DType,
    pad_impl_fn: fn (
        UnsafePointer[Scalar[dtype], address_space = AddressSpace.GENERIC, **_],
        UnsafePointer[Scalar[dtype], address_space = AddressSpace.GENERIC, **_],
        UnsafePointer[Scalar[paddings_type]],
        IndexList[output_layout.rank()],
        UnsafePointer[Scalar[DType.index]],
        UnsafePointer[Scalar[DType.index]],
    ) capturing [_] -> None,
](
    output: LayoutTensor[
        mut=True,
        dtype,
        output_layout,
        address_space = AddressSpace.GENERIC, **_,
    ],
    input: LayoutTensor[
        dtype, input_layout, address_space = AddressSpace.GENERIC, **_
    ],
    paddings: UnsafePointer[Scalar[paddings_type]],
):
    var input_strides_stack = InlineArray[Scalar[DType.index], output.rank](
        uninitialized=True
    )
    var input_strides_buf = LayoutTensor[
        DType.index, Layout.row_major(input.rank)
    ](input_strides_stack)
    var output_strides_stack = InlineArray[Scalar[DType.index], output.rank](
        uninitialized=True
    )
    var output_strides_buf = LayoutTensor[
        DType.index, Layout.row_major(output.rank)
    ](output_strides_stack)
    _fill_strides(input, input_strides_buf)
    _fill_strides(output, output_strides_buf)

    var output_shape = IndexList[output_layout.rank()]()

    @parameter
    for axis in range(output_layout.rank()):
        output_shape[axis] = output.dim[axis]()

    return pad_impl_fn(
        output.ptr,
        input.ptr,
        paddings,
        output_shape,
        output_strides_buf.ptr,
        input_strides_buf.ptr,
    )


@register_passable("trivial")
struct _AxisParams[rank: Int, dtype: DType, paddings_type: DType](
    Copyable & Movable
):
    var pre_pad: Int
    var post_pad: Int
    var non_pad: Int

    var output_offset: Int
    var input_offset: Int
    var pad_with_constant: Bool
    var is_within_padding: Bool
    var next_pad_with_constant: Bool

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

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

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
    ):
        var pre_pad_start_ptr = output.offset(self.output_offset)

        # setting values
        if self.pad_with_constant:
            _fill(pre_pad_start_ptr, constant, axis_dim)
        else:
            var non_pad_start_ptr = pre_pad_start_ptr.offset(self.pre_pad)
            var post_pad_start_ptr = non_pad_start_ptr.offset(self.non_pad)
            var input_start_ptr = input.offset(self.input_offset)
            _fill(pre_pad_start_ptr, constant, self.pre_pad)
            memcpy(non_pad_start_ptr, input_start_ptr, self.non_pad)
            _fill(post_pad_start_ptr, constant, self.post_pad)


@always_inline
fn _pad_constant_axis[
    rank: Int, dtype: DType, paddings_type: DType, axis: Int
](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    constant: Scalar[dtype],
    output_shape: IndexList[rank],
    output_strides: UnsafePointer[Scalar[DType.index]],
    input_strides: UnsafePointer[Scalar[DType.index]],
    owned axis_params: StaticTuple[
        _AxisParams[rank, dtype, paddings_type], rank
    ],
):
    @parameter
    if axis == (rank - 1):
        axis_params[axis].base(output, input, constant, output_shape[axis])
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
    output_strides: UnsafePointer[Scalar[DType.index]],
    input_strides: UnsafePointer[Scalar[DType.index]],
):
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
    )


@always_inline
fn _memcpy_regions_fast[
    dtype: DType
](
    pre_pad: Int,
    post_pad: Int,
    non_pad: Int,
    output_axis_stride: Int,
    pre_pad_start_ptr: UnsafePointer[Scalar[dtype]],
):
    @always_inline
    fn modulo_inc(mut cnt: Int, modulo: Int):
        """
        Returns '(cnt+1)%modulo', provided that 'cnt' is initialized to zero
        and all the increments are via this function.
        """
        cnt += 1
        if cnt == modulo:
            cnt = 0

    @parameter
    fn _common_loop[pre_copy: Bool, singleton: Bool]():
        var curr_rem: Int = 0
        var num_iters = pre_pad if pre_copy else post_pad
        var copy_to: Int = (pre_pad - 1) if pre_copy else (pre_pad + non_pad)

        for curr in range(num_iters):
            var copy_from: Int

            @parameter
            if singleton:  # non_pad == 1
                # handle singleton case
                copy_from = pre_pad
            else:
                # curr_rem = (curr % (non_pad - 1))
                var fwd: Int = (curr_rem + 1) * 2
                modulo_inc(curr_rem, non_pad - 1)

                # copy_from = copy_to +- ((curr % (non_pad - 1)) + 1) * 2
                copy_from = (copy_to + fwd) if pre_copy else (copy_to - fwd)

            var copy_to_ptr = pre_pad_start_ptr + (copy_to * output_axis_stride)
            var copy_from_ptr = pre_pad_start_ptr + (
                copy_from * output_axis_stride
            )

            memcpy(copy_to_ptr, copy_from_ptr, output_axis_stride)
            copy_to += -1 if pre_copy else +1

    if non_pad == 1:
        _common_loop[pre_copy=True, singleton=True]()
        _common_loop[pre_copy=False, singleton=True]()
    else:
        _common_loop[pre_copy=True, singleton=False]()
        _common_loop[pre_copy=False, singleton=False]()


@register_passable("trivial")
struct _AxisParamsReflect[rank: Int, dtype: DType, paddings_type: DType](
    Copyable & Movable
):
    var pre_pad: Int
    var post_pad: Int
    var non_pad: Int

    var next_input_offset: Int
    var next_output_offset: Int

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

        self.next_input_offset = 0
        self.next_output_offset = 0

    @always_inline
    fn init_offsets(
        mut self,
        output_offset: Int,
        input_offset: Int,
        output_axis_stride: Int,
    ):
        # setting offsets for the lower dimensions
        self.next_input_offset = input_offset
        self.next_output_offset = output_offset + (
            output_axis_stride * self.pre_pad
        )

    @always_inline
    fn update_next_offsets(
        mut self, output_axis_stride: Int, input_axis_stride: Int
    ):
        self.next_output_offset += output_axis_stride
        self.next_input_offset += input_axis_stride

    @always_inline
    fn base(
        mut self,
        output_offset: Int,
        input_offset: Int,
        output: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.GENERIC, **_
        ],
        input: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.GENERIC, **_
        ],
    ):
        # no more dimensions to recurse, copy from input to unpadded region
        var non_pad_start_ptr = output.offset(output_offset + self.pre_pad)
        var input_start_ptr = input.offset(input_offset)
        memcpy(non_pad_start_ptr, input_start_ptr, self.non_pad)

    @always_inline
    fn memcpy_regions(
        mut self,
        output_axis_stride: Int,
        output_offset: Int,
        output: UnsafePointer[Scalar[dtype]],
    ):
        var pre_pad_start_ptr = output.offset(output_offset)

        _memcpy_regions_fast(
            self.pre_pad,
            self.post_pad,
            self.non_pad,
            output_axis_stride,
            pre_pad_start_ptr,
        )


@always_inline
fn _pad_reflect_axis[
    rank: Int,
    dtype: DType,
    paddings_type: DType,
    axis: Int,
](
    output: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GENERIC, **_
    ],
    input: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GENERIC, **_
    ],
    output_strides: UnsafePointer[Scalar[DType.index]],
    input_strides: UnsafePointer[Scalar[DType.index]],
    owned axis_params: StaticTuple[
        _AxisParamsReflect[rank, dtype, paddings_type], rank
    ],
):
    var output_axis_stride = Int(output_strides[axis])
    var input_offset: Int
    var output_offset: Int

    @parameter
    if axis == 0:
        input_offset = 0
        output_offset = 0
        # CRITICAL: setting output_offset, input_offset, output and input pointers for the first axis in padding.
        # output_offset: The offset at which output data starts.
        # input_offset: The offset at which input data starts.
        axis_params[0].init_offsets(
            output_offset, input_offset, output_axis_stride
        )
    else:
        output_offset = axis_params[axis - 1].next_output_offset
        input_offset = axis_params[axis - 1].next_input_offset

    @parameter
    if axis == rank - 1:
        axis_params[axis].base(output_offset, input_offset, output, input)
    else:
        var output_axis_stride_next = Int(output_strides[axis + 1])
        var input_axis_stride = Int(input_strides[axis])
        for i in range(
            axis_params[axis].pre_pad,
            axis_params[axis].pre_pad + axis_params[axis].non_pad,
        ):
            axis_params[axis + 1].init_offsets(
                axis_params[axis].next_output_offset,
                axis_params[axis].next_input_offset,
                output_axis_stride_next,
            )
            _pad_reflect_axis[rank, dtype, paddings_type, axis + 1](
                output, input, output_strides, input_strides, axis_params
            )

            axis_params[axis].update_next_offsets(
                output_axis_stride, input_axis_stride
            )
    axis_params[axis].memcpy_regions(output_axis_stride, output_offset, output)


fn _pad_reflect_impl[
    rank: Int,
    dtype: DType,
    paddings_type: DType,
](
    output: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GENERIC, **_
    ],
    input: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.GENERIC, **_
    ],
    paddings: UnsafePointer[Scalar[paddings_type]],
    output_shape: IndexList[rank],
    output_strides: UnsafePointer[Scalar[DType.index]],
    input_strides: UnsafePointer[Scalar[DType.index]],
):
    """
    Fill axis ∈ [axis, rank) in `output` with values from `input`, and edges
    padded with reflected values from the unpadded region

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: The (before, after) padding sizes for each axis.
        output_shape: the shape of the tensor passed to `output`
        output_strides: the stride at each output axis.
        input_strides: the stride at each input axis.
    """

    var axis_params = StaticTuple[
        _AxisParamsReflect[rank, dtype, paddings_type], rank
    ]()

    for r in range(rank):
        axis_params[r] = _AxisParamsReflect[rank, dtype, paddings_type](
            r, paddings, output_shape
        )

    _pad_reflect_axis[rank, dtype, paddings_type, 0](
        output, input, output_strides, input_strides, axis_params
    )


@always_inline
fn pad_repeat[
    output_layout: Layout,
    input_layout: Layout,
    dtype: DType,
    paddings_type: DType,
](
    output: LayoutTensor[mut=True, dtype, output_layout, **_],
    input: LayoutTensor[dtype, input_layout, **_],
    paddings: UnsafePointer[Scalar[paddings_type]],
):
    """
    Fill `output` with values from `input`, and edges padded boundary
    values from the unpadded region.

    Parameters:
        output_layout: Layout of the output buffer.
        input_layout: Layout of the input buffer.
        dtype: DType of the input/output buffer.
        paddings_type: DType of the input, output, and padding buffers.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: Ordered (before, after) padding sizes for each axis.

    Example:
        var input = [[1, 2],
                     [3, 4]]
        var paddings = [2, 2, 1, 0]

        Yields:
        output = [[1, 1, 2],
                  [1, 1, 2],
                  [1, 1, 2],
                  [3, 3, 4],
                  [3, 3, 4],
                  [3, 3, 4]]
    """
    var output_shape = output.layout.shape
    var input_shape = input.layout.shape

    var output_strides = output.layout.stride
    var input_strides = input.layout.stride

    var pre_pads = IndexList[output_layout.rank()]()
    var post_pads = IndexList[output_layout.rank()]()

    for axis in range(output_layout.rank()):
        pre_pads[axis] = Int(paddings[2 * axis])
        post_pads[axis] = Int(paddings[2 * axis + 1])

    var loop_bounds = _NestedLoopIter[output_layout.rank()].LoopBoundSpec(
        fill=output_layout.rank()
    )

    for i in range(output_layout.rank()):
        loop_bounds[i] = IndexList[2](0, input.runtime_layout.shape.value[i])

    var non_pad_iter = _NestedLoopIter[output_layout.rank()](loop_bounds)

    for input_idx in non_pad_iter:
        var output_idx = input_idx + pre_pads
        var in_idx = Int(
            input.runtime_layout(
                RuntimeTuple[input_layout.make_shape_unknown().shape](input_idx)
            )
        )
        var out_idx = Int(
            output.runtime_layout(
                RuntimeTuple[output_layout.make_shape_unknown().shape](
                    output_idx
                )
            )
        )
        output.ptr[out_idx] = input.ptr[in_idx]

    for axis in reversed(range(output_layout.rank())):
        for i in range(axis):
            loop_bounds[i] = IndexList[2](
                pre_pads[i], pre_pads[i] + input.dim(i)
            )

        for i in range(axis + 1, output_layout.rank()):
            loop_bounds[i] = IndexList[2](0, output.dim(i))

        # handle pre-padding of the axis
        var pre_lower = 0
        var pre_upper = pre_pads[axis]

        loop_bounds[axis] = IndexList[2](pre_lower, pre_upper)

        var pre_pad_iter = _NestedLoopIter[output_layout.rank()](loop_bounds)

        for write_idx in pre_pad_iter:
            var read_idx = write_idx
            read_idx[axis] = pre_pads[axis]

            var in_idx = Int(
                output.runtime_layout(
                    RuntimeTuple[input_layout.shape](read_idx)
                )
            )

            var out_idx = Int(
                output.runtime_layout(
                    RuntimeTuple[output_layout.shape](write_idx)
                )
            )
            output.ptr[out_idx] = output.ptr[in_idx]

        # and now post-padding
        var post_lower = pre_pads[axis] + input.dim(axis)
        var post_upper = output.dim(axis)

        loop_bounds[axis] = IndexList[2](post_lower, post_upper)

        var post_pad_iter = _NestedLoopIter[output_layout.rank()](loop_bounds)

        for write_idx in post_pad_iter:
            var read_idx = write_idx
            read_idx[axis] = post_lower - 1

            var in_idx = Int(
                output.runtime_layout(
                    RuntimeTuple[output_layout.shape](read_idx)
                )
            )
            var out_idx = Int(
                output.runtime_layout(
                    RuntimeTuple[output_layout.shape](write_idx)
                )
            )
            output.ptr[out_idx] = output.ptr[in_idx]
