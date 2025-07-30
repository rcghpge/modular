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

from logger import Logger
from collections import OptionalReg
from collections.string.string_slice import get_static_string
from math import ceildiv
from sys import simdwidthof
from sys.info import _is_sm_9x_or_newer
from stdlib.bit import log2_floor
from algorithm.functional import _elementwise_impl_gpu
from buffer import Dim, NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, block_idx, thread_idx
import gpu.block
from gpu.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceContext
from gpu.host import get_gpu_target
from linalg.matmul import matmul
from linalg.utils_gpu import MatmulConfig
from runtime.tracing import trace_arg
from utils.numerics import get_accum_type
from utils.index import IndexList
from utils.numerics import max_finite, min_finite
from .utils import elementwise_epilogue_type
from gpu import global_idx
from utils.index import Index
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout import IntTuple, Layout, LayoutTensor


########################################################
# Static scaled fp8 quantization
########################################################


@always_inline
fn quantize_static_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scale_is_inverted: Bool = True,
](
    out_buffer: NDBuffer[mut=True, out_dtype, 2, *_],
    in_buffer: NDBuffer[in_dtype, 2, *_],
    scale: Float32,
    context: DeviceContext,
) raises:
    constrained[
        in_dtype in (DType.float32, DType.float16, DType.bfloat16),
        "input dtype should be float16, bfloat16 or float32",
    ]()
    constrained[
        out_dtype is DType.float8_e4m3fn, "output dtype should be float8_e4m3fn"
    ]()

    @always_inline
    @parameter
    @__copy_capture(out_buffer, in_buffer, scale)
    fn scaled_fp8_quant[width: Int, rank: Int](idx_arg: IndexList[rank]):
        constrained[
            _is_sm_9x_or_newer(),
            "this kernel is only supported on sm90 or newer",
        ]()
        constrained[rank == 2, "rank should be equal to 2"]()

        var idx = rebind[IndexList[2]](idx_arg)
        var in_vec_f32 = in_buffer.load[width=width](idx).cast[DType.float32]()

        var inversed_scale: Float32 = 1.0 / scale

        @parameter
        for i in range(width):
            var scaled_input_f32: Float32

            scaled_input_f32 = in_vec_f32[i] * inversed_scale
            in_vec_f32[i] = max(
                Float32(min_finite[out_dtype]()),
                min(Float32(max_finite[out_dtype]()), scaled_input_f32),
            )

        var scaled_in_vec = in_vec_f32.cast[out_dtype]()
        out_buffer.store(idx, rebind[SIMD[out_dtype, width]](scaled_in_vec))

    alias target_simd_width = simdwidthof[in_dtype, target = get_gpu_target()]()

    _elementwise_impl_gpu[func=scaled_fp8_quant, simd_width=target_simd_width](
        IndexList[2](in_buffer.dim[0](), in_buffer.dim[1]()), context
    )


########################################################
# dynamic scaled fp8 quantization
########################################################


@always_inline
fn quantize_dynamic_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType, //,
    group_size_or_per_token: Int,
](
    scaled_output: NDBuffer[mut=True, out_dtype, 2, *_],
    scales: NDBuffer[mut=True, scales_dtype, 2, *_],
    input: NDBuffer[in_dtype, 2, *_],
    scale_ub: Float32,
    ctx: DeviceContext,
) raises:
    constrained[
        scales_dtype in (DType.bfloat16, DType.float16, DType.float32),
        "scales dtype should be bfloat16, float16 or float32",
    ]()
    constrained[
        out_dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz),
        "output dtype should be float8_e4m3fn or float8_e4m3fnuz",
    ]()

    alias group_size = input.shape.get[
        1
    ]() if group_size_or_per_token == -1 else group_size_or_per_token
    alias n_groups = input.shape.get[1]() // group_size
    alias simd_width = simdwidthof[in_dtype, target = get_gpu_target()]()
    alias max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    alias warps_per_block = min(
        ceildiv(group_size // simd_width, WARP_SIZE), max_warps_per_block
    )

    alias kernel = quantize_fp8_kernel[
        out_dtype,
        scales_dtype,
        in_dtype,
        warps_per_block,
        group_size,
    ]

    ctx.enqueue_function[kernel](
        scaled_output,
        scales,
        input,
        scale_ub,
        grid_dim=(input.dim[0](), n_groups, 1),
        block_dim=warps_per_block * WARP_SIZE,
        attributes=pdl_launch_attributes(),
    )


fn quantize_fp8_kernel[
    out_type: DType,
    scales_type: DType,
    in_type: DType,
    warps_per_block: Int,
    group_size: Int,
](
    output: NDBuffer[mut=True, out_type, 2, MutableAnyOrigin],
    scales: NDBuffer[mut=True, scales_type, 2, MutableAnyOrigin],
    input: NDBuffer[in_type, 2, MutableAnyOrigin],
    scale_ub: Scalar[scales_type],
):
    alias simd_width = simdwidthof[in_type]()
    alias num_threads = warps_per_block * WARP_SIZE
    alias use_warp_tiling = group_size <= num_threads * simd_width
    alias fp8_max = Scalar[out_type].MAX_FINITE

    var input_vec = SIMD[in_type, simd_width](0)
    var thread_max = Scalar[in_type](0)

    var tid = thread_idx.x
    var row = Int(block_idx.x)
    var group_idx = Int(block_idx.y)

    with PDL():
        for i in range(tid, group_size // simd_width, num_threads):
            var idx: Int = i * simd_width + group_idx * group_size
            input_vec = input.load[width=simd_width](row, idx)
            thread_max = max(thread_max, abs(input_vec).reduce_max())

        var group_max = block.max[block_size=num_threads, broadcast=True](
            thread_max
        )

        var scale_factor = (
            max(group_max.cast[scales_type](), scale_ub)
            / fp8_max.cast[scales_type]()
        )

        if tid == 0:
            scales.store[width=1](IndexList[2](row, group_idx), scale_factor)

        for i in range(tid, group_size // simd_width, num_threads):
            var idx: Int = i * simd_width + group_idx * group_size

            @parameter
            if use_warp_tiling:
                pass
            else:
                input_vec = input.load[width=simd_width](row, idx)

            var output_vec = input_vec.cast[scales_type]() / scale_factor

            output_vec = max(
                SIMD[scales_type, simd_width](-fp8_max),
                min(SIMD[scales_type, simd_width](fp8_max), output_vec),
            )
            output.store[width=simd_width](
                IndexList[2](row, idx), output_vec.cast[out_type]()
            )


########################################################
# scaled fp8 matmul
########################################################


@always_inline
fn matmul_dynamic_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType, //,
    transpose_b: Bool = False,
    config: OptionalReg[
        MatmulConfig[a_type, b_type, c_type, transpose_b]
    ] = None,
    target: StaticString = "cpu",
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    a_scales: NDBuffer[a_scales_type, 2, _, _],
    b_scales: NDBuffer[b_scales_type, 2, _, _],
    ctx: DeviceContext,
) raises:
    constrained[
        a_type in (DType.float8_e4m3fn, DType.float8_e4m3fnuz),
        "input A dtype should be float8_e4m3fn, float8_e4m3fnuz",
    ]()
    constrained[
        b_type in (DType.float8_e4m3fn, DType.float8_e4m3fnuz),
        "input B dtype should be float8_e4m3fn, float8_e4m3fnuz",
    ]()
    constrained[
        a_scales_type in (DType.bfloat16, DType.float16, DType.float32),
        "input A scales dtype should be bfloat16, float16 or float32",
    ]()
    constrained[
        b_scales_type in (DType.bfloat16, DType.float16, DType.float32),
        "input B scales dtype should be bfloat16, float16 or float32",
    ]()

    alias b_k_axis = 1 if transpose_b else 0
    alias b_row_axis = 0 if transpose_b else 1
    alias N = b.shape.get[b_row_axis]()
    var M = a.dim[0]()

    alias _trace_string = get_static_string[
        trace_arg(
            "A_scales",
            IndexList[2](a_scales.shape.get[0](), a_scales.shape.get[1]()),
            a_scales.type,
        ),
        ";",
        trace_arg(
            "B_scales",
            IndexList[2](b_scales.shape.get[0](), b_scales.shape.get[1]()),
            b_scales.type,
        ),
    ]()

    # create a dummy buffer to instruct the matmul kernel to output values
    # in the correct dtype
    var c_dummy = NDBuffer[
        DType.float32, 2, MutableAnyOrigin, DimList(Dim(), N)
    ](
        UnsafePointer[Scalar[DType.float32]](),
        IndexList[2](M, N),
    )

    @parameter
    @__copy_capture(c, a, b, a_scales, b_scales)
    @always_inline
    fn scaled_output_fn[
        dtype: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        var a_scale = a_scales.load[width=1](idx[0], 0).cast[dtype]()
        var b_scale: SIMD[dtype, width]

        @parameter
        if transpose_b:
            b_scale = b_scales.load[width=width](idx[1], 0).cast[dtype]()
        else:
            b_scale = b_scales.load[width=width](0, idx[1]).cast[dtype]()

        var scaled_val = val * a_scale * b_scale

        c.store[width=width, alignment=alignment](
            idx, scaled_val.cast[c_type]()
        )

    matmul[
        target=target,
        transpose_b=transpose_b,
        elementwise_lambda_fn=scaled_output_fn,
        _trace_description=_trace_string,
    ](c_dummy, a, b, Optional[DeviceContext](ctx))


fn naive_blockwise_scaled_fp8_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_type: DType,
    c_shape: DimList,
    a_shape: DimList,
    b_shape: DimList,
    a_scale_shape: DimList,
    b_scale_shape: DimList, //,
    *,
    BLOCK_DIM: Int = 16,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    s_type: DType = get_accum_type[c_type](),
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    a_scales_device: NDBuffer[scales_type, 2, _, a_scale_shape],
    b_scales_device: NDBuffer[scales_type, 2, _, b_scale_shape],
    ctx: DeviceContext,
) raises:
    constrained[
        a_type == b_type == DType.float8_e4m3fn,
        (
            "Only float8_e4m3fn is supported for input dtype for blockwise"
            " scaled fp8 matmul"
        ),
    ]()

    constrained[
        s_type == DType.float32,
        "Only float32 is supported for accumulation for scaled matmul",
    ]()

    var logger = Logger()
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)
    var a_scales = from_ndbuffer_row_major(a_scales_device)
    var b_scales = from_ndbuffer_row_major(b_scales_device)

    var M = c_device.dim[0]()
    var N = c_device.dim[1]()
    var K = a_device.dim[1]()

    logger.info("Executing Naive Blockwise Scaled FP8 GEMM")
    logger.info("Problem Shape: MNK=[", M, ", ", N, ", ", K, "]")
    logger.info(
        "A Scales Shape: [", a_scales.dim[0](), ", ", a_scales.dim[1](), "]"
    )
    logger.info(
        "B Scales Shape: [", b_scales.dim[0](), ", ", b_scales.dim[1](), "]"
    )

    alias kernel = naive_blockwise_scaled_fp8_matmul_kernel[
        c_type,
        a_type,
        b_type,
        scales_type,
        s_type,
        __type_of(a).layout,
        __type_of(b).layout,
        __type_of(c).layout,
        __type_of(a_scales).layout,
        __type_of(b_scales).layout,
        BLOCK_DIM,
        transpose_b,
        elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )


fn naive_blockwise_scaled_fp8_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_type: DType,
    s_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_scale_layout: Layout,
    b_scale_layout: Layout,
    BLOCK_DIM: Int,
    transpose_b: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    a_scales: LayoutTensor[scales_type, a_scale_layout, MutableAnyOrigin],
    b_scales: LayoutTensor[scales_type, b_scale_layout, MutableAnyOrigin],
):
    constrained[
        s_type == DType.float32,
        "Only float32 is supported for accumulation for scaled matmul",
    ]()

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    var x = global_idx.x
    var y = global_idx.y

    if x >= M or y >= N:
        return

    var a_scale_0 = a_scales.dim[0]()
    var a_scale_1 = a_scales.dim[1]()
    var b_scale_0 = b_scales.dim[0]()
    var b_scale_1 = b_scales.dim[1]()
    var MAT_A_ROWS_SCALE_SIZE = M // a_scale_0
    var MAT_A_COLS_SCALE_SIZE = K // a_scale_1
    var MAT_B_ROWS_SCALE_SIZE = (
        N // b_scale_0 if transpose_b else K // b_scale_0
    )
    var MAT_B_COLS_SCALE_SIZE = (
        K // b_scale_1 if transpose_b else N // b_scale_1
    )

    var accum = Scalar[s_type](0)
    for k in range(K):
        var a_val = rebind[Scalar[a_type]](a[x, k]).cast[s_type]()
        var a_scale_factor = rebind[Scalar[s_type]](
            a_scales[x // MAT_A_ROWS_SCALE_SIZE, k // MAT_A_COLS_SCALE_SIZE]
        )

        var b_val: Scalar[s_type]
        var b_scale_factor: Scalar[s_type]

        @parameter
        if transpose_b:
            b_val = rebind[Scalar[b_type]](b[y, k]).cast[s_type]()
            b_scale_factor = rebind[Scalar[s_type]](
                b_scales[y // MAT_B_ROWS_SCALE_SIZE, k // MAT_B_COLS_SCALE_SIZE]
            )
        else:
            b_val = rebind[Scalar[b_type]](b[k, y]).cast[s_type]()
            b_scale_factor = rebind[Scalar[s_type]](
                b_scales[k // MAT_B_ROWS_SCALE_SIZE, y // MAT_B_COLS_SCALE_SIZE]
            )

        accum += a_val * b_val * a_scale_factor * b_scale_factor

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](Index(x, y), accum.cast[c_type]())
    else:
        c[x, y] = accum.cast[c_type]()
