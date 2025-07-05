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
# General imports
# ===-----------------------------------------------------------------------===#

from collections import OptionalReg
from math import (
    atanh,
    ceil,
    cos,
    erf,
    exp,
    floor,
    fma,
    iota,
    isqrt,
    log,
    log1p,
    sin,
    sqrt,
    tanh,
)
from random import randn, seed
from sys import bitwidthof, external_call, llvm_intrinsic
from sys.info import simdwidthof, sizeof
from sys.intrinsics import _type_is_eq

import compiler_internal as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from algorithm import max as reduce_max
from algorithm import mean
from algorithm import min as reduce_min
from algorithm import product, sum
from algorithm.reduction import _reduce_generator, _reduce_generator_cpu
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from builtin.simd import _pow
from compiler_internal import StaticTensorSpec
from gpu.comm.allgather import allgather
from gpu.comm.allreduce import MAX_GPUS, Signal, allreduce
from gpu.host import DeviceContext
from gpu.host.info import is_cpu, is_gpu, is_valid_target
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    KVCollectionT,
    PagedKVCacheCollection,
)
from layout.layout_tensor import Layout, LayoutTensor, RuntimeLayout
from linalg.bmm import batched_matmul, batched_matmul_shape
from linalg.distributed_matmul import matmul_allreduce
from linalg.bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)
from linalg.dual_gemm import swishGLU
from linalg.fp8_quantization import (
    matmul_dynamic_scaled_fp8,
    quantize_dynamic_scaled_fp8,
    quantize_static_scaled_fp8,
)
from linalg.grouped_matmul import grouped_matmul
from linalg.matmul import matmul
from linalg.matrix_band_part import matrix_band_part
from linalg.packing import _pack_b_ndbuffer_impl, pack_matmul_b_shape_func
from linalg.utils import (
    elementwise_compute_lambda_type as matmul_elementwise_compute_lambda_type,
)
from linalg.utils import (
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)
from nn import arg_nonzero
from nn._ragged_utils import merge_ragged_tensors
from nn.activations import gelu, relu
from nn.arange import arange_shape
from nn.argmaxmin import argmax, argmin
from nn.argmaxmin_gpu import argmax_gpu, argmin_gpu
from nn.argsort import argsort
from nn.concat import _concat_cpu, concat, fused_concat
from nn.conv import ConvInfoStatic, conv_gpu, conv_nhwc_direct, conv_shape
from nn.conv import pack_filter as _pack_conv_filter
from nn.conv import pack_filter_shape as pack_filter_shape_conv
from nn.conv_transpose import (
    conv_transpose_shape,
    conv_transposed_cpu,
    conv_transposed_gpu,
)
from nn.conv_transpose import pack_filter as _pack_conv_transpose_filter
from nn.conv_transpose import (
    pack_filter_shape as pack_filter_shape_conv_transpose,
)
from nn.conv_utils import elementwise_simd_epilogue_type
from nn.cumsum import cumsum
from nn.flash_attention import flash_attention as nn_flash_attention
from nn.flash_attention import flash_attention_split_kv
from nn.fold import fold, fold_shape
from nn.gather_scatter import (
    Axis,
    _unsafe_normalize_neg_index,
    gather,
    gather_nd,
    gather_nd_shape,
    gather_reduce,
    gather_shape,
    normalize_neg_index,
    scatter_elements,
    scatter_elements_shape,
    scatter_nd,
    scatter_nd_generator,
    scatter_nd_shape,
    scatter_set_constant,
)
from nn.index_tensor import (
    advanced_indexing_getitem,
    advanced_indexing_getitem_shape,
    advanced_indexing_setitem_inplace,
    index_tensor,
)
from nn.irfft import irfft
from nn.kv_cache import (
    generic_flash_attention_kv_cache_padded,
    generic_flash_attention_kv_cache_padded_materialized_mask,
    generic_fused_qk_rope_bshd_continuous_batch,
    generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch,
    generic_get_continuous_cache,
    generic_get_paged_cache,
    print_kv_cache_cont_batch_generic_cpu,
    print_kv_cache_cont_batch_generic_gpu,
    print_kv_cache_paged_generic_cpu,
    print_kv_cache_paged_generic_gpu,
    rms_norm_kv_cache_ragged_continuous_batching,
    rms_norm_kv_cache_ragged_paged,
)
from nn.kv_cache_ragged import (
    generic_cross_attention_kv_cache,
    generic_flare_mla_decode_kv_cache_ragged,
    generic_flare_mla_decompress_k_cache_ragged_paged,
    generic_flare_mla_prefill_kv_cache_ragged,
    generic_flare_mla_prefill_ragged_paged_plan,
    generic_flash_attention_kv_cache_ragged,
    generic_fused_qk_rope_bshd_continuous_batch_ragged,
    generic_fused_qk_rope_bshd_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_cont_batch_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias,
    generic_fused_qkv_matmul_kv_cache_paged_ragged_scale,
    k_matmul_ragged_paged,
    kv_matmul_ragged_paged,
    unfused_qkv_matmul_ragged_paged_gguf_quantized,
)
from nn.mha import flash_attention
from nn.mha_mask import MHAMask
from nn.mha_score_mod import IdentityScoreMod, ScoreModTrait
from nn.mha_utils import dispatch_mask_and_score_mod
from nn.moe import moe_create_indices
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.normalization import group_norm, layer_norm, rms_norm
from nn.pad import pad_constant, pad_reflect, pad_repeat, pad_shape
from nn.pad_gpu import pad_constant as pad_constant_gpu
from nn.pool import avg_pool, max_pool, pool_shape, pool_shape_ceil
from nn.rand_uniform import random_uniform
from nn.repeat_interleave import repeat_interleave, repeat_interleave_shape
from nn.reshape import reshape, reshape_shape
from nn.resize import resize_linear, resize_nearest_neighbor

from nn.bicubic import resize_bicubic
from nn.roi_align import roi_align_nhwc
from nn.sampling import apply_penalties_to_logits, update_frequency_data
from nn.slice import (
    copy_to_slice,
    slice_as_view,
    slice_dim_as_view,
    slice_shape,
)
from nn.softmax import logsoftmax, softmax
from nn.split import split
from nn.tile import tile, tile_shape
from nn.topk import top_k
from nn.topk import fused_token_sampling_cpu as _fused_token_sampling_cpu
from nn.topk import top_k_shape_impl
from nn.topk import fused_token_sampling_gpu as _fused_token_sampling_gpu
from nn.toppminp import min_p_sampling as min_p_sampling_cpu
from nn.toppminp_gpu import min_p_sampling_gpu
from quantization import (
    Q4sym,
    block_Q4_K,
    block_Q6_K,
    block_QK_K,
    q4_k_dequantize_impl,
    q6_k_dequantize_impl,
)
from quantization.qmatmul import matmul_qint4, matmul_qint4_pack_b
from quantization.qmatmul_gpu import (
    gpu_qint4_repack_GPTQ,
    gpu_qint4_repack_Q4_0,
    matmul_gpu_qint4,
)
from quantization.qmatmul_k import (
    matmul_Q4_K,
    matmul_Q4_K_pack_b,
    matmul_Q6_K,
    matmul_Q6_K_pack_b,
)
from register import register_internal
from runtime.asyncrt import DeviceContextPtr, DeviceContextPtrList
from runtime.tracing import Trace, TraceLevel
from tensor_internal import (
    DynamicTensor,
    InputTensor,
    InputVariadicTensors,
    IOSpec,
    IOUnknown,
    ManagedTensorSlice,
    OutputTensor,
    OutputVariadicTensors,
    VariadicTensors,
    _input_fusion_hook_impl,
    _mixed_precision_input_fusion_hook_impl,
    _mixed_precision_output_fusion_hook_impl,
    _mixed_precision_compute_output_fusion_hook_impl,
    _output_fusion_hook_impl,
    foreach,
    simd_load_from_managed_tensor_slice,
    simd_store_into_managed_tensor_slice,
    view_copy_impl,
)
from tensor_internal._indexing import _dot_prod, _row_major_strides
from tensor_internal.io_spec import IO
from tensor_internal.managed_tensor_slice import _FusedComputeOutputTensor
from tensor_internal.managed_tensor_slice import (
    _FusedInputTensor as FusedInputTensor,
)
from tensor_internal.managed_tensor_slice import (
    _FusedInputVariadicTensors as FusedInputVariadicTensors,
)
from tensor_internal.managed_tensor_slice import (
    _FusedOutputTensor as FusedOutputTensor,
)
from tensor_internal.managed_tensor_slice import (
    _FusedOutputVariadicTensors as FusedOutputVariadicTensors,
)
from tensor_internal.managed_tensor_slice import (
    _MutableInputTensor as MutableInputTensor,
)
from tensor_internal.managed_tensor_slice import (
    _MutableInputVariadicTensors as MutableInputVariadicTensors,
)
from tensor_internal.managed_tensor_slice import get_kernel_simd_width
from tensor_internal.transitional import managed_tensor_slice_to_ndbuffer

from utils import IndexList, StaticTuple
from utils.index import Index
from utils.numerics import isinf, isnan
from utils.static_tuple import _create_array, _set_array_elem

# ===-----------------------------------------------------------------------===#
# Nop functions to expose different types to the compiler.
# ===-----------------------------------------------------------------------===#


@register_internal("float8_e5m2")
fn DTypeFloat8E5M2TypeDef(ty: DType.type) -> DType.type:
    return DType.float8_e5m2.value


@register_internal("float8_e5m2fnuz")
fn DTypeFloat8E5M2FnuzTypeDef(ty: DType.type) -> DType.type:
    return DType.float8_e5m2fnuz.value


@register_internal("float8_e3m4")
fn DTypeFloat8E3M4TypeDef(ty: DType.type) -> DType.type:
    return DType.float8_e3m4.value


@register_internal("float8_e4m3fn")
fn DTypeFloat8E4M3FnTypeDef(ty: DType.type) -> DType.type:
    return DType.float8_e4m3fn.value


@register_internal("float8_e4m3fnuz")
fn DTypeFloat8E4M3FnuzTypeDef(ty: DType.type) -> DType.type:
    return DType.float8_e4m3fnuz.value


@register_internal("bfloat16")
fn DTypeBFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.bfloat16.value


@register_internal("float16")
fn DTypeFloat16TypeDef(ty: DType.type) -> DType.type:
    return DType.float16.value


@register_internal("float32")
fn DTypeFloat32TypeDef(ty: DType.type) -> DType.type:
    return DType.float32.value


@register_internal("float64")
fn DTypeFloat64TypeDef(ty: DType.type) -> DType.type:
    return DType.float64.value


@register_internal("int8")
fn DTypeInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.int8.value


@register_internal("int16")
fn DTypeInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.int16.value


@register_internal("int32")
fn DTypeInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.int32.value


@register_internal("uint32")
fn DTypeUInt32TypeDef(ty: DType.type) -> DType.type:
    return DType.uint32.value


@register_internal("uint64")
fn DTypeUInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.uint64.value


@register_internal("int64")
fn DTypeInt64TypeDef(ty: DType.type) -> DType.type:
    return DType.int64.value


@register_internal("uint8")
fn DTypeUInt8TypeDef(ty: DType.type) -> DType.type:
    return DType.uint8.value


@register_internal("uint16")
fn DTypeUInt16TypeDef(ty: DType.type) -> DType.type:
    return DType.uint16.value


@register_internal("bool")
fn DTypeBoolTypeDef(ty: DType.type) -> DType.type:
    return DType.bool.value


@register_internal("index")
fn IndexTypeDef(ty: Int) -> Int:
    return ty


@register_internal("deviceContext")
fn DeviceContextDef(ty: DeviceContextPtr):
    pass


@register_internal("simd")
fn SimdTypeDef[
    dtype: DType, width: Int
](ty: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return ty


@register_internal("indices")
fn TensorIndicesTypeDef[rank: Int](ty: IndexList[rank]) -> IndexList[rank]:
    return ty


@register_internal("dim_type")
fn DimTypeDef(ty: Dim) -> Dim:
    return ty


@register_internal("managed_tensor_slice")
fn ManagedTensorSliceDef[
    mut: Bool,
    input: IO,
    dtype: DType,
    rank: Int, //,
    io_spec: IOSpec[mut, input],
    static_spec: StaticTensorSpec[dtype, rank],
](
    ty: ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]
) -> ManagedTensorSlice[io_spec=io_spec, static_spec=static_spec]:
    return ty


@register_internal("list_of_tensor")
fn ListOfTensorDef[
    dtype: DType,
    rank: Int,
](
    ty: List[
        InputTensor[
            static_spec = StaticTensorSpec[dtype, rank].create_unknown()
        ]
    ]
) -> __type_of(ty):
    return ty


# ===-----------------------------------------------------------------------===#
# Hooks to help build static shapes.
# ===-----------------------------------------------------------------------===#


@register_internal("create_unknown_dim")
fn create_unknown_dim() -> Dim:
    return Dim()


@register_internal("create_known_dim")
fn create_known_dim[known_val: Int]() -> Dim:
    return Dim(known_val)


@register_internal("reshape_contiguous_managed_tensor_slice")
@always_inline
fn reshape_contiguous_buffer[
    dtype: DType, old_rank: Int, new_rank: Int, mut: Bool, input: IO
](
    buffer: ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, old_rank].create_unknown(),
    ],
    shape: IndexList[new_rank],
) -> DynamicTensor[dtype, new_rank]:
    return DynamicTensor[dtype, new_rank](buffer._ptr, shape)


# ===----------------------------------------------------------------------===#
# Additional expected primitives
# ===-----------------------------------------------------------------------===#


@register_internal("get_simd_width_for_dtypes")
@always_inline
fn get_simd_width_for_dtypes[
    dtypes: StaticTuple[DType], target: StaticString
]() -> Int:
    constrained[dtypes.size > 0]()

    var width = get_kernel_simd_width[dtypes[0], target]()

    @parameter
    for i in range(dtypes.size - 1):
        width = max(get_kernel_simd_width[dtypes[i + 1], target](), width)

    return width


@register_internal("get_address_space")
fn get_address_space() -> AddressSpace:
    return AddressSpace.GENERIC


# Build the StaticTensorSpec parameter for the DPS kernels
@register_internal("build_static_tensor_specs")
fn build_static_tensor_specs[
    dtype: DType,
    rank: Int,
](
    shape: DimList,
    strides: DimList,
    alignment: Int,
    address_space: AddressSpace,
    exclusive: Bool,
) -> StaticTensorSpec[dtype, rank]:
    alias SpecType = StaticTensorSpec[dtype, rank]

    return SpecType(
        shape, strides, alignment, address_space, exclusive, None, None, None
    )


# Build the tuple of StaticTensorSpecs for DPS kernels
@register_internal("build_static_tensor_specs_tuple")
fn build_static_tensor_specs_tuple[
    dtype: DType,
    rank: Int,
    size: Int,
](
    array_of_specs: VariadicList[StaticTensorSpec[dtype, rank]],
    out result: StaticTuple[StaticTensorSpec[dtype, rank], size],
):
    return __type_of(result)(array_of_specs)


# TODO: this should take IOSpec as a param -- will require graph compiler changes
# Used by the graph compiler to construct tensors from MGP repr. of tensor
@register_internal("to_managed_tensor_slice")
@always_inline
fn to_managed_tensor_slice[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    data: UnsafePointer[Scalar[dtype]],
    shape: UnsafePointer[Int],
) -> ManagedTensorSlice[
    io_spec = IOSpec[mut, input](),
    static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
]:
    var shape_ptr = shape
    var shape_tuple = IndexList[rank]()

    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    @parameter
    for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        shape_tuple[i] = shape_ptr[i]
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
    ](data, shape_tuple, stride_tuple)


@always_inline
fn _to_managed_tensor_slice_index_list_shape[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    data: UnsafePointer[Scalar[dtype]],
    shape_tuple: IndexList[rank],
) -> ManagedTensorSlice[
    io_spec = IOSpec[mut, input](),
    static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
]:
    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    @parameter
    for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
    ](data, shape_tuple, stride_tuple)


@always_inline
fn _get_scalar_from_managed_tensor_slice[
    dtype: DType,
](tensor: ManagedTensorSlice[dtype=dtype]) -> Scalar[dtype]:
    # Assumes that tensor is on the host!
    # This is used instead of [0] since __getitem__ for `ManagedTesnorSlice`
    # does not work with `register_internal` out of the box.
    return tensor.load[width=1](IndexList[1](0))


# Extract a scalar from a managed tensor slice.
@register_internal("get_scalar_from_managed_tensor_slice")
@always_inline
fn get_scalar_from_managed_tensor_slice[
    dtype: DType, mut: Bool, input: IO
](
    tensor: ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, 1].create_unknown(),
    ]
) -> Scalar[dtype]:
    return _get_scalar_from_managed_tensor_slice(tensor)


@always_inline("nodebug")
fn _int_bitwidth_safety_check[simd_dtype: DType]():
    constrained[
        bitwidthof[DType.index]() >= bitwidthof[simd_dtype](),
        String(
            (
                "A kernel was specified with an 'Int' but the type of the"
                " corresponding value in the graph op was '"
            ),
            simd_dtype,
            "' a fixed size integer with a width greater than Int's",
        ),
    ]()


@register_internal("get_int_from_shape")
@always_inline
fn get_int_from_shape[
    param_index: Int, rank: Int
](shape: IndexList[rank]) -> Int:
    return shape[param_index]


@register_internal("rebuild_static_tensor_specs_with_output_compute_lambda")
@no_inline
fn rebuild_static_tensor_specs_with_output_compute_lambda[
    func_type: AnyTrivialRegType, //,
    dtype: DType,
    rank: Int,
](
    spec: StaticTensorSpec[dtype, rank],
    out_compute_lambda: func_type,
) -> StaticTensorSpec[dtype, rank]:
    return StaticTensorSpec[dtype, rank](
        shape=spec.shape,
        strides=spec.strides,
        alignment=spec.alignment,
        address_space=spec.address_space,
        exclusive=spec.exclusive,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=rebind[spec.out_compute_lambda_t](
            out_compute_lambda
        ),
    )


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


@always_inline
fn input_variadic_tensors_to_static_tuple_ndbuffer[
    dtype: DType, rank: Int, size: Int
](indices: InputVariadicTensors[dtype, rank, size=size]) -> StaticTuple[
    NDBuffer[dtype, rank, MutableAnyOrigin], size
]:
    var result = StaticTuple[NDBuffer[dtype, rank, MutableAnyOrigin], size]()

    @parameter
    for i in range(size):
        result[i] = managed_tensor_slice_to_ndbuffer(indices[i])
    return result


@always_inline("nodebug")
fn reduce_shape[
    input_rank: Int, input_type: DType, //
](
    input_buf: ManagedTensorSlice[dtype=input_type, rank=input_rank],
    axis: Int,
) raises -> IndexList[input_rank]:
    """
    Compute the output shape of a `reduce` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Input_rank of the input tensor.
        input_type: Type of the input tensor.

    Args:
        input_buf: The input tensor.
        axis: The axis tensor.

    Returns:
        The output shape.
    """

    # compute and return the output shape
    var output_shape = input_buf.shape()
    output_shape[normalize_neg_index(axis, input_rank)] = 1
    return output_shape


# ===----------------------------------------------------------------------===#
# Helpers for Affine Fusion
# ===----------------------------------------------------------------------===#


@register_internal("split_dim_indices")
@always_inline
fn split_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], new_shape_dim: Int64) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [40, 30, 2] and we reshape it to [5, 8, 30, 2].
    # If we are accessing the index [21, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [2, 5, 16, 1].
    # Or [21 // 8, 21 % 8, ...old dims...].
    # In this case, the axis = 0 and the new_shape_dim = 8.

    @parameter
    for i in range(rank + 1):

        @parameter
        if i == axis:
            out[i] = indices[axis] // Int(new_shape_dim)
        elif i == axis + 1:
            out[i] = indices[axis] % Int(new_shape_dim)
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]

    return out


@register_internal("merge_dim_indices")
@always_inline
fn merge_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], old_shape_dim: Int64) -> IndexList[rank - 1]:
    var out = IndexList[rank - 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [5, 8, 30, 2] and we reshape it to [40, 30, 2].
    # If we are accessing the index [2, 5, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [21, 16, 1].
    # Or [2 * 8 + 5, 16, 1].
    # In this case, the axis = 0 and the old_shape_dim = 8.

    @parameter
    for i in range(rank - 1):

        @parameter
        if i == axis:
            out[i] = fma(indices[i], Int(old_shape_dim), indices[i + 1])
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i + 1]

    return out


@register_internal("insert_index")
@always_inline
fn insert_index[
    rank: Int, axis: Int, value: Int
](indices: IndexList[rank]) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    @parameter
    for i in range(rank + 1):

        @parameter
        if i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]
        else:
            out[i] = value

    return out


# TODO(MOCO-1413): remove this need to keep imported exported funcs alive.
@export
fn export():
    alias _simd_load_from_managed_tensor_slice = simd_load_from_managed_tensor_slice
    alias _simd_store_into_managed_tensor_slice = simd_store_into_managed_tensor_slice
    alias __input_fusion_hook_impl = _input_fusion_hook_impl
    alias __output_fusion_hook_impl = _output_fusion_hook_impl
    alias __mixed_precision_input_fusion_hook_impl = _mixed_precision_input_fusion_hook_impl
    alias __mixed_precision_output_fusion_hook_impl = _mixed_precision_output_fusion_hook_impl
    alias __mixed_precision_compute_output_fusion_hook_impl = _mixed_precision_compute_output_fusion_hook_impl


# ===-----------------------------------------------------------------------===#
# Elementwise Kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.range")
struct Range:
    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=1],
        start: Scalar[dtype],
        stop: Scalar[dtype],
        step: Scalar[dtype],
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[1]) -> SIMD[dtype, width]:
            return start + step * (iota[dtype, width](idx[0]))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](output, ctx)

    @staticmethod
    fn shape[
        dtype: DType
    ](
        start: Scalar[dtype],
        stop: Scalar[dtype],
        step: Scalar[dtype],
    ) raises -> IndexList[1]:
        return arange_shape[single_thread_blocking_override=True](
            start,
            stop,
            step,
        )


# ===-----------------------------------------------------------------------===#
# Binary Elementwise Kernels
# ===-----------------------------------------------------------------------===#


# useful for testing --> identity op that simply copies input into output
@compiler.register("copy")
struct Copy:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank],
        input: FusedInputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[rank]) -> SIMD[dtype, width]:
            return input._fused_load[width](idx)

        foreach[func](output, ctx)


@compiler.register("mo.add")
struct Add:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))
            return lhs + rhs

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.sub")
struct Sub:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))
            return lhs - rhs

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.mul")
struct Mul:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))
            return lhs * rhs

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.div")
struct Div:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))
            return lhs / rhs

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.mod")
struct Mod:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))
            return lhs % rhs

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.equal")
struct Equal:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[x.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.dtype, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.dtype, width]](lhs == rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.greater")
struct Greater:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[x.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.dtype, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.dtype, width]](lhs > rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.greater_equal")
struct GreaterEqual:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[x.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.dtype, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.dtype, width]](lhs >= rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.not_equal")
struct NotEqual:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[x.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.dtype, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.dtype, width]](lhs != rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.and")
struct And:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.dtype, width]](lhs & rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.or")
struct Or:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.dtype, width]](lhs | rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.xor")
struct Xor:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.dtype, width]](lhs ^ rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.pow")
struct Pow:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))
            var rhs = y._fused_load[width](idx)
            return _pow(lhs, rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.max")
struct Max:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))
            return max(lhs, rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


@compiler.register("mo.min")
struct Min:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        z: FusedOutputTensor,
        x: FusedInputTensor,
        y: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[z.rank]) -> SIMD[z.dtype, width]:
            var lhs = rebind[SIMD[z.dtype, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.dtype, width]](y._fused_load[width](idx))
            return min(lhs, rhs)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](z, ctx)


# ===-----------------------------------------------------------------------===#
# Unary Elementwise Kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.cast")
struct Cast:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            var answer = x._fused_load[width](idx).cast[y.dtype]()
            return rebind[SIMD[y.dtype, width]](answer)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.negative")
struct Negative:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](-x._fused_load[width](idx))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.relu")
struct ReLU:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](relu(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.gelu")
struct GeLU:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](gelu(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.ceil")
struct Ceil:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](ceil(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.floor")
struct Floor:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](
                floor(x._fused_load[width](idx))
            )

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.tanh")
struct Tanh:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](tanh(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.atanh")
struct ATanh:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](
                atanh(x._fused_load[width](idx))
            )

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.cos")
struct Cos:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](cos(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.sin")
struct Sin:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](sin(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.erf")
struct Erf:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](erf(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.exp")
struct Exp:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](exp(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.round")
struct Round:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](
                round(x._fused_load[width](idx))
            )

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.sqrt")
struct Sqrt:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](sqrt(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.isqrt")
struct Isqrt:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](
                isqrt(x._fused_load[width](idx))
            )

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.select")
struct Select:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        condition: FusedInputTensor,
        true_case: FusedInputTensor,
        false_case: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[output.rank]) -> SIMD[output.dtype, width]:
            var cond = condition._fused_load[width](idx)
            var tc = rebind[SIMD[output.dtype, width]](
                true_case._fused_load[width](idx)
            )
            var fc = rebind[SIMD[output.dtype, width]](
                false_case._fused_load[width](idx)
            )
            return cond.select(tc, fc)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](output, ctx)


@compiler.register("mo.trunc")
struct Trunc:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            var val = x._fused_load[width](idx)
            return rebind[SIMD[y.dtype, width]](
                llvm_intrinsic[
                    "llvm.trunc", __type_of(val), has_side_effect=False
                ](val)
            )

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.log")
struct Log:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](log(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.log1p")
struct Log1p:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](
                log1p(x._fused_load[width](idx))
            )

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.is_nan")
struct IsNan:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](
                isnan(x._fused_load[width](idx))
            )

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.is_inf")
struct IsInf:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](
                isinf(x._fused_load[width](idx))
            )

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.not")
struct Not:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            var val = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            return rebind[SIMD[y.dtype, width]](~val)

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.abs")
struct Abs:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        y: FusedOutputTensor, x: FusedInputTensor, ctx: DeviceContextPtr
    ) capturing raises:
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[y.rank]) -> SIMD[y.dtype, width]:
            return rebind[SIMD[y.dtype, width]](abs(x._fused_load[width](idx)))

        foreach[
            func,
            target=target,
            _trace_name=_trace_name,
        ](y, ctx)


@compiler.register("mo.squeeze_shape")
struct SqueezeShape:
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType,
        indices_type: DType,
    ](
        output_shape: OutputTensor[dtype=dtype, rank=1],
        input_shape: InputTensor[dtype=dtype, rank=1],
        remove_indices: InputTensor[dtype=indices_type, rank=1],
    ) capturing:
        # remove_indices may not be sorted so our strategy is to use -1 to
        # represent removed dimensions in a copied version of our input shape buffer
        var num_input_dims = input_shape.dim_size[0]()
        var num_remove_indices = remove_indices.dim_size[0]()
        var final_rank = num_input_dims - num_remove_indices

        debug_assert(
            final_rank == output_shape.dim_size[0](),
            "Incorrect output shape.",
        )

        alias MAX_VECTOR_LIMIT = 12
        debug_assert(
            num_input_dims <= MAX_VECTOR_LIMIT,
            "Only support shape vectors up to rank-12.",
        )
        var input_shape_copy = IndexList[MAX_VECTOR_LIMIT]()
        for i in range(num_input_dims):
            input_shape_copy[i] = Int(input_shape[i])

        # Mark every squeezed dimension as -1 in our copy of the shape tensor
        for remove_index_index in range(num_remove_indices):
            var remove_index = Int(remove_indices[remove_index_index])
            var remove_index_normalize = remove_index + num_input_dims * Int(
                remove_indices[remove_index_index] < 0
            )
            input_shape_copy[remove_index_normalize] = -1

        # # Copy over the non -1 dimensions
        var output_shape_index = 0
        for input_shape_index in range(num_input_dims):
            if input_shape_copy[input_shape_index] == -1:
                continue
            output_shape[output_shape_index] = input_shape_copy[
                input_shape_index
            ]
            output_shape_index += 1

    @staticmethod
    fn shape[
        dtype: DType, indices_type: DType
    ](
        input_shape: InputTensor[dtype=dtype, rank=1],
        remove_indices: InputTensor[dtype=indices_type, rank=1],
    ) raises -> IndexList[1]:
        var out_dim = input_shape.dim_size[0]() - remove_indices.dim_size[0]()

        if out_dim < 0:
            raise Error(
                "[squeeze_shape] cannot remove more dimensions than there"
                " exists"
            )

        return IndexList[1](out_dim)


@compiler.register("mo.unsqueeze_shape")
struct UnsqueezeShape:
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType,
        indices_type: DType,
    ](
        output_shape: OutputTensor[dtype=dtype, rank=1],
        input_shape: InputTensor[dtype=dtype, rank=1],
        padding_indices: InputTensor[dtype=indices_type, rank=1],
    ) capturing:
        # represent uninitialized dimensions, add the padding dimensions, and copy
        # over the remaining dimensions later.
        var num_input_dims = input_shape.dim_size[0]()
        var num_padding_indices = padding_indices.dim_size[0]()
        var final_rank = num_input_dims + num_padding_indices
        debug_assert(
            final_rank == output_shape.dim_size[0](),
            "Incorrect output shape.",
        )
        for output_index in range(final_rank):
            output_shape[output_index] = -1

        for padding_index_index in range(num_padding_indices):
            var padding_index = Int(padding_indices[padding_index_index])
            var padding_index_normalize = padding_index + final_rank * Int(
                padding_indices[padding_index_index] < 0
            )

            debug_assert(
                padding_index_normalize >= 0
                and padding_index_normalize < final_rank,
                (
                    "Padding indices must be between [-r, r-1] where r is the"
                    " final output rank."
                ),
            )
            debug_assert(
                output_shape[padding_index_normalize] == -1,
                (
                    "Duplicate padding indices point to the same dimension in"
                    " the final output shape."
                ),
            )
            output_shape[padding_index_normalize] = 1

        # Copy over the remaining shapes
        var orig_shape_index = 0
        for output_shape_index in range(final_rank):
            if output_shape[output_shape_index] != -1:
                continue
            output_shape[output_shape_index] = input_shape[orig_shape_index]
            orig_shape_index += 1

    @staticmethod
    fn shape[
        dtype: DType, indices_type: DType
    ](
        input_shape: InputTensor[dtype=dtype, rank=1],
        remove_indices: InputTensor[dtype=indices_type, rank=1],
    ) -> IndexList[1]:
        var out_dim = input_shape.dim_size[0]() + remove_indices.dim_size[0]()
        return IndexList[1](out_dim)


# ===-----------------------------------------------------------------------===#
# ScatterND kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.scatter_nd")
struct ScatterND:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, *_],
        indices: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        scatter_nd[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            False,
            target,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[](
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, *_],
        indices: InputTensor,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=False](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.scatter_nd.add")
struct ScatterNDAdd:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, *_],
        indices: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs + rhs

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            False,
            target,
            reduce_fn=reduce_fn,
            _trace_description="scatter_nd.add",
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[](
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, *_],
        indices: InputTensor,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=False](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.scatter_nd.mul")
struct ScatterNDMul:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, *_],
        indices: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs * rhs

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            False,
            target,
            reduce_fn=reduce_fn,
            _trace_description="scatter_nd.mul",
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[](
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, *_],
        indices: InputTensor,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=False](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.scatter_nd.min")
struct ScatterNDMin:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, *_],
        indices: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return min(lhs, rhs)

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            False,
            target,
            reduce_fn=reduce_fn,
            _trace_description="scatter_nd.min",
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[](
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, *_],
        indices: InputTensor,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=False](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.scatter_nd.max")
struct ScatterNDMax:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, *_],
        indices: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return max(lhs, rhs)

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            False,
            target,
            reduce_fn=reduce_fn,
            _trace_description="scatter_nd.max",
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )

    @staticmethod
    fn shape[](
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, *_],
        indices: InputTensor,
    ) raises -> IndexList[input.rank]:
        return scatter_nd_shape[single_thread_blocking_override=False](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(updates),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.scatter_set_constant")
struct ScatterSetConstant:
    @staticmethod
    fn execute[
        data_type: DType,
        index_type: DType, //,
        target: StaticString,
    ](
        data: MutableInputTensor[dtype=data_type, rank=2],
        indices: InputTensor[dtype=index_type, rank=2],
        fill_value: Scalar[data_type],
        ctx: DeviceContextPtr,
    ) raises:
        scatter_set_constant[target, False](
            data.to_layout_tensor(),
            indices.to_layout_tensor(),
            fill_value,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Scatter kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.scatter")
struct Scatter:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, rank = output.rank],
        indices: InputTensor[rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        @always_inline
        @parameter
        fn reduce_func[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return rhs  # always return the latest update element

        scatter_elements[reduce_func](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
        )

    @staticmethod
    fn shape(
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, rank = input.rank],
        indices: InputTensor[rank = input.rank],
        axis: Scalar,
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        return scatter_elements_shape[single_thread_blocking_override=True](
            input_ndbuffer, updates_ndbuffer, indices_ndbuffer, Int(axis)
        )


@compiler.register("mo.scatter.add")
struct ScatterAdd:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, rank = output.rank],
        indices: InputTensor[rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        @always_inline
        @parameter
        fn reduce_func[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs + rhs

        scatter_elements[reduce_func](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
        )

    @staticmethod
    fn shape(
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, rank = input.rank],
        indices: InputTensor[rank = input.rank],
        axis: Scalar,
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        return scatter_elements_shape[single_thread_blocking_override=True](
            input_ndbuffer, updates_ndbuffer, indices_ndbuffer, Int(axis)
        )


@compiler.register("mo.scatter.max")
struct ScatterMax:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, rank = output.rank],
        indices: InputTensor[rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        @always_inline
        @parameter
        fn reduce_func[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return max(lhs, rhs)

        scatter_elements[reduce_func](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
        )

    @staticmethod
    fn shape(
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, rank = input.rank],
        indices: InputTensor[rank = input.rank],
        axis: Scalar,
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        return scatter_elements_shape[single_thread_blocking_override=True](
            input_ndbuffer, updates_ndbuffer, indices_ndbuffer, Int(axis)
        )


@compiler.register("mo.scatter.min")
struct ScatterMin:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, rank = output.rank],
        indices: InputTensor[rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        @always_inline
        @parameter
        fn reduce_func[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return min(lhs, rhs)

        scatter_elements[reduce_func](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
        )

    @staticmethod
    fn shape(
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, rank = input.rank],
        indices: InputTensor[rank = input.rank],
        axis: Scalar,
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        return scatter_elements_shape[single_thread_blocking_override=True](
            input_ndbuffer, updates_ndbuffer, indices_ndbuffer, Int(axis)
        )


@compiler.register("mo.scatter.mul")
struct ScatterMul:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        updates: InputTensor[dtype = output.dtype, rank = output.rank],
        indices: InputTensor[rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        @always_inline
        @parameter
        fn reduce_func[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return lhs * rhs

        scatter_elements[reduce_func](
            input,
            indices,
            updates,
            normalize_neg_index(Int(axis), output.rank),
            output,
        )

    @staticmethod
    fn shape(
        input: InputTensor,
        updates: InputTensor[dtype = input.dtype, rank = input.rank],
        indices: InputTensor[rank = input.rank],
        axis: Scalar,
    ) raises -> IndexList[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        return scatter_elements_shape[single_thread_blocking_override=True](
            input_ndbuffer, updates_ndbuffer, indices_ndbuffer, Int(axis)
        )


# ===-----------------------------------------------------------------------===#
# View kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.broadcast_to")
struct BroadcastTo:
    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    fn execute(input: InputTensor, shape: InputTensor) raises:
        raise Error("Should never be called!")

    @staticmethod
    fn shape_impl[
        input_rank: Int, output_rank: Int
    ](
        input: InputTensor[rank=input_rank],
        shape: InputTensor[rank=1],
    ) raises -> IndexList[output_rank]:
        if output_rank != shape.dim_size[0]():
            raise Error(
                "[broadcast_to] requires (len(target_shape) == output_rank)"
            )
        if input_rank > output_rank:
            raise Error("[broadcast_to] requires (input_rank <= output_rank)")

        # move the output shape from buffer into a static int tuple
        var output_shape = IndexList[output_rank]()

        for axis in range(output_rank):
            output_shape[axis] = Int(shape[axis])

        # Validate the compatibility between input and output shapes
        # NOTE we don't need to check the padded dims
        for i in range(input_rank):
            var input_axis = input_rank - i - 1
            var output_axis = output_rank - i - 1
            var input_dim = input.dim_size(input_axis)
            var output_dim = output_shape[output_axis]
            if input_dim != 1 and input_dim != output_dim:
                raise Error(
                    "[broadcast_to] input dimension at index ",
                    input_axis,
                    " (",
                    input_dim,
                    ") must be either 1 or equal to output dimension at index ",
                    output_axis,
                    " (",
                    output_dim,
                    ")",
                )
        return output_shape

    @staticmethod
    fn shape[
        input_rank: Int, output_rank: Int
    ](
        input: InputTensor[rank=input_rank],
        shape: InputTensor[rank=1],
    ) raises -> IndexList[output_rank]:
        return BroadcastTo.shape_impl[output_rank=output_rank](input, shape)


@compiler.register("mo.broadcast_shape")
struct BroadcastShape:
    @always_inline
    @staticmethod
    fn broadcast_shape_impl(
        out_buf: ManagedTensorSlice[rank=1],
        lhs_buf: ManagedTensorSlice[rank=1],
        rhs_buf: ManagedTensorSlice[rank=1],
    ):
        # Ensure lhs is always the smaller shape
        var lhs_rank = lhs_buf.size()
        var rhs_rank = rhs_buf.size()
        debug_assert(lhs_rank <= rhs_rank, "lhs shape must be the smaller one")

        # lhs_buf =      [l0, l1, ...]
        # rhs_buf = [..., r0, r1, ...]
        # out_buf = [..., o0, o1, ...]
        var size_diff = rhs_rank - lhs_rank
        for i in range(size_diff):
            out_buf[i] = rhs_buf[i].cast[out_buf.dtype]()

        for lhs_idx in range(lhs_rank):
            var rhs_idx = lhs_idx + size_diff
            var lhs_dim = Int(lhs_buf[lhs_idx])
            var rhs_dim = Int(rhs_buf[rhs_idx])
            if lhs_dim == rhs_dim:
                out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_buf.dtype]()

            elif lhs_dim != 1 and rhs_dim != 1:
                debug_assert(
                    rhs_dim == 1, "one of the differing dimensions must be 1"
                )

            elif lhs_dim != 1:
                out_buf[rhs_idx] = lhs_buf[lhs_idx].cast[out_buf.dtype]()

            elif rhs_dim != 1:
                out_buf[rhs_idx] = rhs_buf[rhs_idx].cast[out_buf.dtype]()

    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.broadcast_to` to always simplify to `mo.static.broadcast_to`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    fn execute(
        out_buf: OutputTensor[rank=1],
        lhs_buf: InputTensor[rank=1],
        rhs_buf: InputTensor[rank=1],
    ):
        var lhs_size = lhs_buf.size()
        var rhs_size = rhs_buf.size()
        if lhs_size > rhs_size:
            return BroadcastShape.broadcast_shape_impl(
                out_buf, rhs_buf, lhs_buf
            )
        return BroadcastShape.broadcast_shape_impl(out_buf, lhs_buf, rhs_buf)

    @staticmethod
    fn shape(
        lhs_buf: InputTensor[rank=1], rhs_buf: InputTensor[rank=1]
    ) raises -> IndexList[1]:
        var lhs_dim = lhs_buf.dim_size[0]()
        var rhs_dim = rhs_buf.dim_size[0]()
        return IndexList[1](max(lhs_dim, rhs_dim))


fn tuple_to_dimlist[size: Int](tuple: StaticTuple[Dim, size]) -> DimList:
    @parameter
    if size == 1:
        return DimList(VariadicList[Dim](tuple[0]))
    elif size == 2:
        return DimList(VariadicList[Dim](tuple[0], tuple[1]))
    elif size == 3:
        return DimList(VariadicList[Dim](tuple[0], tuple[1], tuple[2]))
    elif size == 4:
        return DimList(
            VariadicList[Dim](tuple[0], tuple[1], tuple[2], tuple[3])
        )
    elif size == 5:
        return DimList(
            VariadicList[Dim](tuple[0], tuple[1], tuple[2], tuple[3], tuple[4])
        )

    return DimList.create_unknown[size]()


@compiler.register("mo.static.broadcast_to")
@compiler.view_kernel
struct StaticBroadcastTo:
    @always_inline
    @staticmethod
    fn build_view[
        out_rank: Int,
    ](x: InputTensor,) -> IndexList[out_rank]:
        var new_strides = IndexList[out_rank]()
        alias delta = out_rank - x.rank

        @parameter
        for i in range(out_rank):

            @parameter
            if i < delta:
                new_strides[i] = 0
            else:
                if x.dim_size[i - delta]() <= 1:
                    new_strides[i] = 0
                else:
                    new_strides[i] = x.stride_length[i - delta]()

        return new_strides

    @staticmethod
    fn get_view_strides[
        out_rank: Int,
        in_rank: Int,
    ](input_shape: DimList, input_strides: DimList) -> DimList:
        var new_strides = StaticTuple[Dim, out_rank]()
        alias delta = out_rank - in_rank

        @parameter
        for i in range(out_rank):

            @parameter
            if i < delta:
                new_strides[i] = 0
            else:
                if input_shape.at[i - delta]().is_dynamic():
                    new_strides[i] = Dim()
                elif input_shape.get[i - delta]() <= 1:
                    new_strides[i] = 0
                else:
                    new_strides[i] = input_strides.at[i - delta]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn update_input_view[
        dtype: DType,
        in_rank: Int,
        out_rank: Int, //,
        output_static_shape: DimList,
    ](
        x: InputTensor[dtype=dtype, rank=in_rank],
        output_shape: IndexList[out_rank],
        out result: InputTensor[
            static_spec = x.static_spec.with_layout[out_rank](
                output_static_shape,
                Self.get_view_strides[out_rank, x.rank](
                    x._static_shape, x._static_strides
                ),
            )
        ],
    ):
        var x_runtime_strides = Self.build_view[out_rank](x)
        return __type_of(result)(
            x.unsafe_ptr(), output_shape, x_runtime_strides
        )

    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType,
        in_rank: Int,
        out_rank: Int,
        _trace_name: StaticString,
    ](
        z: OutputTensor[dtype=dtype, rank=out_rank],
        x: InputTensor[dtype=dtype, rank=in_rank],
        output_shape: IndexList[out_rank],
        ctx: DeviceContextPtr,
    ) raises:
        # We need the extra output_shape argument.
        # Using `z.shape` instead will prevent the compiler from fusing the kernels.

        var x_view = Self.update_input_view[z._static_shape](x, output_shape)

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](z, x_view, ctx)


@compiler.register("mo.static.reshape")
@compiler.view_kernel
struct StaticReshape:
    @staticmethod
    fn get_view_strides[
        out_rank: Int,
    ](out_shape: DimList) -> DimList:
        # reshape is a bit special as we assume the input is always contiguous.
        # So it will be the same with the output.
        var new_strides = StaticTuple[Dim, out_rank]()

        var stride = Dim(1)

        @parameter
        for i in reversed(range(out_rank)):
            # Start from the back so we can accumulate the strides.
            new_strides[i] = stride
            stride *= out_shape.at[i]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn update_input_view[
        dtype: DType,
        output_rank: Int, //,
        output_static_shape: DimList,
    ](
        input: InputTensor[dtype=dtype],
        shape: IndexList[output_rank],
        out result: InputTensor[
            static_spec = input.static_spec.with_layout[output_rank](
                output_static_shape,
                Self.get_view_strides[output_rank](output_static_shape),
            )
        ],
    ):
        var view_buffer = reshape(
            managed_tensor_slice_to_ndbuffer(input),
            shape,
        )

        return __type_of(result)(
            view_buffer.data,
            view_buffer.get_shape(),
            view_buffer.get_strides(),
        )

    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
        dtype: DType,
        output_rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=output_rank],
        input: InputTensor[dtype=dtype],
        shape: IndexList[output_rank],
        ctx: DeviceContextPtr,
    ) raises:
        var view_tensor = Self.update_input_view[output._static_shape](
            input, shape
        )

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view_tensor, ctx)


@compiler.register("mo.reshape")
struct Reshape:
    # The `execute` method should never be used in the graph compiler.
    # We expect `mo.reshape` to always simplify to `mo.static.reshape`
    #
    # Sometimes with a call to the below shape function.
    @staticmethod
    fn execute(input: InputTensor, shape: InputTensor) raises:
        raise Error("Should never be called!")

    @staticmethod
    fn shape[
        output_rank: Int
    ](input: InputTensor, shape: InputTensor[rank=1]) raises -> IndexList[
        output_rank
    ]:
        return reshape_shape[
            output_rank=output_rank, single_thread_blocking_override=True
        ](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(shape),
        )


@compiler.register("mo.transpose")
@compiler.view_kernel
struct Transpose:
    @always_inline
    @staticmethod
    fn transpose_in_place(
        input: InputTensor,
        permutations: InputTensor[rank=1],
        out result: (IndexList[input.rank], IndexList[input.rank]),
    ):
        var new_shape = IndexList[input.rank]()
        var new_stride = IndexList[input.rank]()

        @parameter
        for i in range(input.rank):
            var dim = Int(permutations[i])
            new_shape[i] = input.dim_size(dim)
            new_stride[i] = input.stride_length(dim)

        return __type_of(result)(new_shape, new_stride)

    @staticmethod
    fn get_view_strides[
        permutations: DimList, rank: Int
    ](input_strides: DimList) -> DimList:
        var new_strides = StaticTuple[Dim, rank]()

        @parameter
        for i in range(rank):
            alias perm = permutations.at[i]()

            @parameter
            if perm.is_dynamic():
                new_strides[i] = Dim()
            else:
                new_strides[i] = input_strides.at[Int(perm)]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn update_input_view[
        dtype: DType,
        rank: Int, //,
        output_static_shape: DimList,
        static_permutations: DimList,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        permutations: InputTensor[rank=1],
        out result: InputTensor[
            static_spec = input.static_spec.with_layout[rank](
                output_static_shape,
                Self.get_view_strides[static_permutations, rank](
                    input._static_strides
                ),
            )
        ],
    ):
        shape, strides = Self.transpose_in_place(input, permutations)
        return __type_of(result)(input.unsafe_ptr(), shape, strides)

    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
        static_permutations: DimList,
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        permutations: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var view = Self.update_input_view[
            output._static_shape, static_permutations
        ](input, permutations)

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view, ctx)

    # TODO(GEX-1033) Make it possible to have multiple raises.
    @no_inline
    @staticmethod
    fn shape_impl(
        input: InputTensor,
        permutations: InputTensor[rank=1],
    ) raises -> IndexList[input.rank]:
        if permutations.dim_size[0]() != input.rank:
            raise Error("[transpose] permutation size must match input rank")

        @parameter
        for i in range(input.rank):
            var perm = Int(permutations[i])
            if perm < 0 or input.rank <= perm:
                raise Error(
                    "[transpose] each permutation must be within range [0,"
                    " rank)"
                )

        shape, _ = Self.transpose_in_place(input, permutations)
        var out = IndexList[input.rank]()

        @parameter
        for i in range(input.rank):
            out[i] = shape[i]

        return out

    @staticmethod
    fn shape(
        input: InputTensor,
        permutations: InputTensor[rank=1],
    ) raises -> IndexList[input.rank]:
        return Self.shape_impl(input, permutations)


@compiler.register("mo.slice")
@compiler.view_kernel
struct Slice:
    @staticmethod
    fn get_view_strides[
        rank: Int
    ](input_strides: DimList, steps: DimList) -> DimList:
        var new_strides = StaticTuple[Dim, rank]()

        @parameter
        for i in range(rank):
            new_strides[i] = input_strides.at[i]() * steps.at[i]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn update_input_view[
        dtype: DType,
        rank: Int, //,
        output_static_shape: DimList,
        static_steps: DimList,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        starts: InputTensor[rank=1],
        stops: InputTensor[rank=1],
        steps: InputTensor[rank=1],
        out result: InputTensor[
            static_spec = input.static_spec.with_layout[rank](
                output_static_shape,
                Self.get_view_strides[rank](
                    input._static_strides, static_steps
                ),
            )
        ],
    ):
        var view_buffer = slice_as_view(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
        )

        return __type_of(result)(
            view_buffer.data,
            view_buffer.get_shape(),
            view_buffer.get_strides(),
        )

    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
        static_steps: DimList,
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        starts: InputTensor[rank=1],
        stops: InputTensor[rank=1],
        steps: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var view_tensor = Self.update_input_view[
            output._static_shape, static_steps
        ](input, starts, stops, steps)

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view_tensor, ctx)

    @staticmethod
    fn shape(
        input: InputTensor,
        starts: InputTensor[rank=1],
        stops: InputTensor[rank=1],
        steps: InputTensor[rank=1],
    ) raises -> IndexList[input.rank]:
        return slice_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
        )


@compiler.register("mo.mutable.store")
struct MutableStore:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        buffer: MutableInputTensor,
        tensor: FusedInputTensor,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[buffer.rank]) -> SIMD[buffer.dtype, width]:
            return rebind[SIMD[buffer.dtype, width]](
                tensor._fused_load[width](idx)
            )

        @parameter
        @always_inline
        fn out_func[width: Int](index: IndexList[buffer.rank]) capturing:
            var val = func[width](rebind[IndexList[buffer.rank]](index))
            buffer.store[width=width](index, val)

        foreach[
            func,
            out_func,
            target=target,
            _trace_name=_trace_name,
        ](buffer, ctx)


@compiler.register("mo.mutable.store.slice")
struct MutableStoreSlice:
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType,
        rank: Int,
    ](
        to_buffer: MutableInputTensor[dtype=dtype, rank=rank],
        in_slice: InputTensor[dtype=dtype, rank=rank],
        starts: InputTensor[rank=1],
        stops: InputTensor[rank=1],
        steps: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        copy_to_slice[target=target](
            managed_tensor_slice_to_ndbuffer(to_buffer),
            managed_tensor_slice_to_ndbuffer(in_slice),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
            ctx,
        )

    # No shape function as we just directly embed the logic to check the shape
    # of the 'slice' operand of the MO op directly in the kernel.


@compiler.register("mo.slice_dim")
@compiler.view_kernel
struct SliceDim:
    @staticmethod
    fn get_view_strides[
        rank: Int,
        axis: Int,
    ](input_strides: DimList, step: Dim) -> DimList:
        var new_strides = StaticTuple[Dim, rank]()

        @parameter
        for i in range(rank):
            if i == axis:
                new_strides[i] = input_strides.at[i]() * step
            else:
                new_strides[i] = input_strides.at[i]()

        return tuple_to_dimlist(new_strides)

    @staticmethod
    fn update_input_view[
        dtype: DType,
        rank: Int, //,
        output_static_shape: DimList,
        axis: Int,
        static_step: DimList,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        starts: Scalar,
        stops: Scalar,
        steps: Scalar,
        out result: InputTensor[
            static_spec = input.static_spec.with_layout[rank](
                output_static_shape,
                Self.get_view_strides[rank, axis](
                    input._static_strides, static_step.at[0]()
                ),
            )
        ],
    ):
        var view_buffer = slice_dim_as_view[dim=axis](
            managed_tensor_slice_to_ndbuffer(input),
            Int(starts),
            Int(stops),
            Int(steps),
        )

        return __type_of(result)(
            view_buffer.data,
            view_buffer.get_shape(),
            view_buffer.get_strides(),
        )

    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
        dtype: DType,
        rank: Int,
        axis: Int,
        static_step: DimList,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        starts: Scalar,
        stops: Scalar,
        steps: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        var view_tensor = Self.update_input_view[
            output._static_shape, axis, static_step
        ](input, starts, stops, steps)

        view_copy_impl[
            _trace_name=_trace_name,
            target=target,
        ](output, view_tensor, ctx)


# ===-----------------------------------------------------------------------===#
# Data dependent kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.arg_max")
struct ArgMax:
    @staticmethod
    fn execute[
        target: StaticString,
        rank: Int,
        _trace_name: StaticString,
    ](
        output: OutputTensor[rank=rank],
        input: InputTensor[rank=rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        var axis_val = normalize_neg_index(Int(axis), rank)

        with Trace[TraceLevel.OP, target=target](_trace_name):

            @parameter
            if target == "cpu":
                argmax(
                    input.to_layout_tensor(),
                    axis_val,
                    output.to_layout_tensor(),
                )
            else:
                if axis_val != rank - 1:
                    raise Error("axis other than -1 not supported on GPU")

                # Has no static shape info
                var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
                var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)

                # TODO(KERN-1045): Add support for taking advantage of static_shapes
                var cuda_ctx = ctx.get_device_context()
                argmax_gpu(
                    cuda_ctx,
                    input_ndbuffer,
                    output_ndbuffer,
                )


@compiler.register("mo.arg_min")
struct ArgMin:
    @staticmethod
    fn execute[
        target: StaticString,
        rank: Int,
        _trace_name: StaticString,
    ](
        output: OutputTensor[rank=rank],
        input: InputTensor[rank=rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        var axis_val = normalize_neg_index(Int(axis), rank)

        with Trace[TraceLevel.OP, target=target](_trace_name):

            @parameter
            if target == "cpu":
                argmin(
                    input.to_layout_tensor(),
                    axis_val,
                    output.to_layout_tensor(),
                )
            else:
                if axis_val != rank - 1:
                    raise Error("axis other than -1 not supported on GPU")

                # Has no static shape info
                var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
                var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)

                # TODO(KERN-1045): Add support for taking advantage of static_shapes
                var cuda_ctx = ctx.get_device_context()
                argmin_gpu(
                    cuda_ctx,
                    input_ndbuffer,
                    output_ndbuffer,
                )


@compiler.register("mo.arg_nonzero")
struct ArgNonZero:
    @staticmethod
    fn execute(
        output_buffer: OutputTensor[rank=2],
        input_buffer: InputTensor,
    ):
        arg_nonzero.arg_nonzero(
            input_buffer.to_layout_tensor(),
            output_buffer.to_layout_tensor(),
        )

    @staticmethod
    fn shape(input_buffer: InputTensor) -> IndexList[2]:
        return arg_nonzero.arg_nonzero_shape[
            single_thread_blocking_override=True
        ](input_buffer.to_layout_tensor())


@compiler.register("mo.mean")
struct Mean:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype = output.dtype, rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        mean[
            output.dtype,
            input_fn,
            output_fn,
            single_thread_blocking_override=False,
            target=target,
        ](input.shape(), axis_val, output.shape(), ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("mo.reduce.add")
struct ReduceAdd:
    @staticmethod
    fn execute[
        target: StaticString, _trace_name: StaticString
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype = output.dtype, rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        with Trace[TraceLevel.OP, target=target](_trace_name):
            sum[
                output.dtype,
                input_fn,
                output_fn,
                single_thread_blocking_override=False,
                target=target,
            ](input.shape(), axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("mo.reduce.mul")
struct ReduceMul:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype = output.dtype, rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        with Trace[TraceLevel.OP, target=target](_trace_name):
            product[
                output.dtype,
                input_fn,
                output_fn,
                single_thread_blocking_override=False,
                target=target,
            ](input.shape(), axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("mo.reduce.max")
struct ReduceMax:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype = output.dtype, rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        with Trace[TraceLevel.OP, target=target](_trace_name):
            reduce_max[
                output.dtype,
                input_fn,
                output_fn,
                single_thread_blocking_override=False,
                target=target,
            ](input.shape(), axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("mo.reduce.min")
struct ReduceMin:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype = output.dtype, rank = output.rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var axis_val = Int(axis)

        with Trace[TraceLevel.OP, target=target](_trace_name):
            reduce_min[
                output.dtype,
                input_fn,
                output_fn,
                single_thread_blocking_override=False,
                target=target,
            ](input.shape(), axis_val, ctx)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: InputTensor[dtype=input_type, rank=input_rank],
        axis: Scalar,
    ) raises -> IndexList[input_rank]:
        return reduce_shape(input, Int(axis))


@compiler.register("reduce_min_and_max")
struct ReduceMinMax:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        axis0: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        """Given a tensor of shape [A, B, C, D] and reducing along dimension 'C'
        writes to a tensor of shape [A, B, 2, D] where [:, :, 0, :] contains
        the minimum reduction and [:, :, 1, :] contains the maximum reduction.
        """

        alias num_reductions = 2
        var axis = normalize_neg_index(Int(axis0), rank)

        @parameter
        @always_inline
        fn input_0_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[input.dtype, width]:
            return input._fused_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_0_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank], val: SIMD[output.dtype, width]):
            output._fused_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        @always_inline
        @parameter
        fn input_0_fn_wrapper[
            _type: DType, width: Int, rank: Int
        ](idx: IndexList[rank]) -> SIMD[_type, width]:
            return rebind[SIMD[_type, width]](input_0_fn[width, rank](idx))

        @always_inline
        @parameter
        fn output_0_fn_wrapper[
            _type: DType,
            width: Int,
            rank: Int,
        ](
            indices: IndexList[rank],
            val: StaticTuple[SIMD[_type, width], num_reductions],
        ):
            # TODO: once we support multiple outputs, change this to route to
            # TODO: multiple output tensors.
            var indices_min = indices
            indices_min[axis] = 0
            output_0_fn[width, rank](
                indices_min, rebind[SIMD[dtype, width]](val[0])
            )

            var indices_max = indices
            indices_max[axis] = 1
            output_0_fn[width, rank](
                indices_max, rebind[SIMD[dtype, width]](val[1])
            )

        @always_inline
        @parameter
        fn reduce_fn[
            ty: DType,
            width: Int,
            reduction_idx: Int,
        ](left: SIMD[ty, width], right: SIMD[ty, width]) -> SIMD[ty, width]:
            constrained[reduction_idx < num_reductions, "reduction_idx OOB"]()

            @parameter
            if reduction_idx == 0:
                return min(left, right)
            else:
                return max(left, right)

        var init_min = Scalar[dtype].MAX
        var init_max = Scalar[dtype].MIN
        var init = StaticTuple[Scalar[dtype], num_reductions](
            init_min, init_max
        )

        with Trace[TraceLevel.OP, target=target](_trace_name):
            _reduce_generator[
                num_reductions,
                dtype,
                input_0_fn_wrapper,
                output_0_fn_wrapper,
                reduce_fn,
                single_thread_blocking_override=False,
                target=target,
            ](
                input.shape(),
                init=init,
                reduce_dim=axis,
                context=ctx,
            )
        _ = axis

    @staticmethod
    fn shape(input: InputTensor, axis: Scalar) -> IndexList[input.rank]:
        var new_shape = input.shape()
        new_shape[_unsafe_normalize_neg_index(Int(axis), input.rank)] = 2

        return new_shape


# ===-----------------------------------------------------------------------===#
# Pooling kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.avg_pool")
struct AvgPool:
    @staticmethod
    fn execute[
        count_boundary: Bool,
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        input: InputTensor[dtype=dtype, rank=4],
        filter: InputTensor[dtype=int_type, rank=1],
        strides: InputTensor[dtype=int_type, rank=1],
        dilations: InputTensor[dtype=int_type, rank=1],
        paddings: InputTensor[dtype=int_type, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        avg_pool[count_boundary=count_boundary, target=target](
            input.to_layout_tensor(),
            filter.to_layout_tensor(),
            strides.to_layout_tensor(),
            dilations.to_layout_tensor(),
            paddings.to_layout_tensor(),
            output.to_layout_tensor(),
            False,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        int_type: DType,
    ](
        input: InputTensor[dtype=dtype, rank=4],
        filter: InputTensor[dtype=int_type, rank=1],
        strides: InputTensor[dtype=int_type, rank=1],
        dilations: InputTensor[dtype=int_type, rank=1],
        paddings: InputTensor[dtype=int_type, rank=1],
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            pool_shape[single_thread_blocking_override=True](
                input.to_layout_tensor(),
                filter.to_layout_tensor(),
                strides.to_layout_tensor(),
                dilations.to_layout_tensor(),
                paddings.to_layout_tensor(),
            )
        )


@compiler.register("mo.avg_pool_ceil_mode_true")
struct AvgPoolCeilModeTrue:
    @staticmethod
    fn execute[
        count_boundary: Bool,
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        input: InputTensor[dtype=dtype, rank=4],
        filter: InputTensor[dtype=int_type, rank=1],
        strides: InputTensor[dtype=int_type, rank=1],
        dilations: InputTensor[dtype=int_type, rank=1],
        paddings: InputTensor[dtype=int_type, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        avg_pool[count_boundary=count_boundary, target=target](
            input.to_layout_tensor(),
            filter.to_layout_tensor(),
            strides.to_layout_tensor(),
            dilations.to_layout_tensor(),
            paddings.to_layout_tensor(),
            output.to_layout_tensor(),
            True,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        int_type: DType,
    ](
        input: InputTensor[dtype=dtype, rank=4],
        filter: InputTensor[dtype=int_type, rank=1],
        strides: InputTensor[dtype=int_type, rank=1],
        dilations: InputTensor[dtype=int_type, rank=1],
        paddings: InputTensor[dtype=int_type, rank=1],
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            pool_shape_ceil[single_thread_blocking_override=True](
                input.to_layout_tensor(),
                filter.to_layout_tensor(),
                strides.to_layout_tensor(),
                dilations.to_layout_tensor(),
                paddings.to_layout_tensor(),
            )
        )


@compiler.register("mo.max_pool")
struct MaxPool:
    @staticmethod
    fn execute[
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        input: InputTensor[dtype=dtype, rank=4],
        filter: InputTensor[dtype=int_type, rank=1],
        strides: InputTensor[dtype=int_type, rank=1],
        dilations: InputTensor[dtype=int_type, rank=1],
        paddings: InputTensor[dtype=int_type, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        max_pool[target=target](
            input.to_layout_tensor(),
            filter.to_layout_tensor(),
            strides.to_layout_tensor(),
            dilations.to_layout_tensor(),
            paddings.to_layout_tensor(),
            output.to_layout_tensor(),
            False,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        int_type: DType,
    ](
        input: InputTensor[dtype=dtype, rank=4],
        filter: InputTensor[dtype=int_type, rank=1],
        strides: InputTensor[dtype=int_type, rank=1],
        dilations: InputTensor[dtype=int_type, rank=1],
        paddings: InputTensor[dtype=int_type, rank=1],
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            pool_shape[single_thread_blocking_override=True](
                input.to_layout_tensor(),
                filter.to_layout_tensor(),
                strides.to_layout_tensor(),
                dilations.to_layout_tensor(),
                paddings.to_layout_tensor(),
            )
        )


@compiler.register("mo.max_pool_ceil_mode_true")
struct MaxPoolCeilModeTrue:
    @staticmethod
    fn execute[
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        input: InputTensor[dtype=dtype, rank=4],
        filter: InputTensor[dtype=int_type, rank=1],
        strides: InputTensor[dtype=int_type, rank=1],
        dilations: InputTensor[dtype=int_type, rank=1],
        paddings: InputTensor[dtype=int_type, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        max_pool[target=target](
            input.to_layout_tensor(),
            filter.to_layout_tensor(),
            strides.to_layout_tensor(),
            dilations.to_layout_tensor(),
            paddings.to_layout_tensor(),
            output.to_layout_tensor(),
            True,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        int_type: DType,
    ](
        input: InputTensor[dtype=dtype, rank=4],
        filter: InputTensor[dtype=int_type, rank=1],
        strides: InputTensor[dtype=int_type, rank=1],
        dilations: InputTensor[dtype=int_type, rank=1],
        paddings: InputTensor[dtype=int_type, rank=1],
    ) raises -> IndexList[input.rank]:
        return rebind[IndexList[input.rank]](
            pool_shape_ceil[single_thread_blocking_override=True](
                input.to_layout_tensor(),
                filter.to_layout_tensor(),
                strides.to_layout_tensor(),
                dilations.to_layout_tensor(),
                paddings.to_layout_tensor(),
            )
        )


# ===-----------------------------------------------------------------------===#
# Padding kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.pad.constant")
struct PadConstant:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        padding: InputTensor[rank=1],
        constant: Scalar[dtype=dtype],
        ctx: DeviceContextPtr,
    ) raises:
        var paddings_ptr = padding._ptr

        @parameter
        if is_cpu[target]():
            pad_constant(
                output.to_layout_tensor(),
                input.to_layout_tensor(),
                paddings_ptr,
                constant,
            )
        elif is_gpu[target]():
            pad_constant_gpu(
                output._ptr,
                output.shape(),
                input._ptr,
                input.shape(),
                paddings_ptr,
                constant,
                ctx.get_device_context(),
            )
        else:
            constrained[False, "Unknown target " + target]()

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        padding: InputTensor[rank=1],
        constant: Scalar[dtype=dtype],
    ) raises -> IndexList[rank]:
        # rebind is required because mojo can't figure out that
        # input.static_spec.to_layout_tensor().rank == input.rank
        return rebind[IndexList[rank]](
            pad_shape[single_thread_blocking_override=True](
                input.to_layout_tensor(),
                padding.to_layout_tensor(),
            )
        )


@compiler.register("mo.pad.repeat")
struct PadRepeat:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        padding: InputTensor[rank=1],
    ):
        var paddings_ptr = padding._ptr
        pad_repeat(
            output.to_layout_tensor(),
            input.to_layout_tensor(),
            paddings_ptr,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        padding: InputTensor[rank=1],
    ) raises -> IndexList[rank]:
        return rebind[IndexList[rank]](
            pad_shape[single_thread_blocking_override=True](
                input.to_layout_tensor(),
                padding.to_layout_tensor(),
            )
        )


@compiler.register("mo.pad.reflect")
struct PadReflect:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        padding: InputTensor[rank=1],
    ):
        var paddings_ptr = padding._ptr
        pad_reflect(
            output.to_layout_tensor(),
            input.to_layout_tensor(),
            paddings_ptr,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        padding: InputTensor[rank=1],
    ) raises -> IndexList[rank]:
        return rebind[IndexList[rank]](
            pad_shape[single_thread_blocking_override=True](
                input.to_layout_tensor(),
                padding.to_layout_tensor(),
            )
        )


# ===-----------------------------------------------------------------------===#
# Gather kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.gather_nd")
struct GatherND:
    @staticmethod
    fn execute[
        batchDims: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: OutputTensor,
        data: InputTensor[dtype = output.dtype, *_],
        indices: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var data_ndbuffer = managed_tensor_slice_to_ndbuffer(data)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)

        with Trace[TraceLevel.OP, target=target](_trace_name):
            gather_nd[batch_dims=batchDims, target=target](
                data_ndbuffer, indices_ndbuffer, output_ndbuffer, ctx
            )

    @staticmethod
    fn shape[
        batch_dims: Int, output_rank: Int
    ](
        data: InputTensor,
        indices: InputTensor,
    ) raises -> IndexList[
        output_rank
    ]:
        return gather_nd_shape[
            batch_dims=batch_dims,
            output_rank=output_rank,
            single_thread_blocking_override=False,
        ](
            managed_tensor_slice_to_ndbuffer(data),
            managed_tensor_slice_to_ndbuffer(indices),
        )


@compiler.register("mo.gather")
struct Gather:
    @staticmethod
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: FusedInputTensor[dtype = output.dtype, *_],
        indices: InputTensor,
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn indices_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[indices.dtype, width]:
            return indices._fused_load[width=width](
                rebind[IndexList[indices.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank], val: SIMD[output.dtype, width]):
            output._lambda_store[width=width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        with Trace[TraceLevel.OP, target=target](_trace_name):
            gather[
                dtype = output.dtype,
                indices_type = indices.dtype,
                input_fn=input_fn,
                indices_fn=indices_fn,
                output_fn=output_fn,
                target=target,
                single_thread_blocking_override=False,
            ](
                Axis(Int(axis), input.rank),
                input.shape(),
                indices.shape(),
                output.shape(),
                context=ctx,
            )

    @staticmethod
    fn shape[
        output_rank: Int,
    ](
        input: InputTensor,
        indices: InputTensor,
        axis: Scalar,
    ) raises -> IndexList[output_rank]:
        return gather_shape[
            output_rank=output_rank,
            single_thread_blocking_override=True,
        ](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(indices),
            Int(axis),
        )


@compiler.register("mo.gather_sum")
struct GatherSum:
    @staticmethod
    fn execute(
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, *_],
        indices: InputTensor[dtype = DType.int32, *_],
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)

        fn add[
            dtype: DType, simd_width: Int
        ](x: SIMD[dtype, simd_width], y: SIMD[dtype, simd_width]) -> SIMD[
            dtype, simd_width
        ]:
            return x + y

        gather_reduce[output.dtype, 0, 1, simdwidthof[output.dtype](), add](
            output_ndbuffer, input_ndbuffer, indices_ndbuffer, 0
        )


# ===-----------------------------------------------------------------------===#
# Normalization kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.layer_norm")
struct LayerNorm:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank],
        input: FusedInputTensor[dtype=dtype, rank=rank],
        gamma: FusedInputTensor[dtype=dtype, rank=1],
        beta: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn gamma_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return gamma._lambda_load[width=width](rebind[IndexList[1]](coords))

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var beta_buf = managed_tensor_slice_to_ndbuffer(beta)

        layer_norm[dtype, rank, input_fn, gamma_fn, output_fn, target=target](
            input.shape(),
            gamma.shape(),
            beta_buf,
            epsilon,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        beta: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
    ) -> IndexList[rank]:
        return input.shape()


@compiler.register("rms_norm")
struct RMSNorm:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        multiply_before_cast: Bool = True,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank],
        input: FusedInputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if output.shape() != input.shape():
            raise Error("Input and output buffers are not same shape")

        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int, alignment: Int
        ](coords: IndexList[_rank], val: SIMD[dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, width]](val),
            )

        var gamma_buf = managed_tensor_slice_to_ndbuffer(gamma)

        rms_norm[
            dtype,
            rank,
            input_fn,
            output_fn,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            input.shape(),
            gamma_buf,
            epsilon,
            weight_offset,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        weight_offset: Scalar[dtype=dtype],
    ) -> IndexList[rank]:
        return input.shape()


@compiler.register("group_norm")
struct GroupNorm:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: FusedInputTensor[dtype=dtype, rank=rank],
        gamma: FusedInputTensor[dtype=dtype, rank=1],
        beta: FusedInputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        num_groups: Int32,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn gamma_fn[width: Int](coords: IndexList[1]) -> SIMD[dtype, width]:
            return gamma._lambda_load[width=width](coords)

        @parameter
        @always_inline
        fn beta_fn[width: Int](coords: IndexList[1]) -> SIMD[dtype, width]:
            return beta._lambda_load[width=width](coords)

        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        group_norm[dtype, rank, input_fn, gamma_fn, beta_fn, target](
            shape=input.shape(),
            epsilon=epsilon,
            groups=num_groups,
            output=output_buf,
            ctx=ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        input: InputTensor[dtype=dtype, rank=rank],
        gamma: InputTensor[dtype=dtype, rank=1],
        beta: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype=dtype],
        num_groups: Int32,
    ) -> IndexList[rank]:
        return input.shape()


# ===-----------------------------------------------------------------------===#
# TopK kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.bottom_k")
struct BottomK:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        values: OutputTensor[dtype=dtype, rank=rank],
        indices: OutputTensor[dtype = DType.int64, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
        ctx: DeviceContextPtr,
    ) raises:
        top_k[largest=False, target=target](
            managed_tensor_slice_to_ndbuffer(input),
            Int(k),
            Int(axis),
            managed_tensor_slice_to_ndbuffer(values),
            managed_tensor_slice_to_ndbuffer(indices),
            sorted,
            ctx,
        )

    @staticmethod
    fn shape(
        input: InputTensor,
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
    ) raises -> IndexList[input.rank]:
        return top_k_shape_impl[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            Int(k),
            Int(axis),
        )


@compiler.register("mo.top_k")
struct TopK:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        values: OutputTensor[dtype=dtype, rank=rank],
        indices: OutputTensor[dtype = DType.int64, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
        ctx: DeviceContextPtr,
    ) raises:
        with Trace[TraceLevel.OP, target=target](_trace_name):
            top_k[largest=True, target=target](
                managed_tensor_slice_to_ndbuffer(input),
                Int(k),
                Int(axis),
                managed_tensor_slice_to_ndbuffer(values),
                managed_tensor_slice_to_ndbuffer(indices),
                sorted,
                ctx,
            )

    @staticmethod
    fn shape(
        input: InputTensor,
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[DType.bool],
    ) raises -> IndexList[input.rank]:
        return top_k_shape_impl[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            Int(k),
            Int(axis),
        )


# ===-----------------------------------------------------------------------===#
# Non maximum suppression kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.non_maximum_suppression")
struct NonMaximumSuppression:
    @staticmethod
    fn execute[
        dtype: DType
    ](
        output: OutputTensor[dtype = DType.int64, rank=2],
        boxes: InputTensor[dtype=dtype, rank=3],
        scores: InputTensor[dtype=dtype, rank=3],
        max_output_boxes_per_class: Int64,
        iou_threshold: Float32,
        score_threshold: Float32,
    ):
        var max_output_boxes_int = Int(max_output_boxes_per_class)
        var iou_threshold_float = iou_threshold
        var score_threshold_float = score_threshold

        non_max_suppression(
            boxes.to_layout_tensor(),
            scores.to_layout_tensor(),
            output.to_layout_tensor(),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )

    @staticmethod
    fn shape[
        dtype: DType
    ](
        boxes: InputTensor[dtype=dtype, rank=3],
        scores: InputTensor[dtype=dtype, rank=3],
        max_output_boxes_per_class: Int64,
        iou_threshold: Float32,
        score_threshold: Float32,
    ) -> IndexList[2]:
        var max_output_boxes_int = Int(max_output_boxes_per_class)
        var iou_threshold_float = iou_threshold
        var score_threshold_float = score_threshold

        return non_max_suppression_shape_func(
            boxes.to_layout_tensor(),
            scores.to_layout_tensor(),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )


# ===-----------------------------------------------------------------------===#
# Linalg kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.matmul")
struct Matmul:
    @staticmethod
    fn execute[
        transpose_b: Bool,
        packed_b: Bool,
        lambdas_have_fusion: Bool,
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: _FusedComputeOutputTensor[rank=2],
        a: InputTensor[rank=2],
        b: InputTensor[rank=2],
        ctx: DeviceContextPtr,
    ) capturing raises:
        constrained[
            not (packed_b and transpose_b),
            (
                "transpose_b and b_packed cannot both be true because"
                " pre-packing transposes B"
            ),
        ]()

        alias transposed_a = False

        var a_buffer = managed_tensor_slice_to_ndbuffer(a)
        var b_buffer = managed_tensor_slice_to_ndbuffer(b)
        var c_buffer = managed_tensor_slice_to_ndbuffer(c)

        @parameter
        @always_inline
        fn epilgue_fn[
            _dtype: DType, _width: Int, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]):
            c._lambda_store[width=_width, element_alignment=alignment](
                coords,
                rebind[SIMD[c.dtype, _width]](val),
            )

        @parameter
        @always_inline
        fn output_compute_fn[
            _dtype: DType, _width: Int, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[_dtype, _width]) -> SIMD[
            _dtype, _width
        ]:
            return rebind[SIMD[_dtype, _width]](
                c._fused_compute_output_lambda(
                    coords, rebind[SIMD[c.dtype, _width]](val)
                )
            )

        alias has_compute_lambda = c.static_spec.out_compute_lambda is not None

        matmul[
            transposed_a,
            transpose_b,
            packed_b,
            OptionalReg[matmul_elementwise_epilogue_type](
                epilgue_fn
            ) if lambdas_have_fusion
            and not has_compute_lambda else None,
            OptionalReg[matmul_elementwise_compute_lambda_type](
                output_compute_fn
            ) if lambdas_have_fusion
            and has_compute_lambda else None,
            saturated_vnni=False,
            single_thread_blocking_override=False,
            target=target,
            _trace_description=_trace_name,
        ](c_buffer, a_buffer, b_buffer, ctx)


@compiler.register("mo.batch_matmul")
struct BatchMatmul:
    @staticmethod
    fn execute[
        lambdas_have_fusion: Bool,
        rank: Int,
        transpose_b: Bool,
        target: StaticString,
    ](
        c: _FusedComputeOutputTensor[rank=rank],
        a: InputTensor[rank=rank],
        b: InputTensor[rank=rank],
        ctx: DeviceContextPtr,
    ) capturing raises:
        alias transpose_a = False

        var a_buffer = managed_tensor_slice_to_ndbuffer(a)
        var b_buffer = managed_tensor_slice_to_ndbuffer(b)
        var c_buffer = managed_tensor_slice_to_ndbuffer(c)

        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _width: Int, _rank: Int, *, alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_type, _width]):
            alias has_compute_lambda = c.static_spec.out_compute_lambda is not None

            @parameter
            if has_compute_lambda:
                var output = c._fused_compute_output_lambda(
                    rebind[IndexList[c.rank]](coords),
                    rebind[SIMD[c.dtype, _width]](val),
                )
                c.store[element_alignment=alignment](
                    rebind[IndexList[c.rank]](coords), output
                )
            else:
                c._lambda_store[width=_width, element_alignment=alignment](
                    rebind[IndexList[c.rank]](coords),
                    rebind[SIMD[c.dtype, _width]](val),
                )

        batched_matmul[
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            elementwise_epilogue_fn = OptionalReg[
                batched_matmul_elementwise_epilogue_type
            ](output_fn) if lambdas_have_fusion else None,
            saturated_vnni=False,
            single_thread_blocking_override=False,
            target=target,
        ](c_buffer, a_buffer, b_buffer, context=ctx)

    @staticmethod
    fn shape[
        rank: Int,
        a_type: DType,
        b_type: DType,
    ](
        a: InputTensor[dtype=a_type, rank=rank],
        b: InputTensor[dtype=b_type, rank=rank],
    ) raises -> IndexList[rank]:
        var a_buffer = managed_tensor_slice_to_ndbuffer(a)
        var b_buffer = managed_tensor_slice_to_ndbuffer(b)
        return batched_matmul_shape[single_thread_blocking_override=True](
            a_buffer, b_buffer
        )


@compiler.register("mo.linalg.band_part")
struct LinalgBandPart:
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType,
        int_type: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: FusedInputTensor[dtype=dtype, rank=rank],
        num_lower: InputTensor[dtype=int_type, rank=1],
        num_upper: InputTensor[dtype=int_type, rank=1],
        exclude: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        var num_lower_buf = managed_tensor_slice_to_ndbuffer(num_lower)
        var num_upper_buf = managed_tensor_slice_to_ndbuffer(num_upper)
        var exclude_buf = managed_tensor_slice_to_ndbuffer(exclude)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        matrix_band_part[
            input_0_fn=input_fn,
            simd_width = simdwidthof[dtype](),
            single_thread_blocking_override=False,
            target=target,
        ](
            input.shape(),
            num_lower_buf,
            num_upper_buf,
            exclude_buf,
            output_buf,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Resize kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.resize.nearest")
struct ResizeNearest:
    @staticmethod
    fn execute[
        coordinate_transform_mode: Int,
        round_mode: Int,
        rank: Int,
        dtype: DType,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        size: InputTensor[rank=1],
    ) raises:
        resize_nearest_neighbor[coordinate_transform_mode, round_mode](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(output),
        )

    @staticmethod
    fn shape[
        rank: Int
    ](
        input: InputTensor[rank=rank],
        size: InputTensor[rank=1],
    ) -> IndexList[
        rank
    ]:
        var shape = IndexList[rank]()
        for i in range(rank):
            shape[i] = Int(size[i])

        return shape


@compiler.register("mo.resize.linear")
struct ResizeLinear:
    @staticmethod
    fn execute[
        coordinate_transform_mode: Int,
        antialias: Bool,
        rank: Int,
        dtype: DType,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        size: InputTensor[rank=1],
    ):
        resize_linear[coordinate_transform_mode, antialias](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(output),
        )

    @staticmethod
    fn shape[
        rank: Int
    ](
        input: InputTensor[rank=rank],
        size: InputTensor[rank=1],
    ) -> IndexList[
        rank
    ]:
        var shape = IndexList[rank]()
        for i in range(rank):
            shape[i] = Int(size[i])

        return shape


@compiler.register("mo.resize.bicubic")
struct ResizeBicubic:
    @staticmethod
    fn execute[
        rank: Int,
        dtype: DType,
        target: StaticString, //,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        size: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        # Get input and output dimensions from tensors
        var output_buffer = managed_tensor_slice_to_ndbuffer(output)
        var input_buffer = managed_tensor_slice_to_ndbuffer(input)

        resize_bicubic[target](output_buffer, input_buffer, ctx)

    @staticmethod
    fn shape[
        rank: Int
    ](input: InputTensor[rank=rank], size: InputTensor[rank=1]) -> IndexList[
        rank
    ]:
        var shape = IndexList[rank]()
        for i in range(rank):
            shape[i] = Int(size[i])

        return shape


# ===-----------------------------------------------------------------------===#
# ROI align kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.roi_align")
struct ROIAlign:
    @staticmethod
    fn execute[
        aligned: Bool,
        mode: StaticString,
        dtype: DType,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        input: InputTensor[dtype=dtype, rank=4],
        rois: InputTensor[dtype=dtype, rank=2],
        output_height: Int64,
        output_width: Int64,
        spatial_scale: Scalar,
        sampling_ratio: Scalar,
    ):
        roi_align_nhwc[aligned, mode](
            output.to_layout_tensor(),
            input.to_layout_tensor(),
            rois.to_layout_tensor(),
            Int(output_height),
            Int(output_width),
            spatial_scale,
            sampling_ratio,
        )

    @staticmethod
    fn shape(
        input: InputTensor[rank=4],
        rois: InputTensor[rank=2],
        output_height: Int64,
        output_width: Int64,
        spatial_scale: Scalar,
        sampling_ratio: Scalar,
    ) -> IndexList[4]:
        var shape = IndexList[4]()
        # input shape is [N, H, W, C]
        # rois shape is [M, 5]
        # output shape is [M, output_height, output_width, C]
        shape[0] = rois.dim_size[0]()
        shape[1] = Int(output_height)
        shape[2] = Int(output_width)
        shape[3] = input.dim_size[3]()

        return shape


# ===-----------------------------------------------------------------------===#
# Tile kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.tile")
struct Tile:
    @staticmethod
    fn execute[
        dtype: DType, rank: Int
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        repeats: InputTensor,
    ) raises:
        tile(
            input.to_layout_tensor(),
            repeats.to_layout_tensor(),
            output.to_layout_tensor(),
        )

    @staticmethod
    fn shape(
        input: InputTensor,
        repeats: InputTensor[rank=1],
    ) raises -> IndexList[input.rank]:
        # rebind is required because mojo can't figure out that
        # input.static_spec.to_layout_tensor().rank == input.rank
        return rebind[IndexList[input.rank]](
            tile_shape[single_thread_blocking_override=True](
                input.to_layout_tensor(),
                repeats.to_layout_tensor(),
            )
        )


# ===-----------------------------------------------------------------------===#
# Repeat Interleave kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("repeat_interleave")
struct RepeatInterleave:
    @staticmethod
    fn execute(
        output: OutputTensor,
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        repeats: InputTensor[rank=1],
        axis: Scalar,
    ) raises:
        constrained[
            axis.dtype.is_integral(), "axis value must be integer type"
        ]()

        repeat_interleave(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(repeats),
            Int(normalize_neg_index(axis, input.rank)),
            managed_tensor_slice_to_ndbuffer(output),
        )

    @staticmethod
    fn shape(
        input: InputTensor, repeats: InputTensor[rank=1], axis: Scalar
    ) raises -> IndexList[input.rank]:
        constrained[
            axis.dtype.is_integral(), "axis value must be integer type"
        ]()

        return repeat_interleave_shape(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(repeats),
            Int(normalize_neg_index(axis, input.rank)),
        )


# ===-----------------------------------------------------------------------===#
# Random kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.random.normal")
struct RandomNormal:
    @staticmethod
    fn execute(
        output: OutputTensor,
        shape: InputTensor[rank=1],
        mean: Scalar,
        variance: Scalar,
        seed_value: Scalar,
    ):
        seed(Int(seed_value))
        var num_elements = 1
        # TODO: Add __len__ support in InputTensor.
        for i in range(shape.dim_size[0]()):
            num_elements *= Int(shape[i])
        randn(
            output._ptr,
            num_elements,
            mean.cast[DType.float64](),
            variance.cast[DType.float64](),
        )

    @staticmethod
    fn shape[
        output_rank: Int
    ](
        shape: InputTensor[rank=1],
        mean: Scalar,
        variance: Scalar,
        seed_value: Scalar,
    ) -> IndexList[output_rank]:
        var unrolled_shape = IndexList[output_rank]()
        for i in range(output_rank):
            unrolled_shape[i] = Int(shape[i])

        return unrolled_shape


@compiler.register("mo.static.random.normal")
struct StaticRandomNormal:
    @staticmethod
    fn execute(
        output: OutputTensor,
        mean: Scalar,
        variance: Scalar,
        seed_value: Scalar,
    ):
        seed(Int(seed_value))
        var num_elements = output.size()
        randn(
            output._ptr,
            num_elements,
            mean.cast[DType.float64](),
            variance.cast[DType.float64](),
        )


@compiler.register("mo.random.uniform")
struct RandomUniform:
    @staticmethod
    fn execute[
        dtype: DType,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype],
        shape: InputTensor[rank=1],
        lower_bound: Scalar[dtype],
        upper_bound: Scalar[dtype],
        seed_value: Scalar,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn output_fn[
            _width: Int,
            _rank: Int,
        ](coords: IndexList[_rank], val: SIMD[dtype, _width]):
            output._lambda_store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        random_uniform[output_fn, target=target](
            output.shape(), lower_bound, upper_bound, UInt64(seed_value), ctx
        )

    @staticmethod
    fn shape[
        output_rank: Int
    ](
        shape: InputTensor[rank=1],
        mean: Scalar,
        variance: Scalar,
        seed_value: Scalar,
    ) -> IndexList[output_rank]:
        debug_assert(shape.dim_size[0]() == output_rank)

        var unrolled_shape = IndexList[output_rank]()
        for i in range(output_rank):
            unrolled_shape[i] = Int(shape[i])

        return unrolled_shape


# ===-----------------------------------------------------------------------===#
# Softmax kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.softmax")
struct Softmax:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        output: OutputTensor,
        input: FusedInputTensor[dtype = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) capturing raises:
        # shape should be the same between the two inputs
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)

        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        softmax[
            output.dtype,
            simdwidthof[output.dtype](),
            output.rank,
            output_ndbuffer.shape,
            input_fn,
            target,
        ](
            output.shape(),
            output_ndbuffer,
            output.rank - 1,
            context=ctx,
        )


@compiler.register("mo.logsoftmax")
struct LogSoftmax:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        output: OutputTensor,
        input: FusedInputTensor[dtype = output.dtype, rank = output.rank],
    ) capturing raises:
        # shape should be the same between the two inputs
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)

        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: IndexList[_rank]) -> SIMD[output.dtype, width]:
            return input._lambda_load[width=width](
                rebind[IndexList[input.rank]](coords)
            )

        logsoftmax[
            output.dtype,
            simdwidthof[output.dtype](),
            output.rank,
            output_ndbuffer.shape,
            input_fn,
        ](output.shape(), output_ndbuffer, output.rank - 1)


# ===-----------------------------------------------------------------------===#
# Cumsum kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.cumsum")
struct CumSum:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        exclusive: Int,
        reverse: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ):
        cumsum[dtype, exclusive, reverse](
            output.to_layout_tensor(),
            input.to_layout_tensor(),
            _unsafe_normalize_neg_index(Int(axis), rank),
        )


# ===-----------------------------------------------------------------------===#
# Concat kernels
# ===-----------------------------------------------------------------------===#


fn concat_shape_impl[
    dtype: DType, rank: Int, size: Int, io_spec: IOSpec
](
    axis0: Int,
    inputs: VariadicTensors[dtype, rank, size, io_spec=io_spec],
) raises -> IndexList[rank]:
    var axis = normalize_neg_index(axis0, rank)

    @parameter
    @always_inline
    fn shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) -> Bool:
        @parameter
        for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0

    @parameter
    for i in range(inputs.size):
        concat_axis_dim_sum += inputs[i].dim_size(axis)
        if not shape_equal_ignore_axis(
            inputs[0].shape(),
            inputs[i].shape(),
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = inputs[0].shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


@compiler.register("mo.concat")
struct Concat:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: FusedOutputTensor[dtype=dtype, rank=rank],
        axis: Scalar,
        inputs: FusedInputVariadicTensors[dtype, rank, *_],
        ctx: DeviceContextPtr,
    ) capturing raises:
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        var input_shapes = StaticTuple[IndexList[rank], inputs.size]()

        @parameter
        for i in range(inputs.size):
            input_shapes[i] = inputs[i].shape()

        @always_inline
        @parameter
        fn inputs_lambda[
            input_index: Int,
            width: Int,
            _rank: Int,
        ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
            constrained[
                input_index < inputs.size, "tensor index out of bounds"
            ]()
            return inputs[input_index]._lambda_load[width=width](
                rebind[IndexList[rank]](indices)
            )

        @always_inline
        @parameter
        fn epilogue_wrapper[
            _dtype: DType, _rank: Int, width: Int, *, alignment: Int = 1
        ](indices: IndexList[_rank], value: SIMD[_dtype, width]):
            output._lambda_store[width=width, element_alignment=alignment](
                rebind[IndexList[output.rank]](indices),
                rebind[SIMD[output.dtype, width]](value),
            )

        fused_concat[
            dtype,
            rank,
            False,
            inputs_lambda,
            epilogue_wrapper,
            target,
        ](
            normalize_neg_index(Int(axis), rank),
            input_shapes,
            output_buf,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        axis: Scalar, inputs: InputVariadicTensors[dtype, rank, *_]
    ) raises -> IndexList[rank]:
        return concat_shape_impl(Int(axis), inputs)


# Helper method used by compiler to reconcile MGP list with dtype Mojo expects.
@register_internal("to_managed_tensor_slice_list")
@always_inline
fn to_managed_tensor_slice_list[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    raw_list_ptr: OpaquePointer,
) -> List[
    ManagedTensorSlice[
        io_spec = IOSpec[mut, input](),
        static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
    ]
]:
    var num_elements = external_call["MGP_RT_ListSize", Int64](
        raw_list_ptr
    ).__int__()

    var data_ptrs = List[OpaquePointer](capacity=num_elements)
    var dim_values = List[Int64](capacity=num_elements * rank)

    # Collect the data pointers and dimensions of each element from the list.
    external_call["MGP_RT_ListPopulate", NoneType](
        raw_list_ptr, data_ptrs.unsafe_ptr(), dim_values.unsafe_ptr()
    )

    # TODO: revisit the use of unknown here
    # Create output list
    var out_list = List[
        ManagedTensorSlice[
            io_spec = IOSpec[mut, input](),
            static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
        ]
    ](capacity=num_elements)

    # Convert individual elements of the input list into NDBuffer, and
    # accumulate the results to output list.
    for i in range(num_elements):
        var data = data_ptrs[i].bitcast[Scalar[dtype]]()

        var dims = IndexList[rank]()

        @parameter
        for dim in range(rank):
            dims[dim] = dim_values[dim + i * rank].__int__()

        var buffer = _to_managed_tensor_slice_index_list_shape[
            dtype, rank, mut, input
        ](data, dims)
        out_list.append(buffer)

    return out_list^


# NOTE: there are a lot of similarities between this and the shape func
# for mo.concat.
fn concat_from_list_shape_impl[
    dtype: DType, rank: Int
](
    axis0: Int,
    inputs: List[
        InputTensor[
            static_spec = StaticTensorSpec[dtype, rank].create_unknown(),
        ]
    ],
) raises -> IndexList[rank]:
    var axis = normalize_neg_index(axis0, rank)

    @parameter
    @always_inline
    fn shape_equal_ignore_axis(
        s1: IndexList[rank], s2: IndexList[rank]
    ) -> Bool:
        for i in range(rank):
            if i != axis and s1[i] != s2[i]:
                return False
        return True

    var concat_axis_dim_sum = 0
    for i in range(len(inputs)):
        concat_axis_dim_sum += inputs[i].dim_size(axis)
        if not shape_equal_ignore_axis(
            inputs[0].shape(),
            inputs[i].shape(),
        ):
            raise Error(
                "[concat] input shapes must match except at concat axis"
            )

    # compute and return the output shape
    var output_shape = inputs[0].shape()
    output_shape[axis] = concat_axis_dim_sum
    return output_shape


@compiler.register("mo.concat_from_list")
struct ConcatFromList:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        inputs: List[
            InputTensor[
                static_spec = StaticTensorSpec[dtype, rank].create_unknown()
            ]
        ],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        constrained[
            target == "cpu", "only cpu is supported for concat_from_list"
        ]()
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        # TODO: convert underlying kernel to accept lists of ManagedTensorSlice
        var input_as_ndbuffer = List[NDBuffer[dtype, rank, MutableAnyOrigin]](
            capacity=len(inputs)
        )
        for i in range(len(inputs)):
            input_as_ndbuffer.append(
                managed_tensor_slice_to_ndbuffer(inputs[i])
            )

        _concat_cpu[rank, dtype, None, False](
            output_buf,
            normalize_neg_index(Int(axis), rank),
            input_as_ndbuffer,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        inputs: List[
            InputTensor[
                static_spec = StaticTensorSpec[dtype, rank].create_unknown()
            ]
        ],
        axis: Scalar,
    ) raises -> IndexList[rank]:
        return concat_from_list_shape_impl(Int(axis), inputs)


# ===-----------------------------------------------------------------------===#
# Split kernels
# ===-----------------------------------------------------------------------===#


# The shape function for split is special and there is special
# handling in the graph compiler to make things work.
@compiler.register("mo.split")
struct Split:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: OutputVariadicTensors[dtype, rank, *_],
        input: InputTensor[dtype=dtype, rank=rank],
        split_sizes: InputTensor[rank=1],
        axis: Scalar,
        ctx: DeviceContextPtr,
    ) raises:
        var output_bufs = StaticTuple[
            LayoutTensor[dtype, Layout.row_major[rank](), MutableAnyOrigin],
            output.size,
        ]()

        @parameter
        for i in range(output.size):
            var output_tensor = LayoutTensor[dtype, Layout.row_major[rank]()](
                output[i].unsafe_ptr(),
                RuntimeLayout[Layout.row_major[rank]()].row_major(
                    output[i].to_layout_tensor().runtime_layout.shape.value,
                ),
            )
            output_bufs[i] = output_tensor

        split[dtype, target=target, trace_description=_trace_name](
            input.to_layout_tensor(),
            normalize_neg_index(Int(axis), rank),
            output_bufs,
            ctx.get_device_context(),
        )


# In practice this is how it's done. The graph compiler has additional logic
# to properly dispatch this function.
@compiler.register("split_ith_output_shape")
struct SplitOutputShapeHelper:
    @staticmethod
    fn execute(
        input_buf: InputTensor,
        split_sizes_buf: InputTensor,
        split_axis: Scalar,
        output_idx: Scalar,
    ) raises:
        raise Error("Should not be called directly.")

    @staticmethod
    @always_inline
    fn shape[
        rank: Int,
        input_type: DType,
        split_size_type: DType,
    ](
        input_buf: InputTensor[dtype=input_type, rank=rank],
        split_sizes_buf: InputTensor[dtype=split_size_type, rank=1],
        split_axis: Scalar,
        output_idx: Scalar,
    ) raises -> IndexList[rank]:
        # extract relevant hyper parameters
        if not (0 <= Int(output_idx) < split_sizes_buf.size()):
            raise Error(
                "[split] output index must be within range [0,"
                " len(split_sizes))"
            )
        var output_split_size = Int(split_sizes_buf[Int(output_idx)])

        var normalized_split_axis = normalize_neg_index(Int(split_axis), rank)

        var split_sizes_sum = 0

        for i in range(split_sizes_buf.dim_size[0]()):
            split_sizes_sum += Int(split_sizes_buf[i])
        if split_sizes_sum != input_buf.dim_size(normalized_split_axis):
            raise Error(
                "[split] sum of split sizes must match input dimension at split"
                " axis"
            )

        # compute and return the output shape
        var output_shape = input_buf.shape()
        output_shape[normalized_split_axis] = output_split_size
        return output_shape


# ===-----------------------------------------------------------------------===#
# Convolution kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.conv")
struct Conv:
    @staticmethod
    fn execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        lambdas_have_fusion: Bool,
        static_strides: DimList,
        static_dilations: DimList,
        static_padding: DimList,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor,
        input: InputTensor[rank = output.rank],
        filter: InputTensor,
        strides: InputTensor,
        dilation: InputTensor,
        paddings: InputTensor,
        num_groups: Scalar,
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn output_fn[
            _dtype: DType, _rank: Int, _width: Int
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            output._lambda_store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        constrained[
            strides.dtype.is_integral() and dilation.dtype.is_integral(),
            "stride and dilation must have integral type",
        ]()

        constrained[
            input_layout == "NHWC", "only NHWC input layout is supported"
        ]()

        if strides.size() != input.rank - 2:
            raise Error("(input_rank-2) values expected in conv strides")

        if dilation.size() != input.rank - 2:
            raise Error("(input_rank-2) values expected in conv dilation")

        if paddings.size() != 2 * (input.rank - 2):
            raise Error("(2*(input_rank-2)) value expected in conv paddings")

        var stride_tuple = IndexList[input.rank - 2](0)
        var dilation_tuple = IndexList[input.rank - 2](0)

        @parameter
        for i in range(input.rank - 2):
            stride_tuple[i] = Int(strides._ptr[i])
            dilation_tuple[i] = Int(dilation._ptr[i])

        if dilation_tuple != IndexList[input.rank - 2](1):
            raise Error("Non-unit dilation is not supported yet.")

        var pad_d_tuple = IndexList[2](0)
        var pad_h_tuple = IndexList[2](0)
        var pad_w_tuple = IndexList[2](0)

        @parameter
        if input.rank == 3:
            pad_w_tuple = Index(paddings._ptr[0], paddings._ptr[1])
        elif input.rank == 4:
            pad_h_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_w_tuple = Index(paddings._ptr[2], paddings._ptr[3])
        elif input.rank == 5:
            pad_d_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_h_tuple = Index(paddings._ptr[2], paddings._ptr[3])
            pad_w_tuple = Index(paddings._ptr[4], paddings._ptr[5])

        alias conv_attr = ConvInfoStatic[input.rank - 2](
            static_padding,
            static_strides,
            static_dilations,
            input._static_shape.at[input.rank - 1](),  # input C, NHWC
            filter._static_shape.at[
                filter.rank - 2
            ](),  # filter C, RSCF or FRSCf
        )

        alias filter_packed = filter_layout == "FRSCf" or filter_layout == "FQRSCf"
        alias filter_is_fcrs = filter_layout == "FCRS"

        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var filter_buf = managed_tensor_slice_to_ndbuffer(filter)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        with Trace[TraceLevel.OP, target=target](_trace_name):

            @parameter
            if is_cpu[target]():
                constrained[
                    not filter_is_fcrs,
                    "Filter layout FCRS is not supported on CPU",
                ]()
                conv_nhwc_direct[
                    input.rank,
                    filter.rank,
                    input._static_shape,  # input shape
                    filter._static_shape,  # filter shape
                    output._static_shape,  # output shape
                    input.dtype,
                    filter.dtype,
                    output.dtype,
                    filter_packed,
                    conv_attr,
                    lambdas_have_fusion,
                    output_fn,
                ](
                    input_buf,
                    filter_buf,
                    output_buf,
                    stride_tuple,
                    dilation_tuple,
                    pad_d_tuple,
                    pad_h_tuple,
                    pad_w_tuple,
                    Int(num_groups),
                )
            else:
                constrained[
                    (input.rank == 4 and filter.rank == 4)
                    or (input.rank == 5 and filter.rank == 5),
                    "only rank 4 or 5 tensor is supported on cuda gpu",
                ]()
                constrained[
                    filter_packed == False,
                    "only unpacked filter is supported on cuda gpu",
                ]()

                var cuda_ctx = ctx.get_device_context()
                var pad_tuple = IndexList[input.rank - 2](0)

                @parameter
                if input.rank == 4:
                    pad_tuple[0] = pad_h_tuple[0]
                    pad_tuple[1] = pad_w_tuple[0]
                elif input.rank == 5:
                    pad_tuple[0] = pad_d_tuple[0]
                    pad_tuple[1] = pad_h_tuple[0]
                    pad_tuple[2] = pad_w_tuple[0]

                conv_gpu[
                    input.rank,
                    filter.rank,
                    input._static_shape,  # input shape
                    filter._static_shape,  # filter shape
                    output._static_shape,  # output shape
                    input.dtype,
                    filter.dtype,
                    output.dtype,
                    output_fn,
                    filter_is_fcrs,
                ](
                    input_buf,
                    filter_buf,
                    output_buf,
                    stride_tuple,
                    dilation_tuple,
                    pad_tuple,
                    Int(num_groups),
                    cuda_ctx,
                )

    @staticmethod
    fn shape[
        dtype: DType
    ](
        input: InputTensor,
        filter: InputTensor,
        strides: InputTensor[rank=1],
        dilations: InputTensor[rank=1],
        paddings: InputTensor[rank=1],
        num_groups: Scalar,
    ) raises -> IndexList[input.rank]:
        return conv_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            num_groups,
        )


@compiler.register("mo.conv_transpose")
struct ConvTranspose:
    @staticmethod
    fn execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        lambdas_have_fusion: Bool,
        target: StaticString,
    ](
        output: FusedOutputTensor,
        input: InputTensor[rank = output.rank],
        filter: InputTensor,
        strides: InputTensor[rank=1],
        dilation: InputTensor[rank=1],
        paddings: InputTensor[rank=1],
        output_paddings: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) capturing raises:
        constrained[
            strides.dtype.is_integral()
            and dilation.dtype.is_integral()
            and output_paddings.dtype.is_integral()
        ]()

        if strides.size() != input.rank - 2:
            raise Error(
                "(input_rank-2) values expected in convTranspose stride"
            )

        if dilation.size() != input.rank - 2:
            raise Error(
                "(input_rank-2) values expected in convTranspose dilation"
            )

        if output_paddings.size() != input.rank - 2:
            raise Error(
                "(input_rank-2) values expected in convTranspose output"
                " paddings"
            )

        if paddings.size() != 2 * (input.rank - 2):
            raise Error(
                "(2*(input_rank-2)) value expected in convTranspose paddings"
            )

        var stride_tuple = IndexList[input.rank - 2](0)
        var dilation_tuple = IndexList[input.rank - 2](0)

        @parameter
        for i in range(input.rank - 2):
            stride_tuple[i] = Int(strides._ptr[i])
            dilation_tuple[i] = Int(dilation._ptr[i])

        var pad_d = IndexList[2](0)
        var pad_h = IndexList[2](0)
        var pad_w = IndexList[2](0)

        @parameter
        if input.rank == 3:
            pad_w = Index(paddings[0], paddings[1])
        elif input.rank == 4:
            pad_h = Index(paddings[0], paddings[1])
            pad_w = Index(paddings[2], paddings[3])
        elif input.rank == 5:
            pad_d = Index(paddings[0], paddings[1])
            pad_h = Index(paddings[2], paddings[3])
            pad_w = Index(paddings[4], paddings[5])

        @parameter
        @always_inline
        fn output_fn[
            _dtype: DType, _rank: Int, _width: Int
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            output._lambda_store[width=_width](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        alias filter_packed = filter_layout == "FRSCf" or filter_layout == "FQRSCf"
        alias filter_is_cfrs = filter_layout == "CFRS"

        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var filter_buf = managed_tensor_slice_to_ndbuffer(filter)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        @parameter
        if is_cpu[target]():
            conv_transposed_cpu[
                input.rank,
                filter.rank,
                input._static_shape,  # Input shape.
                filter._static_shape,  # Filter shape.
                output._static_shape,  # Output shape.
                input.dtype,
                filter.dtype,  # Filter dtype.
                output.dtype,  # Output dtype.
                filter_packed,
                filter_is_cfrs,
                lambdas_have_fusion,
                output_fn,
            ](
                output_buf,
                input_buf,
                filter_buf,
                stride_tuple,
                dilation_tuple,
                pad_d,
                pad_h,
                pad_w,
            )
        else:
            constrained[
                (input.rank == 4 and filter.rank == 4),
                "only rank 4 tensor is supported on cuda gpu",
            ]()
            constrained[
                filter_packed == False,
                "only unpacked filter is supported on cuda gpu",
            ]()

            var cuda_ctx = ctx.get_device_context()
            var pad_tuple = IndexList[input.rank - 2](0)

            @parameter
            if input.rank == 4:
                pad_tuple[0] = pad_h[0]
                pad_tuple[1] = pad_w[0]

            conv_transposed_gpu[
                input.rank,
                filter.rank,
                input._static_shape,
                filter._static_shape,
                output._static_shape,
                input.dtype,
                filter.dtype,
                output.dtype,
                elementwise_epilogue = OptionalReg[
                    elementwise_simd_epilogue_type
                ](output_fn) if lambdas_have_fusion else None,
            ](
                output_buf,
                input_buf,
                filter_buf,
                stride_tuple,
                dilation_tuple,
                pad_tuple,
                cuda_ctx,
            )

    @staticmethod
    fn shape[
        dtype: DType
    ](
        input: InputTensor[dtype=dtype],
        filter: InputTensor[dtype=dtype],
        strides: InputTensor[rank=1],
        dilations: InputTensor[rank=1],
        paddings: InputTensor[rank=1],
        output_paddings: InputTensor[rank=1],
    ) raises -> IndexList[input.rank]:
        return conv_transpose_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output_paddings),
        )


@compiler.register("fold")
struct Fold:
    @staticmethod
    fn execute[
        dtype: DType,
        stride_h: Int,
        stride_w: Int,
        dilation_h: Int,
        dilation_w: Int,
        padding_h: Int,
        padding_w: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        input: InputTensor[dtype=dtype, rank=3],
        output_size: InputTensor,
        kernel_size: InputTensor,
        ctx: DeviceContextPtr,
    ) raises:
        constrained[
            kernel_size.dtype.is_integral() and output_size.dtype.is_integral(),
            "kernel_size and output_size must have integral type",
        ]()
        var output_size_tuple = Index(output_size._ptr[0], output_size._ptr[1])
        var kernel_size_tuple = Index(kernel_size._ptr[0], kernel_size._ptr[1])

        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        fold[
            stride= (stride_h, stride_w),
            dilation= (dilation_h, dilation_w),
            padding= (padding_h, padding_w),
            target=target,
        ](
            input_buf,
            output_buf,
            output_size_tuple,
            kernel_size_tuple,
            ctx,
        )

    @staticmethod
    fn shape[
        dtype: DType,
        stride_h: Int,
        stride_w: Int,
        dilation_h: Int,
        dilation_w: Int,
        padding_h: Int,
        padding_w: Int,
    ](
        input: InputTensor[dtype=dtype, rank=3],
        output_size: InputTensor,
        kernel_size: InputTensor,
    ) raises -> IndexList[4]:
        constrained[
            kernel_size.dtype.is_integral() and output_size.dtype.is_integral(),
            "kernel_size and output_size must have integral type",
        ]()
        var output_size_tuple = Index(output_size._ptr[0], output_size._ptr[1])
        var kernel_size_tuple = Index(kernel_size._ptr[0], kernel_size._ptr[1])
        return fold_shape(
            managed_tensor_slice_to_ndbuffer(input),
            output_size_tuple,
            kernel_size_tuple,
        )


# ===-----------------------------------------------------------------------===#
# FFT kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("irfft")
struct IRFFT:
    @staticmethod
    fn execute[
        target: StaticString,
        dtype: DType,
        rank: Int,
        n: Int,
        buffer_size_mb: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        input: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        irfft(
            input.to_layout_tensor(),
            output.to_layout_tensor(),
            n,
            buffer_size_mb,
            ctx.get_device_context(),
        )


# ===-----------------------------------------------------------------------===#
# Attention kernels
# ===-----------------------------------------------------------------------===#


@compiler.register("masked_flash_attention_gpu")
struct MaskedFlashAttentionGPU:
    @staticmethod
    fn execute[
        target: StaticString, rank: Int
    ](
        output: OutputTensor[rank=rank],
        q: InputTensor[rank=rank],
        k: InputTensor[rank=rank],
        v: InputTensor[rank=rank],
        mask: InputTensor,
        scale: Scalar[dtype = DType.float32],
        ctx: DeviceContextPtr,
    ) raises:
        """`masked_flash_attention_gpu` is a hand-fused operator which does
        something analogous to the following list of operations.

        **Step 0:
        Transpose:
        query_processed = transpose(query) # BSHD --> BHSD
        key_processed = transpose(key)     # BSHD --> BHDS
        value_processed = transpose(value) # BSHD --> BHSD

        **Step 1:
        attentionMatrix = query_processed @ key_processed

        **Step 2:
        norm = broadcast_to(normScalar, shape_of(attentionMatrix))

        **Step 3:
        # Normalize and apply masking
        attentionMatrixNorm = attentionMatrix * scale

        # Note attention_mask is HSS and auto-broadcasts
        attentionMatrixNormMasked = attentionMatrixNorm + attention_mask

        **Step 4:
        # Apply softmax and reproject result
        attentionMatrixSoftMax = softmax(attentionMatrixNormMasked)
        answer = attentionMatrixSoftMax @ value_processed
        answer = transpose(answer) # BHSD --> BSHD

        Compared to the CPU patterns the notable differences are:
        1. The mask is rank 3 and is of shape BSS
        2. The transposes are part of the kernel itself

        Finally, this pattern supports grouped attention patterns. That is if we
        have G groups, then let h = H / G. Key and value are allowed to be BShD
        in these scenarios. Both key and value must be BShD if one is. If this is
        true the following is equivalently run before Step 0:

        ** Step -1:
        key = concat(key, ...) # concat BShD --> BSHD
        value = concat(value, ...) # concat BShD --> BSHD

        The underlying fusion follows ideas taken from the 2022 FlashAttention paper
        by Tri Dao et al.
        """
        constrained[is_gpu[target](), "only valid on GPUs"]()

        var output_buffer = managed_tensor_slice_to_ndbuffer(output)
        var q_buffer = managed_tensor_slice_to_ndbuffer(q)
        var k_buffer = managed_tensor_slice_to_ndbuffer(k)
        var v_buffer = managed_tensor_slice_to_ndbuffer(v)
        var mask_buffer = managed_tensor_slice_to_ndbuffer(mask)

        flash_attention(
            output_buffer,
            q_buffer,
            k_buffer,
            v_buffer,
            mask_buffer,
            scale,
            context=ctx,
        )


@compiler.register("mo.mha.no_cache")
struct FlashAttentionGPU:
    @staticmethod
    fn execute[
        rank: Int, //,
        target: StaticString,
        mask_str: StaticString,
        score_mod_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[rank=rank],
        q: InputTensor[rank=rank],
        k: InputTensor[rank=rank],
        v: InputTensor[rank=rank],
        scale: Float32,
        ctx: DeviceContextPtr,
    ) raises:
        """`mo.mha.no_cache` is a hand-fused operator which does
        something analogous to the following list of operations.

        **Step 0:
        Transpose:
        query_processed = transpose(query) # BSHD --> BHSD
        key_processed = transpose(key)     # BSHD --> BHDS
        value_processed = transpose(value) # BSHD --> BHSD

        **Step 1:
        attentionMatrix = query_processed @ key_processed

        **Step 2:
        norm = broadcast_to(normScalar, shape_of(attentionMatrix))

        **Step 3:
        # Normalize and apply masking
        attentionMatrixNormMasked = mask_functor(attentionMatrix * scale)

        **Step 4:
        # Apply softmax and reproject result
        attentionMatrixSoftMax = softmax(attentionMatrixNormMasked)
        answer = attentionMatrixSoftMax @ value_processed
        answer = transpose(answer) # BHSD --> BSHD

        Compared to the CPU patterns the notable differences are:
        1. The transposes are part of the kernel itself

        Finally, this pattern supports grouped attention patterns. That is if we
        have G groups, then let h = H / G. Key and value are allowed to be BShD
        in these scenarios. Both key and value must be BShD if one is. If this is
        true the following is equivalently run before Step 0:

        ** Step -1:
        key = concat(key, ...) # concat BShD --> BSHD
        value = concat(value, ...) # concat BShD --> BSHD

        The underlying fusion follows ideas taken from the 2022 FlashAttention paper
        by Tri Dao et al.
        """
        constrained[is_gpu[target](), "only valid on GPUs"]()

        var output_buffer = managed_tensor_slice_to_ndbuffer(output)
        var q_buffer = managed_tensor_slice_to_ndbuffer(q)
        var k_buffer = managed_tensor_slice_to_ndbuffer(k)
        var v_buffer = managed_tensor_slice_to_ndbuffer(v)

        alias num_kv_heads = k_buffer.shape.get[
            2
        ]() if k_buffer.shape.has_value[2]() else -1

        @parameter
        @__copy_capture(output_buffer, q_buffer, k_buffer, v_buffer)
        fn _dispatch_flash_attention[
            mask_t: MHAMask, score_mod_t: ScoreModTrait
        ](mask: mask_t, score_mod: score_mod_t) raises:
            alias use_score_mod = not _type_is_eq[
                score_mod_t, IdentityScoreMod
            ]()

            flash_attention[use_score_mod=use_score_mod](
                output_buffer,
                q_buffer,
                k_buffer,
                v_buffer,
                mask,
                score_mod,
                scale,
                ctx[],
            )

        dispatch_mask_and_score_mod[
            mask_str,
            score_mod_str,
            _dispatch_flash_attention,
            local_window_size,
            num_kv_heads,
        ]()


@compiler.register("mo.mha.padded.no_cache")
struct PaddedFlashAttentionGPU:
    @staticmethod
    fn execute[
        rank: Int, //,
        target: StaticString,
        mask_str: StaticString,
        score_mod_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[rank=rank],
        q: InputTensor[rank=rank],
        k: InputTensor[rank=rank],
        v: InputTensor[rank=rank],
        valid_length: InputTensor[dtype = DType.uint32, rank=1],
        scale: Float32,
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        var output_buffer = managed_tensor_slice_to_ndbuffer(output)
        var q_buffer = managed_tensor_slice_to_ndbuffer(q)
        var k_buffer = managed_tensor_slice_to_ndbuffer(k)
        var v_buffer = managed_tensor_slice_to_ndbuffer(v)

        alias valid_length_t = ManagedTensorSlice[
            IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ]
        _valid_length = rebind[valid_length_t](valid_length)

        alias num_kv_heads = k_buffer.shape.get[
            2
        ]() if k_buffer.shape.has_value[2]() else -1

        @parameter
        @__copy_capture(output_buffer, q_buffer, k_buffer, v_buffer)
        fn _dispatch_flash_attention[
            mask_t: MHAMask, score_mod_t: ScoreModTrait
        ](mask: mask_t, score_mod: score_mod_t) raises:
            alias use_score_mod = not _type_is_eq[
                score_mod_t, IdentityScoreMod
            ]()

            flash_attention[
                use_score_mod=use_score_mod,
                _use_valid_length=True,
                _padded_ndbuffer=True,
            ](
                output_buffer,
                q_buffer,
                k_buffer,
                v_buffer,
                mask,
                score_mod,
                scale,
                ctx[],
                valid_length=OptionalReg[valid_length_t](_valid_length),
            )

        dispatch_mask_and_score_mod[
            mask_str,
            score_mod_str,
            _dispatch_flash_attention,
            local_window_size,
            num_kv_heads,
        ]()


@compiler.register("no_mask_flash_attention_cpu")
struct NoMaskFlashAttentionCPU:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        q: InputTensor[dtype=dtype, rank=rank],
        k: FusedInputTensor[dtype=dtype, rank=rank],
        v: FusedInputTensor[dtype=dtype, rank=rank],
        scale: Scalar[dtype = DType.float32],
    ) capturing raises:
        @parameter
        @always_inline
        fn k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.dtype, width]:
            return k._lambda_load[width=width](
                rebind[IndexList[k.rank]](coords)
            )

        @parameter
        @always_inline
        fn v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.dtype, width]:
            return v._lambda_load[width=width](
                rebind[IndexList[v.rank]](coords)
            )

        @parameter
        @always_inline
        fn mask_input_fn[
            width: Int, _rank: Int
        ](idx: IndexList[_rank]) -> SIMD[dtype, width]:
            return SIMD[dtype, width](0)

        nn_flash_attention[k_input_fn, v_input_fn, mask_input_fn](
            managed_tensor_slice_to_ndbuffer(q),
            k.shape(),
            v.shape(),
            IndexList[0](),
            managed_tensor_slice_to_ndbuffer(output),
            scale.cast[DType.float32](),
        )


@compiler.register("with_mask_flash_attention_split_kv_cpu")
struct WithMaskFlashAttentionSplitKVCPU:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        q: InputTensor[dtype=dtype, rank=rank],
        k: FusedInputTensor[dtype=dtype, rank=rank],
        v: FusedInputTensor[dtype=dtype, rank=rank],
        k_cache: FusedInputTensor[dtype=dtype, rank = rank + 1],
        v_cache: FusedInputTensor[dtype=dtype, rank = rank + 1],
        mask: FusedInputTensor[dtype=dtype],
        scale: Scalar[dtype = DType.float32],
    ) capturing raises:
        @parameter
        @always_inline
        fn k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.dtype, width]:
            return k._lambda_load[width=width](
                rebind[IndexList[k.rank]](coords)
            )

        @parameter
        @always_inline
        fn v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.dtype, width]:
            return v._lambda_load[width=width](
                rebind[IndexList[v.rank]](coords)
            )

        @parameter
        @always_inline
        fn k_cache_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k_cache.dtype, width]:
            return k_cache._lambda_load[width=width](
                rebind[IndexList[k_cache.rank]](coords)
            )

        @parameter
        @always_inline
        fn v_cache_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v_cache.dtype, width]:
            return v_cache._lambda_load[width=width](
                rebind[IndexList[v_cache.rank]](coords)
            )

        @parameter
        @always_inline
        fn mask_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[mask.dtype, width]:
            return mask._lambda_load[width=width](
                rebind[IndexList[mask.rank]](coords)
            )

        flash_attention_split_kv[
            k_input_fn,
            v_input_fn,
            k_cache_input_fn,
            v_cache_input_fn,
            mask_input_fn,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            k.shape(),
            v.shape(),
            k_cache.shape(),
            v_cache.shape(),
            mask.shape(),
            managed_tensor_slice_to_ndbuffer(output),
            scale.cast[DType.float32](),
        )

    @staticmethod
    fn shape[
        dtype: DType,
        rank: Int,
    ](
        q: InputTensor[dtype=dtype, rank=rank],
        k: InputTensor[dtype=dtype, rank=rank],
        v: InputTensor[dtype=dtype, rank=rank],
        k_cache: InputTensor[dtype=dtype, rank = rank + 1],
        v_cache: InputTensor[dtype=dtype, rank = rank + 1],
        mask: InputTensor[dtype=dtype],
        scale: Scalar[dtype = DType.float32],
    ) -> IndexList[q.rank]:
        return q.shape()


@compiler.register("with_mask_flash_attention_cpu")
struct WithMaskFlashAttentionCPU:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        q: InputTensor[dtype=dtype, rank=rank],
        k: FusedInputTensor[dtype=dtype, rank=rank],
        v: FusedInputTensor[dtype=dtype, rank=rank],
        mask: FusedInputTensor[dtype=dtype],
        scale: Scalar[dtype = DType.float32],
    ) capturing raises:
        @parameter
        @always_inline
        fn k_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[k.dtype, width]:
            return k._lambda_load[width=width](
                rebind[IndexList[k.rank]](coords)
            )

        @parameter
        @always_inline
        fn v_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[v.dtype, width]:
            return v._lambda_load[width=width](
                rebind[IndexList[v.rank]](coords)
            )

        @parameter
        @always_inline
        fn mask_input_fn[
            width: Int, rank: Int
        ](coords: IndexList[rank]) -> SIMD[mask.dtype, width]:
            return mask._lambda_load[width=width](
                rebind[IndexList[mask.rank]](coords)
            )

        nn_flash_attention[k_input_fn, v_input_fn, mask_input_fn](
            managed_tensor_slice_to_ndbuffer(q),
            k.shape(),
            v.shape(),
            mask.shape(),
            managed_tensor_slice_to_ndbuffer(output),
            scale.cast[DType.float32](),
        )


# ===-----------------------------------------------------------------------===#
# Quantization for CPU
# ===-----------------------------------------------------------------------===#

######
# Q4_0
######


@compiler.register("ggml_q4_0_dequantize")
struct GGMLQ40Dequantize:
    @staticmethod
    @always_inline
    fn execute[
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype = DType.float32, rank=2],
        input: InputTensor[dtype = DType.uint8, rank=2],
    ) raises:
        with Trace[TraceLevel.OP, target = StaticString("cpu")](_trace_name):
            Q4sym[group_size=32].dequantize_and_write_to_tensor(
                managed_tensor_slice_to_ndbuffer(input),
                managed_tensor_slice_to_ndbuffer(output),
                output.shape(),
            )

    @staticmethod
    @always_inline
    fn shape(input: InputTensor[dtype = DType.uint8, rank=2]) -> IndexList[2]:
        alias block_nbytes = sizeof[Q4sym[group_size=32]]()
        alias quants_per_block = 32
        var num_block_per_batch = (
            input.size() // input.dim_size[0]()
        ) // block_nbytes
        return (input.dim_size[0](), quants_per_block * num_block_per_batch)


@compiler.register("vroom_q4_0_matmul")
struct VroomQ40Matmul:
    @staticmethod
    @always_inline
    fn execute[
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype = DType.float32, rank=2],
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) raises:
        with Trace[TraceLevel.OP, target = StaticString("cpu")](_trace_name):
            matmul_qint4[32](
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(c),
            )

    @staticmethod
    @always_inline
    fn shape(
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("vroom_q4_0_repack_weights")
struct VroomQ40RepackWeights:
    @staticmethod
    @always_inline
    fn execute[
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype = DType.uint8, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) raises:
        with Trace[TraceLevel.OP, target = StaticString("cpu")](_trace_name):
            matmul_qint4_pack_b[32](
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
            )

    @staticmethod
    @always_inline
    fn shape(b: InputTensor[dtype = DType.uint8, rank=2]) -> IndexList[2]:
        return b.shape()


######
# Q4_K
######


@compiler.register("ggml_q4_k_dequantize")
struct GGMLQ4KDequantize:
    @staticmethod
    @always_inline
    fn execute[
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype = DType.float32, rank=2],
        input: InputTensor[dtype = DType.uint8, rank=2],
    ) raises:
        with Trace[TraceLevel.OP, target = StaticString("cpu")](_trace_name):
            q4_k_dequantize_impl(
                managed_tensor_slice_to_ndbuffer(input),
                managed_tensor_slice_to_ndbuffer(output),
            )

    @staticmethod
    @always_inline
    fn shape(input: InputTensor[dtype = DType.uint8, rank=2]) -> IndexList[2]:
        alias block_nbytes = sizeof[block_Q4_K]()
        alias elements_per_block = block_QK_K.quantized_k

        var num_block_per_batch = (
            input.size() // input.dim_size[0]()
        ) // block_nbytes

        return (
            input.dim_size[0](),
            elements_per_block * num_block_per_batch,
        )


@compiler.register("vroom_q4_k_matmul")
struct VroomQ4KMatmul:
    @staticmethod
    @always_inline
    fn execute[
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype = DType.float32, rank=2],
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) raises:
        with Trace[TraceLevel.OP, target = StaticString("cpu")](_trace_name):
            matmul_Q4_K(
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(c),
            )

    @staticmethod
    @always_inline
    fn shape(
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("vroom_q4_k_repack_weights")
struct VroomQ4KRepackWeights:
    @staticmethod
    @always_inline
    fn execute[
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype = DType.uint8, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) raises:
        with Trace[TraceLevel.OP, target = StaticString("cpu")](_trace_name):
            matmul_Q4_K_pack_b(
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
            )

    @staticmethod
    @always_inline
    fn shape(
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) -> IndexList[2]:
        return b.shape()


######
# Q6_K
######


@compiler.register("ggml_q6_k_dequantize")
struct GGMLQ6KDequantize:
    @staticmethod
    @always_inline
    fn execute[
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype = DType.float32, rank=2],
        input: InputTensor[dtype = DType.uint8, rank=2],
    ) raises:
        with Trace[TraceLevel.OP, target = StaticString("cpu")](_trace_name):
            q6_k_dequantize_impl(
                managed_tensor_slice_to_ndbuffer(input),
                managed_tensor_slice_to_ndbuffer(output),
                output.shape(),
            )

    @staticmethod
    @always_inline
    fn shape(input: InputTensor[dtype = DType.uint8, rank=2]) -> IndexList[2]:
        alias block_nbytes = sizeof[block_Q6_K]()
        alias elements_per_block = block_QK_K.quantized_k

        var num_block_per_batch = (
            input.size() // input.dim_size[0]()
        ) // block_nbytes

        return (
            input.dim_size[0](),
            elements_per_block * num_block_per_batch,
        )


@compiler.register("vroom_q6_k_matmul")
struct VroomQ6KMatmul:
    @staticmethod
    @always_inline
    fn execute[
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype = DType.float32, rank=2],
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) raises:
        with Trace[TraceLevel.OP, target = StaticString("cpu")](_trace_name):
            matmul_Q6_K(
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(c),
            )

    @staticmethod
    @always_inline
    fn shape(
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("vroom_q6_k_repack_weights")
struct VroomQ6KRepackWeights:
    @staticmethod
    @always_inline
    fn execute[
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype = DType.uint8, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) raises:
        with Trace[TraceLevel.OP, target = StaticString("cpu")](_trace_name):
            matmul_Q6_K_pack_b(
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
            )

    @staticmethod
    @always_inline
    fn shape(
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) -> IndexList[2]:
        return b.shape()


######
# 4-bit quant GPU implementation
######


@compiler.register("qmatmul_b4_g32")
struct QMatmulGPU_b4_g32:
    @staticmethod
    @always_inline
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype = DType.bfloat16, rank=2],
        a: InputTensor[dtype = DType.bfloat16, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target](_trace_name):
            matmul_gpu_qint4[32, target](
                managed_tensor_slice_to_ndbuffer(c),
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("qmatmul_b4_g128")
struct QMatmulGPU_b4_g128:
    @staticmethod
    @always_inline
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        c: OutputTensor[dtype = DType.bfloat16, rank=2],
        a: InputTensor[dtype = DType.bfloat16, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target](_trace_name):
            matmul_gpu_qint4[128, target](
                managed_tensor_slice_to_ndbuffer(c),
                managed_tensor_slice_to_ndbuffer(a),
                managed_tensor_slice_to_ndbuffer(b),
                ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        a: InputTensor[dtype = DType.float32, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) -> IndexList[2]:
        return IndexList[2](a.dim_size[0](), b.dim_size[0]())


@compiler.register("GGUF_gpu_repack_q4_0")
struct QMatmulGPURepackGGUF:
    @staticmethod
    @always_inline
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype = DType.uint8, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target](_trace_name):
            gpu_qint4_repack_Q4_0[b_shape = b.static_spec.shape, target](
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
                ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) -> IndexList[2]:
        return b.shape()


@compiler.register("GPTQ_gpu_repack_b4_g128")
struct QMatmulGPURepackGPTQ_b4_g128:
    @staticmethod
    @always_inline
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype = DType.uint8, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target](_trace_name):
            gpu_qint4_repack_GPTQ[128, target](
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
                ctx=ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        b: InputTensor[dtype = DType.uint8, rank=2],
    ) -> IndexList[2]:
        return IndexList[2](b.dim_size[1](), b.dim_size[0]())


@compiler.register("GPTQ_gpu_repack_b4_g128_desc_act")
struct QMatmulGPURepackGPTQ_b4_g128_desc_act:
    @staticmethod
    @always_inline
    fn execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        b_packed: OutputTensor[dtype = DType.uint8, rank=2],
        b: InputTensor[dtype = DType.uint8, rank=2],
        perm_idx: InputTensor[dtype = DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        with Trace[TraceLevel.OP, target=target](_trace_name):
            gpu_qint4_repack_GPTQ[128, target](
                managed_tensor_slice_to_ndbuffer(b),
                managed_tensor_slice_to_ndbuffer(b_packed),
                rebind[NDBuffer[DType.int32, 1, MutableAnyOrigin]](
                    managed_tensor_slice_to_ndbuffer(perm_idx)
                ),
                ctx=ctx,
            )

    @staticmethod
    @always_inline
    fn shape(
        b: InputTensor[dtype = DType.uint8, rank=2],
        perm_idx: InputTensor[dtype = DType.int32, rank=1],
    ) -> IndexList[2]:
        return IndexList[2](b.dim_size(1), b.dim_size(0))


# ===----------------------------------------------------------------------===#
# KV Cache
# ===-----------------------------------------------------------------------===#


# ===-----------------------------------------------------------------------===#
# Fused QKV matmul
#
# Expected kernel name format:
# mo.fused_qkv_matmul.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qkv_matmul_kv_cache_cont_batch_ragged_kernel_api[
    target: StaticString,
    dtype: DType,
](
    output: ManagedTensorSlice[dtype=dtype, rank=2],
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=2],
    input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    weight: ManagedTensorSlice[dtype=dtype, rank=2],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        input_row_offsets: Tensor with shape (batch_size + 1,).
            The value at each index is the start_idx of the corresponding batch in hidden_state.
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """
    generic_fused_qkv_matmul_kv_cache_cont_batch_ragged[target=target](
        managed_tensor_slice_to_ndbuffer(hidden_state),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        managed_tensor_slice_to_ndbuffer(weight),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@compiler.register("mo.fused_qkv_matmul.ragged.continuous_batching")
struct Struct_fused_qkv_matmul_ragged_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, num_heads: Int, head_dim: Int, //, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        hidden_state: InputTensor[dtype=dtype, rank=2],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight: InputTensor[dtype=dtype, rank=2],
        kv_collection: ContinuousBatchingKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        generic_fused_qkv_matmul_kv_cache_cont_batch_ragged_kernel_api[target](
            output,
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            ctx,
        )


@always_inline
fn generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch_kernel_api[
    target: StaticString,
    dtype: DType,
](
    output: ManagedTensorSlice[dtype=dtype, rank=3],
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=3],
    weight: ManagedTensorSlice[dtype=dtype, rank=2],
    kv_collection: ContinuousBatchingKVCacheCollection,
    layer_idx: UInt32,
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused QKV matmul. Q outputs are written to the output argument
    while K and V outputs are written in-place into k_cache and v_cache.

    Args:
        output: The pre-allocated output buffer for Q projections. K and V
            projections are written in-place to k_cache and v_cache.
        hidden_state: Tensor with shape (batch_size, seq_len, num_heads * head_size).
        weight: Tensor with shape (num_heads * head_size, num_kv_heads * head_size).
        kv_collection: The historical KVCache for keys and values. The KVCache for
            this layer is retrieved via layer_idx.
        layer_idx: The index of the layer being executed. Used to retrieve the KVCache
            for the given layer from kv_collection.
        ctx: The call context pointer, passed by the graph compiler.
    """
    generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch[target=target](
        managed_tensor_slice_to_ndbuffer(hidden_state),
        managed_tensor_slice_to_ndbuffer(weight),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@compiler.register("mo.fused_qkv_matmul.padded.continuous_batching")
struct Struct_fused_qkv_matmul_padded_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, num_heads: Int, head_dim: Int, //, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        hidden_state: InputTensor[dtype=dtype, rank=3],
        weight: InputTensor[dtype=dtype, rank=2],
        kv_collection: ContinuousBatchingKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        generic_fused_qkv_matmul_kv_cache_bshd_continuous_batch_kernel_api[
            target
        ](output, hidden_state, weight, kv_collection, layer_idx, ctx)


@always_inline
fn generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
    dtype: DType,
    weight_type: DType,
    target: StaticString,
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=2],
    input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    weight: ManagedTensorSlice[dtype=weight_type, rank=2],
    kv_collection: PagedKVCacheCollection[
        dtype,
        *_,
    ],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=2],
    ctx: DeviceContextPtr,
) raises:
    generic_fused_qkv_matmul_kv_cache_paged_ragged[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        managed_tensor_slice_to_ndbuffer(hidden_state),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        managed_tensor_slice_to_ndbuffer(weight),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@always_inline
fn generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
    dtype: DType,
    weight_type: DType,
    target: StaticString,
    group_size: OptionalReg[Int] = None,
    has_zp: OptionalReg[Bool] = None,
](
    hidden_state: ManagedTensorSlice[dtype=dtype, rank=2],
    input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    weight: ManagedTensorSlice[dtype=weight_type, rank=2],
    kv_collection: PagedKVCacheCollection[
        dtype,
        *_,
    ],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=2],
    bias: ManagedTensorSlice[dtype=dtype, rank=1],
    ctx: DeviceContextPtr,
) raises:
    generic_fused_qkv_matmul_kv_cache_paged_ragged_bias[
        target=target,
        group_size=group_size,
        has_zp=has_zp,
    ](
        managed_tensor_slice_to_ndbuffer(hidden_state),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        managed_tensor_slice_to_ndbuffer(weight),
        kv_collection,
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        managed_tensor_slice_to_ndbuffer(bias),
        ctx,
    )


@compiler.register("mo.fused_qkv_matmul.ragged.paged")
struct Struct_fused_qkv_matmul_padded_ragged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        hidden_state: InputTensor[dtype=dtype, rank=2],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight: InputTensor[dtype=dtype, rank=2],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
            target=target
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@compiler.register("mo.fused_qkv_matmul.ragged.paged.quantized")
struct Struct_fused_qkv_matmul_padded_ragged_quantized:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        weight_type: DType,
        num_heads: Int,
        head_dim: Int,
        group_size: Int,
        has_zp_int: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        hidden_state: InputTensor[dtype=dtype, rank=2],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight: InputTensor[dtype=weight_type, rank=2],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        # In the group-wise quantization scheme, every `group_size` quantized weights
        # share the same scale. If `has_zp_int` is non-zero, there is also a group-wise
        # zero point that need to be subtracted from the quantized weights.
        alias has_zp = True if has_zp_int == 1 else False

        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api[
            target=target,
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            ctx,
        )


@compiler.register("mo.fused_qkv_matmul.ragged.paged.bias")
struct Struct_fused_qkv_matmul_padded_ragged_bias:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        hidden_state: InputTensor[dtype=dtype, rank=2],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight: InputTensor[dtype=dtype, rank=2],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
            target=target
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            bias,
            ctx,
        )


@compiler.register("mo.fused_qkv_matmul.ragged.paged.scale")
struct Struct_fused_qkv_matmul_padded_ragged_scale:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        scale_type: DType,
        output_type: DType,
        kv_type: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2],
        hidden_state: InputTensor[dtype=dtype, rank=2],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight: InputTensor[dtype=dtype, rank=2],
        input_scale: InputTensor[dtype=scale_type, rank=2],
        weight_scale: InputTensor[dtype=scale_type, rank=2],
        kv_collection: PagedKVCacheCollection[
            kv_type,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        return generic_fused_qkv_matmul_kv_cache_paged_ragged_scale[
            target=target
        ](
            managed_tensor_slice_to_ndbuffer(hidden_state),
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            managed_tensor_slice_to_ndbuffer(weight),
            managed_tensor_slice_to_ndbuffer(input_scale),
            managed_tensor_slice_to_ndbuffer(weight_scale),
            kv_collection,
            layer_idx,
            managed_tensor_slice_to_ndbuffer(output),
            ctx,
        )


@compiler.register("mo.fused_qkv_matmul.ragged.paged.bias.quantized")
struct Struct_fused_qkv_matmul_padded_ragged_bias_quantized:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        weight_type: DType,
        num_heads: Int,
        head_dim: Int,
        group_size: Int,
        has_zp_int: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=2],
        hidden_state: InputTensor[dtype=dtype, rank=2],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight: InputTensor[dtype=weight_type, rank=2],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        bias: InputTensor[dtype=dtype, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        # In the group-wise quantization scheme, every `group_size` quantized weights
        # share the same scale. If `has_zp_int` is non-zero, there is also a group-wise
        # zero point that need to be subtracted from the quantized weights.
        alias has_zp = True if has_zp_int == 1 else False

        return generic_fused_qkv_matmul_kv_cache_paged_ragged_kernel_api_bias[
            target=target,
            group_size=group_size,
            has_zp=has_zp,
        ](
            hidden_state,
            input_row_offsets,
            weight,
            kv_collection,
            layer_idx,
            output,
            bias,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Fused QK RoPE

# Expected kernel name format:
# mo.fused_qk_rope.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qk_rope_bshd_continuous_batch_kernel_api[
    dtype: DType, //,
    *,
    interleaved: Bool,
    target: StaticString,
](
    output: ManagedTensorSlice[dtype=dtype, rank=4],
    q_proj: ManagedTensorSlice[dtype=dtype, rank=4],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: ManagedTensorSlice[dtype=dtype, rank=2],
    layer_idx: UInt32,
    ctx: DeviceContextPtr,
) raises:
    """Performs a fused RoPE projection for Q and K projections.

    We have a manually fused QKV projection with mo.opaque types in our Llama model.
    Due to a limitation in custom op definitions, we can't declare both a tensor
    and opaque type as output from a custom kernel. This requires us to only note
    Q_proj as an output from the QKV projection. If we immediately follow the
    QKV proj kernel with a RoPE kernel applied to K, we'll get a race condition
    because the graph compiler doesn't know about the dependency between these
    kernels in the graph definition. Here we fuse the RoPE kernel applied to
    Q_proj with K_proj, so K_proj RoPE is only executed after QKV completes.
    """
    generic_fused_qk_rope_bshd_continuous_batch[
        interleaved=interleaved, target=target
    ](
        managed_tensor_slice_to_ndbuffer(q_proj),
        kv_collection,
        managed_tensor_slice_to_ndbuffer(freqs_cis),
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@compiler.register("mo.fused_qk_rope.padded.continuous_batching")
struct Struct_fused_qk_rope_padded_continuous_batching[interleaved: Bool]:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, num_heads: Int, head_dim: Int, //, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        q_proj: InputTensor[dtype=dtype, rank=4],
        kv_collection: ContinuousBatchingKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        freqs_cis: InputTensor[dtype=dtype, rank=2],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        generic_fused_qk_rope_bshd_continuous_batch_kernel_api[
            interleaved=interleaved, target=target
        ](
            output,
            q_proj,
            kv_collection,
            freqs_cis,
            layer_idx,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Fused QK Rope Ragged
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_fused_qk_rope_bshd_continuous_batch_ragged_kernel_api[
    dtype: DType, //, *, interleaved: Bool, target: StaticString
](
    output: ManagedTensorSlice[dtype=dtype, rank=3],
    q_proj: ManagedTensorSlice[dtype=dtype, rank=3],
    input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    kv_collection: ContinuousBatchingKVCacheCollection,
    freqs_cis: ManagedTensorSlice[dtype=dtype, rank=2],
    layer_idx: UInt32,
    ctx: DeviceContextPtr,
) raises:
    generic_fused_qk_rope_bshd_continuous_batch_ragged[
        interleaved=interleaved, target=target
    ](
        managed_tensor_slice_to_ndbuffer(q_proj),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        kv_collection,
        managed_tensor_slice_to_ndbuffer(freqs_cis),
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        ctx,
    )


@compiler.register("mo.fused_qk_rope.ragged.continuous_batching")
struct Struct_fused_qk_rope_bshd_continuous_batch_ragged[interleaved: Bool]:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, num_heads: Int, head_dim: Int, //, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        q_proj: InputTensor[dtype=dtype, rank=3],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: ContinuousBatchingKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        freqs_cis: InputTensor[dtype=dtype, rank=2],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        generic_fused_qk_rope_bshd_continuous_batch_ragged_kernel_api[
            interleaved=interleaved, target=target
        ](
            output,
            q_proj,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            ctx,
        )


@always_inline
fn generic_fused_qk_rope_bshd_paged_ragged_kernel_api[
    dtype: DType, //,
    *,
    interleaved: Bool,
    target: StaticString,
](
    q_proj: ManagedTensorSlice[dtype=dtype, rank=3],
    input_row_offsets: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    kv_collection: PagedKVCacheCollection[
        dtype,
        *_,
    ],
    freqs_cis: ManagedTensorSlice[dtype=dtype, rank=2],
    layer_idx: UInt32,
    output: ManagedTensorSlice[dtype=dtype, rank=3],
    context: DeviceContextPtr,
) raises:
    generic_fused_qk_rope_bshd_paged_ragged[
        interleaved=interleaved, target=target
    ](
        managed_tensor_slice_to_ndbuffer(q_proj),
        managed_tensor_slice_to_ndbuffer(input_row_offsets),
        kv_collection,
        managed_tensor_slice_to_ndbuffer(freqs_cis),
        layer_idx,
        managed_tensor_slice_to_ndbuffer(output),
        context,
    )


@compiler.register("mo.fused_qk_rope.ragged.paged")
struct Struct_fused_qk_rope_ragged_paged[interleaved: Bool]:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        q_proj: InputTensor[dtype=dtype, rank=3],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        freqs_cis: InputTensor[dtype=dtype, rank=2],
        layer_idx: UInt32,
        context: DeviceContextPtr = DeviceContextPtr(),
    ) raises:
        generic_fused_qk_rope_bshd_paged_ragged_kernel_api[
            interleaved=interleaved, target=target
        ](
            q_proj,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            output,
            context,
        )


# ===-----------------------------------------------------------------------===#
# MHA
#
# Expected kernel name format:
# mo.mha.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.mha.padded.continuous_batching.tensor_mask")
struct Struct_mha_padded_continuous_batching_tensor_mask:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int, //,
        score_mod_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        q: InputTensor[dtype=dtype, rank=4],
        kv_collection: ContinuousBatchingKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: UInt32,
        mask: InputTensor[dtype=dtype],
        valid_lengths: InputTensor[dtype = DType.uint32, rank=1],
        scale: Float32,
        context: DeviceContextPtr,
    ) raises:
        generic_flash_attention_kv_cache_padded_materialized_mask[
            target=target,
            score_mod_str=score_mod_str,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            kv_collection,
            layer_idx,
            managed_tensor_slice_to_ndbuffer(mask),
            valid_lengths,
            scale,
            managed_tensor_slice_to_ndbuffer(output),
            context,
        )


@compiler.register("mo.mha.padded.continuous_batching")
struct Struct_mha_padded_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int, //,
        mask_str: StaticString,
        score_mod_str: StaticString,
        target: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[dtype=dtype, rank=4],
        q: InputTensor[dtype=dtype, rank=4],
        kv_collection: ContinuousBatchingKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: UInt32,
        valid_lengths: InputTensor[dtype = DType.uint32, rank=1],
        scale: Float32,
        context: DeviceContextPtr,
    ) raises:
        generic_flash_attention_kv_cache_padded[
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            kv_collection,
            layer_idx,
            valid_lengths,
            scale,
            managed_tensor_slice_to_ndbuffer(output),
            context,
        )


@compiler.register("mo.mha.ragged.continuous_batching")
struct Struct_mha_ragged_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int, //,
        mask_str: StaticString,
        score_mod_str: StaticString,
        target: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        q: InputTensor[dtype=dtype, rank=3],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: ContinuousBatchingKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: UInt32,
        scale: Float32,
        context: DeviceContextPtr,
    ) raises:
        generic_flash_attention_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            managed_tensor_slice_to_ndbuffer(output),
            context,
        )


@compiler.register("mo.mha.ragged.paged")
struct Struct_mha_ragged_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
        mask_str: StaticString,
        score_mod_str: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        q: InputTensor[dtype=dtype, rank=3],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        scale: Float32,
        context: DeviceContextPtr,
    ) raises:
        generic_flash_attention_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            input_row_offsets,
            kv_collection,
            layer_idx,
            scale,
            managed_tensor_slice_to_ndbuffer(output),
            context,
        )


# ===-----------------------------------------------------------------------===#
# MLA
#
# Expected kernel name format:
# mo.mla.<prefill/decode>.ragged.paged
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.mla.decode.ragged.paged")
struct Struct_mla_decode_ragged_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        mask_str: StaticString,
        score_mod_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        q: InputTensor[dtype=dtype, rank=3],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        scale: Float32,
        context: DeviceContextPtr,
    ) raises:
        generic_flare_mla_decode_kv_cache_ragged[
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            kv_collection,
            layer_idx,
            scale,
            managed_tensor_slice_to_ndbuffer(output),
            context,
        )


@compiler.register("mo.mla.prefill.init.ragged.paged")
struct Struct_mla_prefill_init_ragged_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        softmax_type: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        mask_str: StaticString,
        score_mod_str: StaticString,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        softmax_info: OutputTensor[dtype=softmax_type, rank=3],
        q: InputTensor[dtype=dtype, rank=3],
        k: InputTensor[dtype=dtype, rank=3],
        v: InputTensor[dtype=dtype, rank=3],
        buffer_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        cache_offsets: InputTensor[dtype = DType.uint32, rank=1],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        scale: Float32,
        context: DeviceContextPtr,
    ) raises:
        generic_flare_mla_prefill_kv_cache_ragged[
            write_softmax_info=True,
            use_cascade_attention=False,
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            managed_tensor_slice_to_ndbuffer(k),
            managed_tensor_slice_to_ndbuffer(v),
            managed_tensor_slice_to_ndbuffer(buffer_row_offsets),
            managed_tensor_slice_to_ndbuffer(cache_offsets),
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            kv_collection,
            layer_idx,
            scale,
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(softmax_info),
            context,
        )


@compiler.register("mo.mla.prefill.ragged.paged")
struct Struct_mla_prefill_ragged_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        softmax_type: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
        mask_str: StaticString,
        score_mod_str: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        softmax_info: OutputTensor[dtype=softmax_type, rank=3],
        q: InputTensor[dtype=dtype, rank=3],
        k: InputTensor[dtype=dtype, rank=3],
        v: InputTensor[dtype=dtype, rank=3],
        buffer_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        cache_offsets: InputTensor[dtype = DType.uint32, rank=1],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        scale: Float32,
        prev_output: InputTensor[dtype=dtype, rank=3],
        prev_softmax_info: InputTensor[dtype=softmax_type, rank=3],
        context: DeviceContextPtr,
    ) raises:
        var prev_output_nd = managed_tensor_slice_to_ndbuffer(prev_output)
        var prev_softmax_info_nd = managed_tensor_slice_to_ndbuffer(
            prev_softmax_info
        )
        generic_flare_mla_prefill_kv_cache_ragged[
            write_softmax_info=True,
            use_cascade_attention=True,
            target=target,
            mask_str=mask_str,
            score_mod_str=score_mod_str,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            managed_tensor_slice_to_ndbuffer(k),
            managed_tensor_slice_to_ndbuffer(v),
            managed_tensor_slice_to_ndbuffer(buffer_row_offsets),
            managed_tensor_slice_to_ndbuffer(cache_offsets),
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            kv_collection,
            layer_idx,
            scale,
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(softmax_info),
            context,
            OptionalReg[NDBuffer[dtype, 3, MutableAnyOrigin]](prev_output_nd),
            OptionalReg[NDBuffer[softmax_type, 3, MutableAnyOrigin]](
                prev_softmax_info_nd
            ),
        )


@compiler.register("mo.mla.prefill.ragged.plan")
struct Struct_mla_prefill_ragged_plan:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        buffer_row_offsets: OutputTensor[dtype = DType.uint32, rank=2],
        cache_offsets: OutputTensor[dtype = DType.uint32, rank=2],
        buffer_lengths: OutputTensor[dtype = DType.int32, rank=1],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        buffer_tok_size: UInt32,
        context: DeviceContextPtr,
    ) raises:
        generic_flare_mla_prefill_ragged_paged_plan[target=target](
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            kv_collection,
            layer_idx,
            buffer_tok_size,
            managed_tensor_slice_to_ndbuffer(buffer_row_offsets),
            managed_tensor_slice_to_ndbuffer(cache_offsets),
            managed_tensor_slice_to_ndbuffer(buffer_lengths),
            context,
        )


@compiler.register("mo.mla.decompress.k.cache.ragged.paged")
struct Struct_mla_decompress_k_cache_ragged_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        k_latent_buffer: OutputTensor[dtype=dtype, rank=2],
        k_buffer: OutputTensor[dtype=dtype, rank=2],
        buffer_row_offsets_1d: InputTensor[dtype = DType.uint32, rank=1],
        cache_offsets_1d: InputTensor[dtype = DType.uint32, rank=1],
        buffer_length: Int32,
        weight: InputTensor[dtype=dtype, rank=2],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        context: DeviceContextPtr,
    ) raises:
        generic_flare_mla_decompress_k_cache_ragged_paged[target=target](
            managed_tensor_slice_to_ndbuffer(buffer_row_offsets_1d),
            managed_tensor_slice_to_ndbuffer(cache_offsets_1d),
            buffer_length,
            managed_tensor_slice_to_ndbuffer(weight),
            kv_collection,
            layer_idx,
            managed_tensor_slice_to_ndbuffer(k_latent_buffer),
            managed_tensor_slice_to_ndbuffer(k_buffer),
            context,
        )


@compiler.register("mo.kv_cache.get_max_seq_len.paged")
struct Struct_kv_cache_get_max_seq_len_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        max_seq_len: OutputTensor[dtype = DType.uint32, rank=1],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        context: DeviceContextPtr,
    ) raises:
        # TODO: use max_lengths[0, 0] in the graphcause a CUDA_INVALID_MEMORY_ACCESS error,
        # as the graph compiler assumes it is a GPU tensor, and inserts a DtoH copy.
        max_seq_len[0] = kv_collection.max_seq_length


# ===-----------------------------------------------------------------------===#
# Cross attention
#
# Expected kernel name format:
# mo.cross_attention.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.cross_attention.ragged.paged")
struct Struct_cross_attention_ragged_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        mask_str: StaticString,
        score_mod_str: StaticString,
        target: StaticString,
        local_window_size: Int = -1,
    ](
        output: OutputTensor[dtype=dtype, rank=3],
        q: InputTensor[dtype=dtype, rank=3],
        q_input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        q_max_seq_len: InputTensor[dtype = DType.uint32, rank=1],
        kv_input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        scale: Float32,
        context: DeviceContextPtr,
    ) raises:
        generic_cross_attention_kv_cache[
            mask_str=mask_str,
            score_mod_str=score_mod_str,
            local_window_size=local_window_size,
            target=target,
        ](
            managed_tensor_slice_to_ndbuffer(q),
            q_input_row_offsets,
            managed_tensor_slice_to_ndbuffer(q_max_seq_len),
            managed_tensor_slice_to_ndbuffer(kv_input_row_offsets),
            kv_collection,
            layer_idx,
            scale,
            managed_tensor_slice_to_ndbuffer(output),
            context,
        )


# ===-----------------------------------------------------------------------===#
# Mixture of Experts
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.moe.create.indices")
struct Struct_moe_create_indices:
    @always_inline
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        token_expert_order: OutputTensor[dtype = DType.uint32, rank=1],
        expert_start_indices: OutputTensor[dtype = DType.uint32, rank=1],
        restore_token_order: OutputTensor[dtype = DType.uint32, rank=1],
        expert_ids: OutputTensor[dtype = DType.uint32, rank=1],
        expert_usage_stats: OutputTensor[dtype = DType.uint32, rank=1],
        topk_ids: InputTensor[dtype = DType.uint32, rank=1],
        context: DeviceContextPtr,
    ) raises:
        moe_create_indices[input_type = DType.uint32, target=target](
            token_expert_order.to_layout_tensor(),
            expert_start_indices.to_layout_tensor(),
            restore_token_order.to_layout_tensor(),
            expert_ids.to_layout_tensor(),
            expert_usage_stats.to_layout_tensor(),
            topk_ids.to_layout_tensor(),
            context,
        )


@compiler.register("mo.grouped.matmul.ragged")
struct Struct_grouped_matmul_ragged:
    @always_inline
    @staticmethod
    fn execute[
        c_type: DType,
        a_type: DType,
        b_type: DType, //,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2],
        a: InputTensor[dtype=a_type, rank=2],
        b: InputTensor[dtype=b_type, rank=3],
        expert_start_indices: InputTensor[dtype = DType.uint32, rank=1],
        expert_ids: InputTensor[dtype = DType.uint32, rank=1],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        context: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "grouped matmul only support GPUs"]()
        cuda_ctx = context.get_device_context()
        grouped_matmul(
            managed_tensor_slice_to_ndbuffer(c),
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(b),
            managed_tensor_slice_to_ndbuffer(expert_start_indices),
            managed_tensor_slice_to_ndbuffer(expert_ids),
            Int(max_num_tokens_per_expert),
            Int(num_active_experts),
            cuda_ctx,
        )


# ===-----------------------------------------------------------------------===#
# KV Collection Constructors (Ctor)
#
# Expected kernel name format:
# mo.kv_collection_ctor.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@always_inline
fn generic_get_continuous_cache_kernel_api[
    dtype: DType,
    kv_params: KVCacheStaticParams,
](
    blocks: ManagedTensorSlice[dtype=dtype, rank=6],
    cache_lengths: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    lookup_table: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    max_lengths: ManagedTensorSlice[dtype = DType.uint32, rank=2],
) -> ContinuousBatchingKVCacheCollection[
    dtype,
    kv_params,
]:
    return generic_get_continuous_cache[dtype, kv_params](
        managed_tensor_slice_to_ndbuffer(blocks),
        managed_tensor_slice_to_ndbuffer(cache_lengths),
        managed_tensor_slice_to_ndbuffer(lookup_table),
        managed_tensor_slice_to_ndbuffer(max_lengths),
    )


@compiler.register("mo.kv_collection_ctor.continuous_batching")
struct Struct_kv_collection_ctor_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, num_heads: Int, head_dim: Int, target: StaticString
    ](
        blocks: InputTensor[dtype=dtype, rank=6],
        cache_lengths: InputTensor[dtype = DType.uint32, rank=1],
        lookup_table: InputTensor[dtype = DType.uint32, rank=1],
        max_lengths: InputTensor[dtype = DType.uint32, rank=2],
    ) -> ContinuousBatchingKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads, head_dim),
    ]:
        return generic_get_continuous_cache[
            kv_params = KVCacheStaticParams(num_heads, head_dim)
        ](
            managed_tensor_slice_to_ndbuffer(blocks),
            managed_tensor_slice_to_ndbuffer(cache_lengths),
            managed_tensor_slice_to_ndbuffer(lookup_table),
            managed_tensor_slice_to_ndbuffer(max_lengths),
        )


# ===-----------------------------------------------------------------------===#
# LayoutTransforms
# ===-----------------------------------------------------------------------===#


# TODO(GEX-1492): use filter_rank+1 instead of packed_filter_rank
fn layout_transform_conv_transpose_filter_common[
    dtype: DType,
    filter_rank: Int,
    packed_filter_rank: Int,
](
    packed_filter: ManagedTensorSlice[dtype=dtype, rank=packed_filter_rank],
    filter: ManagedTensorSlice[dtype=dtype, rank=filter_rank],
):
    constrained[filter_rank + 1 == packed_filter_rank]()
    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_transpose_filter(
        managed_tensor_slice_to_ndbuffer(filter),
        managed_tensor_slice_to_ndbuffer(packed_filter),
        1,
    )


@compiler.register("layout_transform_RSFC_to_FRSCf")
struct LayoutTransformRSFC2FRSCf:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_filter_rank],
        filter: InputTensor[dtype=dtype, rank=filter_rank],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@compiler.register("layout_transform_QRSFC_to_FQRSCf")
struct LayoutTransformQRSFC2FQRSCf:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_filter_rank],
        filter: InputTensor[dtype=dtype, rank=filter_rank],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@compiler.register("pack_conv_filter_shape")
struct PackConvFilterShape:
    @always_inline
    @staticmethod
    fn execute(filter_buf: InputTensor) raises:
        raise Error("Only meant to be used for shape function!")

    @always_inline
    @staticmethod
    fn shape[
        rank: Int,
        filter_type: DType,
        input_shape: DimList,
        filter_shape: DimList,
        output_shape: DimList,
        strides: DimList,
        dilations: DimList,
        paddings: DimList,
        num_groups: Int,
    ](filter_buf: InputTensor[dtype=filter_type, rank=rank]) -> IndexList[
        rank + 1
    ]:
        """
        Compute the output shape of convolution filter packing.

        Parameters:
            rank: Rank of the un-packed filter.
            filter_type: Type of the filter.
            input_shape: NHWC layout.
            filter_shape: Filter shape.
            output_shape: NHWC layout.
            strides: Should be rank 1 size 2.
            dilations: Should be rank 1 size 2.
            paddings: Should be rank 1 size 4.
            num_groups: The number of groups in the convolution.

        Args:
            filter_buf: The filter to be packed.

        Returns:
            The output shape.
        """

        return pack_filter_shape_conv[
            filter_type,
            input_shape,
            filter_shape,
            output_shape,
            strides,
            dilations,
            paddings,
            num_groups,
            False,
        ](managed_tensor_slice_to_ndbuffer(filter_buf))


@compiler.register("pack_conv_transpose_filter_shape")
struct PackConvTransposeFilterShape:
    @always_inline
    @staticmethod
    fn execute[
        rank: Int,
        filter_type: DType,
    ](filter_buf: NDBuffer[filter_type, rank, MutableAnyOrigin]) raises:
        raise Error("Only meant to be used for shape function!")

    @always_inline
    @staticmethod
    fn shape[
        rank: Int,
        filter_type: DType,
    ](filter_buf: NDBuffer[filter_type, rank, MutableAnyOrigin]) -> IndexList[
        rank + 1
    ]:
        return pack_filter_shape_conv_transpose(filter_buf, 1)


# Wrapper that take `num_groups` as a parameter.
# This is required unti `mo.layout.transform` passes `num_groups` as a runtime
# value.
fn layout_transform_conv_filter_common[
    dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
](
    packed_filter: ManagedTensorSlice[dtype=dtype, rank=packed_rank],
    filter: ManagedTensorSlice[dtype=dtype, rank=filter_rank],
):
    constrained[packed_rank == filter_rank + 1]()

    # last param is num_groups which is currently not an available
    # arg for the MO level op
    _pack_conv_filter(
        managed_tensor_slice_to_ndbuffer(filter),
        managed_tensor_slice_to_ndbuffer(packed_filter),
        num_groups,
    )


@compiler.register("layout_transform_QRSCF_to_FQRSCf")
struct LayoutTransformQRSCF2FQRSCf:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank],
        filter: InputTensor[dtype=dtype, rank=filter_rank],
    ):
        layout_transform_conv_filter_common[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_RSCF_to_FRSCf")
struct LayoutTransformRSCF2FRSCf:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank],
        filter: InputTensor[dtype=dtype, rank=filter_rank],
    ):
        layout_transform_conv_filter_common[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_KN_to_KNkni")
struct LayoutTransformMatmulKN2KNkni:
    @always_inline
    @staticmethod
    fn execute[
        a_type: DType,
        a_shape: DimList,
        b_type: DType,
        b_shape: DimList,
        c_type: DType,
        c_shape: DimList,
    ](
        output_buffer: OutputTensor[dtype=b_type, rank=2],
        b_input: InputTensor[dtype=b_type, rank=2],
    ) raises:
        # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
        var kernel_type_m = 0

        @parameter
        if a_shape.at[0]().has_value():
            kernel_type_m = a_shape.at[0]().get()
        _pack_b_ndbuffer_impl[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            transposed=False,
        ](
            managed_tensor_slice_to_ndbuffer(b_input),
            managed_tensor_slice_to_ndbuffer(output_buffer),
            kernel_type_m,
        )


@compiler.register("layout_transform_NK_to_KNkni")
struct LayoutTransformMatmulNK2KNkni:
    @always_inline
    @staticmethod
    fn execute[
        a_type: DType,
        a_shape: DimList,
        b_type: DType,
        b_shape: DimList,
        c_type: DType,
        c_shape: DimList,
    ](
        output_buffer: OutputTensor[dtype=b_type, rank=2],
        b_input: InputTensor[dtype=b_type, rank=2],
    ) raises:
        # NOTE `get_kernel_type` expects `m == 0` for dynamic M.
        var kernel_type_m = 0

        @parameter
        if a_shape.at[0]().has_value():
            kernel_type_m = a_shape.at[0]().get()
        _pack_b_ndbuffer_impl[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            transposed=True,
        ](
            managed_tensor_slice_to_ndbuffer(b_input),
            managed_tensor_slice_to_ndbuffer(output_buffer),
            kernel_type_m,
        )


@compiler.register("pack_matmul_b_shape_func")
struct PackMatmulBShapeFunc:
    @always_inline
    @staticmethod
    fn execute(b_input: InputTensor) raises:
        raise Error("Only meant to be used for shape function!")

    @always_inline
    @staticmethod
    fn shape[
        a_type: DType,
        a_shape: DimList,
        b_type: DType,
        b_shape: DimList,
        c_type: DType,
        c_shape: DimList,
        transpose_in_0: Bool,
    ](b_input: InputTensor[dtype=b_type, rank=2]) -> IndexList[2]:
        return pack_matmul_b_shape_func[
            a_type,
            a_shape,
            b_type,
            b_shape,
            c_type,
            c_shape,
            transpose_in_0,
            False,
        ](managed_tensor_slice_to_ndbuffer(b_input))


# ===-----------------------------------------------------------------------===#
# RMSNorm
#
# Expected kernel name format:
# mo.rms_norm_kv_cache.<padded/ragged>.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.rms_norm_kv_cache.ragged.continuous_batching")
struct Struct_rms_norm_kv_cache_ragged_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        multiply_before_cast: Bool,
        per_head_norm: Bool, //,
        target: StaticString,
    ](
        kv_collection: ContinuousBatchingKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype],
        layer_idx: UInt32,
        total_seq_len: UInt32,
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight_offset: Scalar[dtype=dtype],
        context: DeviceContextPtr,
    ) raises:
        rms_norm_kv_cache_ragged_continuous_batching[
            target=target,
            multiply_before_cast=multiply_before_cast,
            per_head_norm=per_head_norm,
        ](
            kv_collection,
            managed_tensor_slice_to_ndbuffer(gamma),
            epsilon,
            weight_offset,
            layer_idx,
            total_seq_len,
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            context,
        )


@compiler.register("mo.rms_norm_kv_cache.ragged.paged")
struct Struct_rms_norm_kv_cache_ragged_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int,
        multiply_before_cast: Bool,
        per_head_norm: Bool, //,
        target: StaticString,
    ](
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        gamma: InputTensor[dtype=dtype, rank=1],
        epsilon: Scalar[dtype],
        layer_idx: UInt32,
        total_seq_len: UInt32,
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight_offset: Scalar[dtype],
        context: DeviceContextPtr,
    ) raises:
        rms_norm_kv_cache_ragged_paged[
            target=target,
            multiply_before_cast=multiply_before_cast,
            per_head_norm=per_head_norm,
        ](
            kv_collection,
            managed_tensor_slice_to_ndbuffer(gamma),
            epsilon,
            weight_offset,
            layer_idx,
            total_seq_len,
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            context,
        )


# ===-----------------------------------------------------------------------===#
# Print KV Cache
#
# Expected kernel name format:
# mo.print_kv_cache.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


fn print_kv_cache_cont_batch_generic_kernel_api[
    dtype: DType, //, target: StaticString
](
    valid_lengths: InputTensor[dtype = DType.uint32, rank=1],
    kv_collection: ContinuousBatchingKVCacheCollection[dtype, _],
    layer_idx: UInt32,
    is_print_compact: InputTensor[dtype = DType.bool, rank=1],
    context: DeviceContextPtr,
) raises:
    @parameter
    if is_gpu[target]():
        print_kv_cache_cont_batch_generic_gpu[target](
            managed_tensor_slice_to_ndbuffer(valid_lengths),
            kv_collection,
            layer_idx,
            is_print_compact[0],
            context,
        )
    elif is_cpu[target]():
        print_kv_cache_cont_batch_generic_cpu[target](
            managed_tensor_slice_to_ndbuffer(valid_lengths),
            kv_collection,
            layer_idx,
            is_print_compact[0],
            context,
        )


fn print_kv_cache_paged_generic_kernel_api[
    dtype: DType, //,
    target: StaticString,
    kv_params: KVCacheStaticParams,
    page_size: Int,
](
    valid_lengths: InputTensor[dtype = DType.uint32, rank=1],
    kv_collection: PagedKVCacheCollection[dtype, kv_params, page_size],
    layer_idx: UInt32,
    is_print_compact: InputTensor[dtype = DType.bool, rank=1],
    context: DeviceContextPtr,
) raises:
    @parameter
    if is_gpu[target]():
        print_kv_cache_paged_generic_gpu[target](
            managed_tensor_slice_to_ndbuffer(valid_lengths),
            kv_collection,
            layer_idx,
            True,
            context,
        )
    elif is_cpu[target]():
        print_kv_cache_paged_generic_cpu[target](
            managed_tensor_slice_to_ndbuffer(valid_lengths),
            kv_collection,
            layer_idx,
            is_print_compact[0],
            context,
        )


@compiler.register("mo.print_kv_cache.paged")
struct Struct_print_kv_cache_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        valid_lengths: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        is_print_compact: InputTensor[dtype = DType.bool, rank=1],
        context: DeviceContextPtr,
    ) raises:
        print_kv_cache_paged_generic_kernel_api[target](
            valid_lengths,
            kv_collection,
            layer_idx,
            is_print_compact,
            context,
        )


@compiler.register("mo.print_kv_cache.continuous_batching")
struct Struct_print_kv_cache_continuous_batching:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType, num_heads: Int, head_dim: Int, //, target: StaticString
    ](
        valid_lengths: InputTensor[dtype = DType.uint32, rank=1],
        kv_collection: ContinuousBatchingKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
        ],
        layer_idx: UInt32,
        is_print_compact: InputTensor[dtype = DType.bool, rank=1],
        context: DeviceContextPtr,
    ) raises:
        print_kv_cache_cont_batch_generic_kernel_api[target](
            valid_lengths,
            kv_collection,
            layer_idx,
            is_print_compact,
            context,
        )


# ===-----------------------------------------------------------------------===#
# KV Collection Constructors (Ctor)
#
# Expected kernel name format:
# mo.kv_collection_ctor.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.kv_collection_ctor.paged")
struct Struct_kv_collection_ctor_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int,
        target: StaticString,
    ](
        blocks: InputTensor[dtype=dtype, rank=6],
        cache_lengths: InputTensor[dtype = DType.uint32, rank=1],
        lookup_table: InputTensor[dtype = DType.uint32, rank=2],
        max_lengths: InputTensor[dtype = DType.uint32, rank=2],
    ) -> PagedKVCacheCollection[
        dtype, KVCacheStaticParams(num_heads, head_dim), page_size
    ]:
        return generic_get_paged_cache[
            kv_params = KVCacheStaticParams(num_heads, head_dim),
            page_size=page_size,
        ](
            managed_tensor_slice_to_ndbuffer(blocks),
            managed_tensor_slice_to_ndbuffer(cache_lengths),
            managed_tensor_slice_to_ndbuffer(lookup_table),
            managed_tensor_slice_to_ndbuffer(max_lengths),
        )


# ===-----------------------------------------------------------------------===#
# Matmul KV cache
#
# Expected kernel name format:
# mo.kv_matmul.ragged.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.kv_matmul.ragged.paged")
struct Struct_kv_matmul_ragged_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        hidden_state: InputTensor[dtype=dtype, rank=2],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight: InputTensor[dtype=dtype, rank=2],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        kv_matmul_ragged_paged[target=target](
            managed_tensor_slice_to_ndbuffer(hidden_state),
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            managed_tensor_slice_to_ndbuffer(weight),
            kv_collection,
            layer_idx,
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Matmul K cache
#
# Expected kernel name format:
# mo.k_matmul.ragged.<continuous_batching/paged>
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.k_matmul.ragged.paged")
struct Struct_k_matmul_ragged_paged:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        target: StaticString,
    ](
        hidden_state: InputTensor[dtype=dtype, rank=2],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        weight: InputTensor[dtype=dtype, rank=2],
        kv_collection: PagedKVCacheCollection[
            dtype,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        k_matmul_ragged_paged[target=target](
            managed_tensor_slice_to_ndbuffer(hidden_state),
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            managed_tensor_slice_to_ndbuffer(weight),
            kv_collection,
            layer_idx,
            ctx,
        )


@compiler.register("mo.unfused_qkv_matmul.ragged.paged.gguf_quantized")
struct Struct_unfused_qkv_matmul_ragged_paged_gguf_quantized:
    @always_inline
    @staticmethod
    fn execute[
        num_heads: Int,
        head_dim: Int,
        page_size: Int, //,
        quantization_encoding_q: StaticString,
        quantization_encoding_k: StaticString,
        quantization_encoding_v: StaticString,
    ](
        output: OutputTensor[dtype = DType.float32, rank=2],
        hidden_state: InputTensor[dtype = DType.float32, rank=2],
        input_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        q_weight: InputTensor[dtype = DType.uint8, rank=2],
        k_weight: InputTensor[dtype = DType.uint8, rank=2],
        v_weight: InputTensor[dtype = DType.uint8, rank=2],
        kv_collection: PagedKVCacheCollection[
            DType.float32,
            KVCacheStaticParams(num_heads=num_heads, head_size=head_dim),
            page_size,
        ],
        layer_idx: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        unfused_qkv_matmul_ragged_paged_gguf_quantized[
            quantization_encoding_q,
            quantization_encoding_k,
            quantization_encoding_v,
        ](
            managed_tensor_slice_to_ndbuffer(hidden_state),
            managed_tensor_slice_to_ndbuffer(input_row_offsets),
            managed_tensor_slice_to_ndbuffer(q_weight),
            managed_tensor_slice_to_ndbuffer(k_weight),
            managed_tensor_slice_to_ndbuffer(v_weight),
            kv_collection,
            layer_idx,
            managed_tensor_slice_to_ndbuffer(output),
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Sampling Operations
# ===-----------------------------------------------------------------------===#


@compiler.register("sampler.fused_token_sampling")
struct Struct_fused_token_sampling:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        out_idx_type: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_idxs: OutputTensor[dtype=out_idx_type, rank=rank],
        K: InputTensor[dtype = DType.int64, rank=1],
        max_k: Scalar,
        temperature: InputTensor[dtype = DType.float32, rank=1],
        top_p: InputTensor[dtype = DType.float32, rank=1],
        seed: InputTensor[dtype = DType.uint64, rank=1],
        input: InputTensor[dtype=dtype, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_valid_target[target](), "not a valid target"]()

        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var out_idxs_buf = managed_tensor_slice_to_ndbuffer(out_idxs)
        var K_buf = OptionalReg[NDBuffer[DType.int64, 1, MutableAnyOrigin]](
            managed_tensor_slice_to_ndbuffer(K)
        )
        var temperature_buf = OptionalReg[
            NDBuffer[DType.float32, 1, MutableAnyOrigin]
        ](managed_tensor_slice_to_ndbuffer(temperature))
        var top_p_buf = OptionalReg[
            NDBuffer[DType.float32, 1, MutableAnyOrigin]
        ](managed_tensor_slice_to_ndbuffer(top_p))
        var seed_buf = OptionalReg[NDBuffer[DType.uint64, 1, MutableAnyOrigin]](
            managed_tensor_slice_to_ndbuffer(seed)
        )
        with Trace[TraceLevel.OP, target=target](_trace_name):

            @parameter
            if is_cpu[target]():
                # When top_k == 1, argmax is equivalent to our topk_fused_sampling with k == 1
                # However, switching to just using our topk_fused_sampling leads to a -37% perf
                # drop in q4_k benchmarking for llama 3.
                if max_k == 1:
                    argmax(
                        input.to_layout_tensor(),
                        rank - 1,
                        out_idxs.to_layout_tensor(),
                    )
                    return
                _fused_token_sampling_cpu(
                    Int(max_k),
                    input_buf,
                    out_idxs_buf,
                    k=K_buf,
                    temperature=temperature_buf,
                    top_p=top_p_buf,
                    seed=seed_buf,
                )
            else:
                var cuda_ctx = ctx.get_device_context()
                _fused_token_sampling_gpu(
                    cuda_ctx,
                    Int(max_k),
                    input_buf,
                    out_idxs_buf,
                    k=K_buf,
                    temperature=temperature_buf,
                    top_p=top_p_buf,
                    seed=seed_buf,
                )


@compiler.register("min_p_sampling")
struct Struct_min_p_sampling:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        out_idx_type: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_token_ids: OutputTensor[dtype=out_idx_type, rank=rank],
        min_ps: InputTensor[dtype=dtype, rank=1],
        input: InputTensor[dtype=dtype, rank=rank],
        temperature: Scalar[dtype],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_valid_target[target](), "not a valid target"]()

        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var out_token_ids_buf = managed_tensor_slice_to_ndbuffer(out_token_ids)
        var min_ps_buf = managed_tensor_slice_to_ndbuffer(min_ps)
        with Trace[TraceLevel.OP, target=target](_trace_name):

            @parameter
            if is_cpu[target]():
                min_p_sampling_cpu(
                    min_ps.to_layout_tensor(),
                    input.to_layout_tensor(),
                    out_token_ids.to_layout_tensor(),
                    temperature,
                )
            else:
                var cuda_ctx = ctx.get_device_context()
                min_p_sampling_gpu(
                    cuda_ctx,
                    min_ps.to_layout_tensor(),
                    input.to_layout_tensor(),
                    out_token_ids.to_layout_tensor(),
                    temperature,
                )


@compiler.register("sampler.apply_penalties")
struct Struct_sampler_apply_penalties:
    @always_inline
    @staticmethod
    fn execute[
        logit_type: DType,
        penalty_type: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        logits: MutableInputTensor[dtype=logit_type, rank=rank],
        compressed_frequency_data: InputTensor[dtype = DType.int32, rank=2],
        frequency_offsets: InputTensor[dtype = DType.uint32, rank=1],
        frequency_penalty: InputTensor[dtype=penalty_type, rank=1],
        presence_penalty: InputTensor[dtype=penalty_type, rank=1],
        repetition_penalty: InputTensor[dtype=penalty_type, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_valid_target[target](), "not a valid target"]()

        with Trace[TraceLevel.OP, target=target](_trace_name):
            apply_penalties_to_logits[target=target](
                logits.to_layout_tensor(),
                compressed_frequency_data.to_layout_tensor(),
                frequency_offsets.to_layout_tensor(),
                frequency_penalty.to_layout_tensor(),
                presence_penalty.to_layout_tensor(),
                repetition_penalty.to_layout_tensor(),
                ctx,
            )


@compiler.register("sampler.update_frequency_data")
struct Struct_sampler_update_frequency_data:
    @always_inline
    @staticmethod
    fn execute[
        token_type: DType, //,
        target: StaticString,
        _trace_name: StaticString,
    ](
        compressed_frequency_data: MutableInputTensor[
            dtype = DType.int32, rank=2
        ],
        frequency_offsets: InputTensor[dtype = DType.uint32, rank=1],
        new_tokens: InputTensor[dtype=token_type, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_valid_target[target](), "not a valid target"]()

        with Trace[TraceLevel.OP, target=target](_trace_name):
            update_frequency_data[target=target](
                compressed_frequency_data.to_layout_tensor(),
                frequency_offsets.to_layout_tensor(),
                new_tokens.to_layout_tensor(),
                ctx,
            )


# ===-----------------------------------------------------------------------===#
# Misc Operations
# ===-----------------------------------------------------------------------===#


@compiler.register("swishGLU")
struct Struct_swishGLU:
    @always_inline
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        c: OutputTensor[rank=2],
        a: InputTensor[rank=2],
        b0: InputTensor[rank=2],
        b1: InputTensor[dtype = b0.dtype, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        swishGLU[
            a_type = a.static_spec.dtype,
            a_shape = a.static_spec.shape,
            b_type = b0.static_spec.dtype,
            b_shape = b0.static_spec.shape,
            target=target,
        ](
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(b0),
            managed_tensor_slice_to_ndbuffer(b1),
            managed_tensor_slice_to_ndbuffer(c),
            ctx,
        )


@always_inline
fn _check_signal_buffer_size(
    signal_buffer_size: Int, input_size_bytes: Int
) raises:
    # The signal buffer has to be large enough to hold the entire input buffer.
    var min_signal_buffer_size = sizeof[Signal]() + input_size_bytes
    if signal_buffer_size < min_signal_buffer_size:
        raise Error(
            "expected signal buffer to be at least ",
            min_signal_buffer_size,
            " bytes, but got ",
            signal_buffer_size,
        )


@compiler.register("mo.distributed.allreduce.sum")
struct DistributedAllReduceSum:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: FusedOutputVariadicTensors[dtype, rank, *_],
        inputs: InputVariadicTensors[dtype, rank, *_],
        signal_buffers: MutableInputVariadicTensors[
            dtype = DType.uint8, rank=1, *_
        ],
        dev_ctxs_input: DeviceContextPtrList,
    ) capturing raises:
        """Distributed allreduce operation implementation for sum reduction.

        Args:
            outputs: Output tensors (one per GPU) to store reduced results.
            inputs: Input tensors (one per GPU) containing values to reduce.
            signal_buffers: Preallocated synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Implementation Notes:
            1. Uses naive reduction implementation when P2P access unavailable.
            2. Requires input/output buffers to be device-allocated and aligned.
            3. Signal buffers must be device-allocated and large enough to fit
               the buffer + signals metadata.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in allreduce.mojo)
            - Tensor element count must be multiple of SIMD width (per allreduce.mojo)
            - Requires identical tensor shapes across all participating GPUs
        """
        alias num_devices = inputs.size
        constrained[
            signal_buffers.size == num_devices and outputs.size == num_devices,
            (
                "expected allreduce inputs, outputs, and signal buffers to all"
                " have the same number of elements"
            ),
        ]()

        var input_size_bytes = inputs[0].size() * sizeof[dtype]()
        _check_signal_buffer_size(signal_buffers[0].size(), input_size_bytes)

        var dev_ctxs = List[DeviceContext]()
        for i in range(len(dev_ctxs_input)):
            dev_ctxs.append(dev_ctxs_input[i])

        # Marshal input and output variadic tensors into the expected format.
        var in_bufs = InlineArray[
            NDBuffer[dtype, rank, MutableAnyOrigin], inputs.size
        ](NDBuffer[dtype, rank, MutableAnyOrigin]())

        @parameter
        for i in range(inputs.size):
            in_bufs[i] = managed_tensor_slice_to_ndbuffer(inputs[i])

        var out_bufs = InlineArray[
            NDBuffer[dtype, rank, MutableAnyOrigin], num_devices
        ](NDBuffer[dtype, rank, MutableAnyOrigin]())

        @parameter
        for i in range(num_devices):
            out_bufs[i] = managed_tensor_slice_to_ndbuffer(outputs[i])

        var rank_sigs = InlineArray[UnsafePointer[Signal], MAX_GPUS](
            UnsafePointer[Signal]()
        )

        @parameter
        for i in range(signal_buffers.size):
            rank_sigs[i] = signal_buffers[i]._ptr.bitcast[Signal]()

        @always_inline
        @parameter
        fn outputs_lambda[
            input_index: Int,
            _dtype: DType,
            _rank: Int,
            _width: Int,
            *,
            _alignment: Int,
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]) -> None:
            constrained[
                input_index < num_devices, "tensor index out of bounds"
            ]()
            return outputs[input_index]._lambda_store[
                width=_width, element_alignment=_alignment
            ](rebind[IndexList[rank]](coords), rebind[SIMD[dtype, _width]](val))

        with Trace[TraceLevel.OP, target=target](_trace_name):
            allreduce[ngpus=num_devices, outputs_lambda=outputs_lambda](
                in_bufs, out_bufs, rank_sigs, dev_ctxs
            )


@compiler.register("mo.distributed.allgather")
struct DistributedAllGather:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: OutputVariadicTensors[dtype, rank, *_],
        inputs: InputVariadicTensors[dtype, rank, *_],
        signal_buffers: MutableInputVariadicTensors[
            dtype = DType.uint8, rank=1, *_
        ],
        dev_ctxs_input: DeviceContextPtrList,
    ) capturing raises:
        """Distributed allgather operation implementation.

        Args:
            outputs: Output tensors (one per GPU) to store gathered results.
            inputs: Input tensors (one per GPU) containing values to gather.
            signal_buffers: Device buffer values used for synchronization.
            dev_ctxs_input: Device contexts for participating GPUs.
        """
        alias num_devices = inputs.size
        constrained[
            signal_buffers.size == num_devices
            and outputs.size == num_devices * num_devices,
            (
                "expected allgather inputs, signal buffers to have the same"
                " number of elements and outputs to have num_devices *"
                " num_devices"
            ),
        ]()

        var input_size_bytes = inputs[0].size() * sizeof[dtype]()
        _check_signal_buffer_size(signal_buffers[0].size(), input_size_bytes)

        var dev_ctxs = List[DeviceContext]()
        for i in range(len(dev_ctxs_input)):
            dev_ctxs.append(dev_ctxs_input[i])

        # Marshal input and output variadic tensors into the expected format.
        var in_bufs = InlineArray[
            NDBuffer[dtype, rank, MutableAnyOrigin], inputs.size
        ](NDBuffer[dtype, rank, MutableAnyOrigin]())

        @parameter
        for i in range(inputs.size):
            in_bufs[i] = managed_tensor_slice_to_ndbuffer(inputs[i])

        var out_bufs = InlineArray[
            NDBuffer[dtype, rank, MutableAnyOrigin], num_devices * num_devices
        ](NDBuffer[dtype, rank, MutableAnyOrigin]())

        @parameter
        for i in range(num_devices * num_devices):
            out_bufs[i] = managed_tensor_slice_to_ndbuffer(outputs[i])

        var rank_sigs = InlineArray[UnsafePointer[Signal], MAX_GPUS](
            UnsafePointer[Signal]()
        )

        @parameter
        for i in range(signal_buffers.size):
            rank_sigs[i] = signal_buffers[i]._ptr.bitcast[Signal]()

        with Trace[TraceLevel.OP, target=target](_trace_name):
            allgather[ngpus=num_devices](in_bufs, out_bufs, rank_sigs, dev_ctxs)


@compiler.register("mo.distributed.matmul_allreduce")
struct DistributedMatmulAllReduce:
    @staticmethod
    fn execute[
        a_type: DType,
        b_type: DType,
        c_type: DType,
        target: StaticString,
        _trace_name: StaticString,
    ](
        outputs: FusedOutputVariadicTensors[c_type, 2, *_],
        inputs: InputVariadicTensors[a_type, 2, *_],
        weights: InputVariadicTensors[b_type, 2, *_],
        signal_buffers: MutableInputVariadicTensors[
            dtype = DType.uint8, rank=1, *_
        ],
        dev_ctxs_input: DeviceContextPtrList,
    ) capturing raises:
        """Distributed allreduce operation implementation for sum reduction.


        Args:
            outputs: Output tensors (one per GPU) to store reduced results.
            inputs: Input tensors (one per GPU) containing matmul inputs.
            weights: Input tensors (one per GPU) containing matmul inputs.
            signal_buffers: Preallocated synchronization buffers for cross-GPU coordination.
            dev_ctxs_input: Device contexts for participating GPUs.

        Implementation Notes:
            1. Uses naive reduction implementation when P2P access unavailable.
            2. Requires input/output buffers to be device-allocated and aligned.
            3. Signal buffers must be device-allocated and large enough to fit
               the buffer + signals metadata.

        Limitations:
            - Maximum of 8 GPUs supported (matches MAX_GPUS in allreduce.mojo)
            - Tensor element count must be multiple of SIMD width (per allreduce.mojo)
            - Requires identical tensor shapes across all participating GPUs
        """
        alias num_devices = inputs.size
        constrained[
            weights.size == num_devices
            and signal_buffers.size == num_devices
            and outputs.size == num_devices,
            (
                "expected allreduce inputs, weights, outputs, and signal"
                " buffers to all have the same number of elements"
            ),
        ]()

        var input_size_bytes = outputs[0].size() * sizeof[c_type]()
        _check_signal_buffer_size(signal_buffers[0].size(), input_size_bytes)

        var dev_ctxs = List[DeviceContext]()
        for i in range(len(dev_ctxs_input)):
            dev_ctxs.append(dev_ctxs_input[i])

        # Get the static buffer dimensions
        alias n_dim = weights.static_specs[0].shape.at[0]()
        alias k_dim = weights.static_specs[0].shape.at[1]()
        constrained[not n_dim.is_dynamic(), "n dimension should be static"]()
        constrained[not k_dim.is_dynamic(), "k dimension should be static"]()

        alias A_static_shape = DimList(Dim(), k_dim)
        alias B_static_shape = DimList(n_dim, k_dim)
        alias C_static_shape = DimList(Dim(), n_dim)

        # Marshal input and output variadic tensors into the expected format.
        var in_bufs = InlineArray[
            NDBuffer[a_type, 2, MutableAnyOrigin, A_static_shape], num_devices
        ](NDBuffer[a_type, 2, MutableAnyOrigin, A_static_shape]())
        var weight_bufs = InlineArray[
            NDBuffer[b_type, 2, MutableAnyOrigin, B_static_shape], num_devices
        ](NDBuffer[b_type, 2, MutableAnyOrigin, B_static_shape]())

        @parameter
        for i in range(num_devices):
            in_bufs[i] = rebind[
                NDBuffer[a_type, 2, MutableAnyOrigin, A_static_shape]
            ](managed_tensor_slice_to_ndbuffer(inputs[i]))
            weight_bufs[i] = managed_tensor_slice_to_ndbuffer(weights[i])

        var out_bufs = InlineArray[
            NDBuffer[c_type, 2, MutableAnyOrigin, C_static_shape], num_devices
        ](NDBuffer[c_type, 2, MutableAnyOrigin, C_static_shape]())

        @parameter
        for i in range(num_devices):
            out_bufs[i] = rebind[
                NDBuffer[c_type, 2, MutableAnyOrigin, C_static_shape]
            ](managed_tensor_slice_to_ndbuffer(outputs[i]))

        var rank_sigs = InlineArray[UnsafePointer[Signal], MAX_GPUS](
            UnsafePointer[Signal]()
        )

        @parameter
        for i in range(signal_buffers.size):
            rank_sigs[i] = signal_buffers[i]._ptr.bitcast[Signal]()

        @always_inline
        @parameter
        fn outputs_lambda[
            input_index: Int,
            _type: DType,
            _rank: Int,
            _width: Int,
            *,
            _alignment: Int,
        ](coords: IndexList[_rank], val: SIMD[_type, _width]) -> None:
            constrained[
                input_index < num_devices, "tensor index out of bounds"
            ]()
            return outputs[input_index]._lambda_store[
                width=_width, element_alignment=_alignment
            ](rebind[IndexList[2]](coords), rebind[SIMD[c_type, _width]](val))

        # Allocate temporarie buffers to store the matmul outputs
        var c_temp_bufs = InlineArray[
            NDBuffer[c_type, 2, MutableAnyOrigin, C_static_shape], num_devices
        ](NDBuffer[c_type, 2, MutableAnyOrigin, C_static_shape]())

        @parameter
        for i in range(num_devices):
            var device_buffer = dev_ctxs[i].enqueue_create_buffer[c_type](
                out_bufs[i].num_elements()
            )
            c_temp_bufs[i] = NDBuffer[
                c_type, 2, MutableAnyOrigin, C_static_shape
            ](device_buffer.unsafe_ptr(), out_bufs[i].dynamic_shape)

        with Trace[TraceLevel.OP, target=target](_trace_name):
            matmul_allreduce[
                ngpus=num_devices,
                outputs_lambda=outputs_lambda,
            ](in_bufs, weight_bufs, c_temp_bufs, out_bufs, rank_sigs, dev_ctxs)


# Note: this is not a "real" index_tensor op that covers all cases, but rather
# a stopgap measure for some important models (DLRM, CLIP-ViT, LLaMa2)
@compiler.register("index_tensor")
struct IndexTensor:
    @staticmethod
    fn execute[
        dtype: DType,
        indices_type: DType,
        data_rank: Int,
        indices_rank: Int,
        output_rank: Int,
        batch_dims: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=output_rank],
        data: InputTensor[dtype=dtype, rank=data_rank],
        indices: InputTensor[dtype=indices_type, rank=indices_rank],
        ctx: DeviceContextPtr,
    ) raises:
        index_tensor[
            dtype,
            indices_type,
            data_rank,
            indices_rank,
            output_rank,
            batch_dims,
            target=target,
        ](
            managed_tensor_slice_to_ndbuffer(data),
            managed_tensor_slice_to_ndbuffer(indices),
            managed_tensor_slice_to_ndbuffer(output),
            ctx,
        )


# ===-----------------------------------------------------------------------===#
# Advanced Indexing
# ===-----------------------------------------------------------------------===#


@compiler.register("advanced_indexing_getitem")
struct AdvancedIndexingGetItem:
    @always_inline
    @staticmethod
    fn execute[
        input_rank: Int,
        index_rank: Int,
        input_type: DType,
        index_type: DType,
        num_index_tensors: Int, //,
        start_axis: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        out_tensor: OutputTensor[
            dtype=input_type, rank = input_rank + index_rank - num_index_tensors
        ],
        input_tensor: FusedInputTensor[dtype=input_type, rank=input_rank],
        indices: FusedInputVariadicTensors[
            index_type, index_rank, size=num_index_tensors
        ],
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn input_tensor_fn[
            width: Int
        ](idx: IndexList[input_rank]) capturing -> SIMD[input_type, width]:
            return input_tensor._fused_load[width](idx)

        @always_inline
        @parameter
        fn indices_fn[
            indices_index: Int,
        ](coordinates: IndexList[index_rank]) capturing -> SIMD[index_type, 1]:
            constrained[
                indices_index < num_index_tensors, "tensor index out of bounds"
            ]()
            return indices[indices_index]._fused_load[width=1](coordinates)

        advanced_indexing_getitem[
            input_rank=input_rank,
            start_axis=start_axis,
            num_index_tensors=num_index_tensors,
            target=target,
            single_thread_blocking_override=False,
            trace_description=_trace_name,
            input_tensor_fn=input_tensor_fn,
            indices_fn=indices_fn,
        ](
            managed_tensor_slice_to_ndbuffer(out_tensor),
            input_tensor.strides(),
            ctx,
        )

    @always_inline
    @staticmethod
    fn shape[
        input_rank: Int,
        index_rank: Int,
        input_type: DType,
        index_type: DType,
        num_index_tensors: Int, //,
        start_axis: Int,
    ](
        input_tensor: InputTensor[dtype=input_type, rank=input_rank],
        indices: InputVariadicTensors[
            index_type, index_rank, size=num_index_tensors
        ],
    ) -> IndexList[input_rank + index_rank - num_index_tensors]:
        return advanced_indexing_getitem_shape[
            start_axis=start_axis, num_index_tensors=num_index_tensors
        ](input_tensor.shape(), indices[0].shape())


@compiler.register("advanced_indexing_setitem_inplace")
struct AdvancedIndexingSetItemInplace:
    @always_inline
    @staticmethod
    fn execute[
        input_rank: Int,
        index_rank: Int,
        updates_rank: Int,
        input_type: DType,
        index_type: DType,
        num_index_tensors: Int, //,
        start_axis: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        input_tensor: MutableInputTensor[dtype=input_type, rank=input_rank],
        updates: FusedInputTensor[dtype=input_type, rank=updates_rank],
        indices: FusedInputVariadicTensors[
            index_type, index_rank, size=num_index_tensors
        ],
        ctx: DeviceContextPtr,
    ) capturing raises:
        @parameter
        @always_inline
        fn updates_tensor_fn[
            width: Int
        ](idx: IndexList[updates_rank]) capturing -> SIMD[input_type, width]:
            return updates._fused_load[width](idx)

        @always_inline
        @parameter
        fn indices_fn[
            indices_index: Int,
        ](coordinates: IndexList[index_rank]) capturing -> SIMD[index_type, 1]:
            constrained[
                indices_index < num_index_tensors, "tensor index out of bounds"
            ]()
            return indices[indices_index]._fused_load[width=1](coordinates)

        advanced_indexing_setitem_inplace[
            start_axis=start_axis,
            num_index_tensors=num_index_tensors,
            target=target,
            single_thread_blocking_override=False,
            trace_description=_trace_name,
            updates_tensor_fn=updates_tensor_fn,
            indices_fn=indices_fn,
        ](
            managed_tensor_slice_to_ndbuffer(input_tensor),
            indices[0].shape(),
            updates.strides(),
            ctx,
        )


@compiler.register("advanced_indexing_setitem")
struct AdvancedIndexingSetItem:
    @always_inline
    @staticmethod
    fn execute[
        input_rank: Int,
        index_rank: Int,
        updates_rank: Int,
        input_type: DType,
        index_type: DType,
        num_index_tensors: Int, //,
        start_axis: Int,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output_tensor: OutputTensor[dtype=input_type, rank=input_rank],
        input_tensor: FusedInputTensor[dtype=input_type, rank=input_rank],
        updates: FusedInputTensor[dtype=input_type, rank=updates_rank],
        indices: FusedInputVariadicTensors[
            index_type, index_rank, size=num_index_tensors
        ],
        ctx: DeviceContextPtr,
    ) capturing raises:
        """Implement basic numpy-style advanced indexing with assignment but returns a copy.
        """

        # First copy over input tensor into the output
        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: IndexList[output_tensor.rank]) -> SIMD[
            output_tensor.dtype, width
        ]:
            return input_tensor._fused_load[width](idx)

        foreach[
            func,
            target=target,
            _trace_name = _trace_name + "_p1/2_copy",
        ](output_tensor, ctx)

        # Then run the updates in-place.
        # For type checking
        var tensor = MutableInputTensor[
            dtype=input_type,
            rank=input_rank,
            static_spec = output_tensor.static_spec,
        ](
            output_tensor._ptr,
            output_tensor._spec,
            output_tensor._runtime_strides,
        )
        AdvancedIndexingSetItemInplace.execute[
            target=target,
            start_axis=start_axis,
            _trace_name = _trace_name + "_p2/2_update",
        ](tensor, updates, indices, ctx)


# ===-----------------------------------------------------------------------===#
# ArgSort
# ===-----------------------------------------------------------------------===#


@compiler.register("mx.argsort")
struct ArgSort[*, ascending: Bool]:
    @staticmethod
    fn execute[
        target: StaticString
    ](
        indices: OutputTensor[rank=1],
        input: InputTensor[rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            argsort[ascending=ascending](
                indices.to_layout_tensor(), input.to_layout_tensor()
            )
        else:
            var cuda_ctx = ctx.get_device_context()
            argsort[ascending=ascending, target=target](
                indices.to_layout_tensor(), input.to_layout_tensor(), cuda_ctx
            )


# ===-----------------------------------------------------------------------===#
# Float8
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.quantize_static_scaled_float8")
struct QuantizeStaticScaledFloat8[*, scale_is_inverted: Bool]:
    @always_inline
    @staticmethod
    fn execute[
        input_type: DType,
        scale_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype = DType.float8_e4m3fn, rank=2],
        input: InputTensor[dtype=input_type, rank=2],
        scale: Scalar[scale_type],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()
        var scale_loaded = scale.cast[DType.float32]()
        quantize_static_scaled_fp8[scale_is_inverted=scale_is_inverted](
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(input),
            scale_loaded,
            ctx.get_device_context(),
        )


@compiler.register("mo.quantize_dynamic_scaled_float8")
struct QuantizeDynamicScaledFloat8:
    @always_inline
    @staticmethod
    fn execute[
        input_type: DType,
        scales_type: DType,
        output_type: DType, //,
        group_size_or_per_token: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2],
        scales: OutputTensor[dtype=scales_type, rank=2],
        input: InputTensor[dtype=input_type, rank=2],
        scale_ub: Float32,
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        quantize_dynamic_scaled_fp8[group_size_or_per_token](
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(scales),
            managed_tensor_slice_to_ndbuffer(input),
            scale_ub,
            ctx.get_device_context(),
        )


@compiler.register("mo.matmul_dynamic_scaled_fp8")
struct MatmulDynamicScaledFloat8:
    @always_inline
    @staticmethod
    fn execute[
        input_type: DType,
        scales_type: DType,
        output_type: DType, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=output_type, rank=2],
        a: InputTensor[dtype=input_type, rank=2],
        b: InputTensor[dtype=input_type, rank=2],
        a_scales: InputTensor[dtype=scales_type, rank=2],
        b_scales: InputTensor[dtype=scales_type, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        matmul_dynamic_scaled_fp8[transpose_b=True, target=target,](
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(b),
            managed_tensor_slice_to_ndbuffer(a_scales),
            managed_tensor_slice_to_ndbuffer(b_scales),
            ctx.get_device_context(),
        )


@compiler.register("mo.matmul_static_scaled_float8")
struct MatmulStaticScaledFloat8:
    @always_inline
    @staticmethod
    fn execute[
        output_type: DType,
        scale_type: DType,
        target: StaticString,
    ](
        output_tensor: OutputTensor[dtype=output_type, rank=2],
        input_tensor: InputTensor[dtype = DType.float8_e4m3fn, rank=2],
        weight_tensor: InputTensor[dtype = DType.float8_e4m3fn, rank=2],
        input_scale: Scalar[scale_type],
        weight_scale: Scalar[scale_type],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[is_gpu[target](), "only valid on GPUs"]()

        var output = managed_tensor_slice_to_ndbuffer(output_tensor)
        var input = managed_tensor_slice_to_ndbuffer(input_tensor)
        var weight = managed_tensor_slice_to_ndbuffer(weight_tensor)

        @parameter
        @__copy_capture(output, input_scale, weight_scale)
        @always_inline
        fn scaled_output_fn[
            dtype: DType, width: Int, *, alignment: Int = 1
        ](idx: IndexList[2], val: SIMD[dtype, width]):
            var scale = input_scale.cast[dtype]() * weight_scale.cast[dtype]()
            var scaled_val = val * scale

            output.store[width=width, alignment=alignment](
                idx, scaled_val.cast[output_type]()
            )

        # create a dummy buffer to instruct the matmul kernel to output values
        # in the correct type
        alias N = weight.shape.get[0]()
        var M = input.dim[0]()
        var output_dummy = NDBuffer[
            DType.float32, 2, MutableAnyOrigin, DimList(Dim(), N)
        ](
            UnsafePointer[Scalar[DType.float32]](),
            IndexList[2](M, N),
        )

        matmul[
            target=target,
            transpose_b=True,
            elementwise_lambda_fn=scaled_output_fn,
        ](
            output_dummy,
            input,
            weight,
            Optional[DeviceContext](ctx.get_device_context()),
        )


# ===-----------------------------------------------------------------------===#
# Ragged Tensor Operations
# ===-----------------------------------------------------------------------===#


@compiler.register("mo.merge_ragged_tensors")
struct MergeRaggedTensors:
    @always_inline
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int, //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank],
        output_row_offsets: OutputTensor[dtype = DType.uint32, rank=1],
        a: InputTensor[dtype=dtype, rank=rank],
        a_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        b: InputTensor[dtype=dtype, rank=rank],
        b_row_offsets: InputTensor[dtype = DType.uint32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        merge_ragged_tensors[target=target](
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(output_row_offsets),
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(a_row_offsets),
            managed_tensor_slice_to_ndbuffer(b),
            managed_tensor_slice_to_ndbuffer(b_row_offsets),
            ctx,
        )
