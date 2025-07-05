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
from collections import OptionalReg
from math import ceildiv
from sys import simdwidthof, sizeof

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList, _make_tuple
from gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier
from gpu.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    cluster_sync_relaxed,
    elect_one_sync,
)
from gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from gpu.grid_controls import (
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.compile import _compile_code_asm
from gpu.host import get_gpu_target
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import H100
from gpu.id import (
    block_dim,
    block_id_in_cluster,
    block_idx,
    global_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from gpu.mma import (
    WGMMADescriptor,
    st_matrix,
    wgmma_async,
    wgmma_commit_group_sync,
    wgmma_fence_aligned,
    wgmma_wait_group_sync,
)
from layout import IntTuple, Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout
from layout.tensor_core_async import (
    TensorCoreAsync,
    st_matrix_n_layout,
    tile_layout_k_major,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
)
from linalg.matmul_sm90 import (
    _get_c_smem_layout,
    cluster_size,
    consumer_main_loop,
    warp_specialized_gemm_output,
)
from linalg.matmul_loadop_sm90 import async_load_AB
from memory.pointer import _GPUAddressSpace

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from .utils import elementwise_epilogue_type
from .utils_gpu import MatmulConfig, block_swizzle

alias NumWarpPerWarpGroup = WARPGROUP_SIZE // WARP_SIZE

# ===----------------------------------------------------------------------=== #
# Naive grouped matmul
# ===----------------------------------------------------------------------=== #


fn naive_grouped_matmul[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool = True,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    constrained[transpose_b, "Only support transposed B in grouped matmul."]()

    ctx.enqueue_function[
        naive_grouped_matmul_kernel[
            c_type,
            c_shape,
            a_type,
            a_shape,
            b_type,
            b_shape,
        ]
    ](
        c,
        a,
        b,
        a_offsets,
        expert_ids,
        grid_dim=(
            ceildiv(c.dim[1](), 32),
            ceildiv(max_num_tokens_per_expert, 16),
            num_active_experts,
        ),
        block_dim=(32, 16, 1),
    )


fn naive_grouped_matmul_kernel[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
):
    # There has to be a better way :(
    var M: UInt = UInt(a_offsets[block_idx.z + 1] - a_offsets[block_idx.z])
    N = b.dim[1]()
    K = b.dim[2]()

    a_start_row = a_offsets[block_idx.z]
    a_by_expert = a.data + a_start_row * K

    expert = expert_ids[block_idx.z]
    b_by_expert = b.data + expert * N * K

    # indices in current matmul
    n = global_idx.x
    m = global_idx.y

    if n >= N or m >= M:
        return

    alias accum_type = get_accum_type[a_type]()

    var accum = Scalar[accum_type](0.0)

    for k in range(K):
        accum += (
            a_by_expert[m * K + k].cast[accum_type]()
            * b_by_expert[n * K + k].cast[accum_type]()
        )

    c_by_expert = c.data + a_start_row * N
    c_by_expert[m * N + n] = accum.cast[c_type]()


# ===----------------------------------------------------------------------=== #
# H100 grouped matmul
# ===----------------------------------------------------------------------=== #


@always_inline
fn default_config_sm90[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    wgmma_shape: IndexList[3],
]() -> MatmulConfig[a_type, b_type, c_type, transpose_b, wgmma_shape]:
    return MatmulConfig[a_type, b_type, c_type, transpose_b, wgmma_shape,](
        block_tile_shape=Index(128, 256, 64),
        cluster_shape=Index(1, 1, 1),
        num_pipeline_stages=4,
        num_consumer=2,
        partitioned_multicast=False,
    )


fn grouped_matmul_sm90[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
    *,
    transpose_b: Bool = True,
    wgmma_shape: IndexList[3] = Index(64, 256, 16),
    config: MatmulConfig[
        a_type, b_type, c_type, transpose_b, wgmma_shape
    ] = default_config_sm90[
        a_type,
        b_type,
        c_type,
        transpose_b,
        wgmma_shape,
    ](),
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    alias num_experts = b.shape.get[0]()
    alias N = b.shape.get[1]()
    alias K = b.shape.get[2]()

    alias cluster_shape = StaticTuple[Int32, 3](
        config.cluster_shape[0],
        config.cluster_shape[1],
        config.cluster_shape[2],
    )

    alias CLUSTER_N = UInt(cluster_shape[0])
    alias CLUSTER_M = UInt(cluster_shape[1])

    alias c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        config.num_pipeline_stages,
    ]()
    alias c_smem_tile = Index(
        c_smem_layout.shape[0].value(), c_smem_layout.shape[1].value()
    )

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias c_swizzle = TensorMapSwizzle.SWIZZLE_NONE

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]

    # Create TMA op for the entire A tensor including all tokens.
    a_tensor = from_ndbuffer_row_major(a)
    a_tma_op = create_tma_tile[
        a_type,
        2,
        Index(BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a_tensor)

    # Flattne B tensor into a 2D tensor for easier TMA support.
    b_tensor = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        MutableAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ](b.data)
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, BK),
        swizzle_mode=b_swizzle,
    ](ctx, b_tensor)

    # Create a dummy TMA op for C, we don't support TMA store for output.
    c_tensor = from_ndbuffer_row_major(c)
    c_tma_op = create_tma_tile[
        c_type,
        2,
        Index(BM, BK),
        swizzle_mode=c_swizzle,
    ](ctx, c_tensor)

    alias num_threads = WARPGROUP_SIZE * config.num_consumer + WARPGROUP_SIZE
    alias smem_size = Int(config.num_pipeline_stages) * (
        BM * BK * sizeof[a_type]()
        + BN * BK * sizeof[b_type]()
        + (sizeof[Int64]() * 2)
    ) + c_smem_layout.size() * sizeof[c_type]()

    alias kernel = grouped_matmul_kernel[
        a_type,
        b_type,
        c_type,
        __type_of(a_tensor).layout,
        __type_of(b_tensor).layout,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(c_tensor).layout,
        config.block_tile_shape,
        wgmma_shape,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        __type_of(c_tma_op).desc_layout,
        c_smem_layout,
        c_swizzle=c_swizzle,
        cluster_shape=cluster_shape,
        transpose_b=True,
        num_threads=num_threads,
        pipeline_stages = config.num_pipeline_stages,
        use_tma_store=False,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        a_offsets,
        expert_ids,
        c_tensor,
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(max_num_tokens_per_expert, BM),
            num_active_experts,
        ),
        block_dim=(num_threads),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
    `nvvm.cluster_dim`=cluster_shape,
)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
fn grouped_matmul_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    c_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    c_smem_layout: Layout,
    cluster_shape: StaticTuple[Int32, 3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    transpose_b: Bool = True,
    num_threads: Int = 128,
    pipeline_stages: Int = 7,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_smem_layout, c_desc_layout],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
):
    constrained[transpose_b, "Only support transposed B in layout"]()

    alias num_consumer = (num_threads // 128) - 1
    alias num_consumer_threads = num_consumer * 128
    alias CLUSTER_N = UInt(cluster_shape[0])
    alias CLUSTER_M = UInt(cluster_shape[1])
    alias CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

    alias K = b_layout.shape[1].value()
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias a_smem_layout = tile_layout_k_major[a_type, BM, BK, a_swizzle]()
    alias b_smem_layout = tile_layout_k_major[b_type, BN, BK, b_swizzle]()

    alias simd_size = simdwidthof[c_type]()

    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]

    alias accum_type = get_accum_type[a_type]()
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128

    alias use_cluster = cluster_size[cluster_shape]() > 1

    var block_idx_swizzle = block_swizzle(
        Index[dtype = DType.uint32](block_idx.x, block_idx.y),
        Index[dtype = DType.uint32](grid_dim.x, grid_dim.y),
    ) if not use_cluster else Index[dtype = DType.uint32](
        block_idx.x, block_idx.y
    )

    # The block may be OOB because we create blocks based the maximum
    # number of tokens per expert.
    M = a_offsets[block_idx.z + 1] - a_offsets[block_idx.z]
    if UInt32(block_idx_swizzle[1] * BM) >= M:
        return

    a_start_row = a_offsets[block_idx.z]

    alias N = c_layout.shape[1].value()
    expert = expert_ids[block_idx.z]
    b_start_row = expert * N

    wgmma_op = TensorCoreAsync[
        accum_type,
        a_type,
        b_type,
        wgmma_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    var smem = external_memory[
        UInt8, address_space = AddressSpace.SHARED, alignment=8
    ]()

    alias a_smem_size = a_smem_layout.size() * pipeline_stages
    alias b_smem_size = b_smem_layout.size() * pipeline_stages

    alias a_smem_bytes = a_smem_size * sizeof[a_type]()
    alias b_smem_bytes = b_smem_size * sizeof[b_type]()

    alias c_smem_size = c_smem_layout.size()
    alias c_smem_bytes = c_smem_size * sizeof[c_type]()

    var a_smem = smem.bitcast[Scalar[a_type]]()
    var b_smem = (smem + a_smem_bytes).bitcast[Scalar[b_type]]()
    var c_smem = (smem + a_smem_bytes + b_smem_bytes).bitcast[Scalar[c_type]]()
    var smem_poll = (smem + a_smem_bytes + b_smem_bytes + c_smem_bytes).bitcast[
        Int64
    ]()

    var a_smem_iter = LayoutTensorIter[
        a_type,
        a_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=True,
    ](a_smem.static_alignment_cast[128](), a_smem_size)

    var b_smem_iter = LayoutTensorIter[
        b_type,
        b_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=True,
    ](b_smem.static_alignment_cast[128](), b_smem_size)

    var c_smem_tile = LayoutTensor[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](c_smem.static_alignment_cast[128]())

    var a_mbars_ptr = smem_poll.bitcast[Int64]()
    var b_mbars_ptr = smem_poll.bitcast[Int64]() + pipeline_stages

    full = a_mbars_ptr.bitcast[SharedMemBarrier]()
    empty = b_mbars_ptr.bitcast[SharedMemBarrier]()

    var warp_group_idx = thread_idx.x // WARPGROUP_SIZE
    var warp_group_thread_idx = thread_idx.x % WARPGROUP_SIZE
    alias num_k_iters = K // BK

    var rank_m = block_id_in_cluster.y
    var rank_n = block_id_in_cluster.x

    var lane_predicate = elect_one_sync()
    if thread_idx.x == 0:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()

        @parameter
        for i in range(pipeline_stages):
            full[i].init(1)
            empty[i].init(num_consumer * CLUSTER_SIZE)

    # We need this to guarantee that the Pipeline init is visible
    # To all producers and consumer blocks in the cluster
    @parameter
    if cluster_size[cluster_shape]() > 1:
        fence_mbarrier_init()
        cluster_sync_relaxed()
    else:
        barrier()

    if warp_group_idx == 0:
        alias num_regs = 24 if num_consumer <= 2 else 32
        warpgroup_reg_dealloc[num_regs]()
        if warp_group_thread_idx == 0 and lane_predicate:
            var write_pipeline_states = PipelineState[pipeline_stages]()

            var m_coord = (
                block_idx.y * BM if CLUSTER_N
                > 1 else UInt(Int(a_start_row))
                + UInt(block_idx_swizzle[1]) * BM
            )

            var n_coord = (
                block_idx.x * BN if CLUSTER_M
                > 1 else UInt(Int(b_start_row))
                + UInt(block_idx_swizzle[0]) * BN
            )

            async_load_AB[
                block_tile_shape=block_tile_shape,
                cluster_shape=cluster_shape,
                partitioned_multicast=False,
                num_k_iters=num_k_iters,
            ](
                a_tma_op,
                b_tma_op,
                a_smem_iter,
                b_smem_iter,
                m_coord,
                n_coord,
                rank_n,
                rank_m,
                write_pipeline_states,
                empty,
                full,
            )

    else:

        @parameter
        if num_consumer == 1 or num_consumer == 2:
            alias num_regs = 256 if num_consumer == 1 else 240
            warpgroup_reg_alloc[num_regs]()
        else:
            warpgroup_reg_alloc[160]()

        var local_warp_group_idx = warp_group_idx - 1

        var c_reg_tile = LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        var dummy_c_reg_tile = LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        _ = c_reg_tile.fill(0.0)

        @parameter
        for i in range(pipeline_stages):

            @parameter
            if cluster_size[cluster_shape]() > 1:
                if warp_group_thread_idx < CLUSTER_SIZE:
                    _ = empty[i].arrive_cluster(warp_group_thread_idx)
            else:
                if warp_group_thread_idx == 0:
                    _ = empty[i].arrive()

        var read_pipeline_states = PipelineState[pipeline_stages]()

        consumer_main_loop[
            cluster_shape=cluster_shape,
            num_consumer=num_consumer,
            num_k_iters=num_k_iters,
        ](
            dummy_c_reg_tile,
            c_reg_tile,
            a_smem_iter,
            b_smem_iter,
            read_pipeline_states,
            full,
            empty,
            wgmma_op,
            local_warp_group_idx,
            warp_group_thread_idx,
        )

        # C layout for current expert
        alias c_gmem_layout = Layout(IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1))
        alias c_gmem_type = LayoutTensor[
            c_type,
            c_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            address_space = AddressSpace.GENERIC,
        ]

        c_gmem_runtime_layout = RuntimeLayout[
            c_gmem_layout,
            element_type = c_gmem_type.layout_int_type,
            linear_idx_type = c_gmem_type.linear_idx_type,
        ](
            Index[dtype = c_gmem_type.layout_int_type](M, N),
            Index[dtype = c_gmem_type.linear_idx_type](N, 1),
        )

        var c_by_expert = c_gmem_type(
            c.ptr + a_start_row * N, c_gmem_runtime_layout
        )

        warp_specialized_gemm_output[
            c_tile_shape = Index(BM, BN),
            c_swizzle=c_swizzle,
            wgmma_shape=wgmma_shape,
            num_consumer=num_consumer,
            use_tma_store=use_tma_store,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            c_tma_op,
            c_by_expert,
            c_smem_tile,
            c_reg_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            thread_idx.x - WARPGROUP_SIZE,
            block_idx_swizzle[1],
            block_idx_swizzle[0],
        )

    # TO ensure SEMEM destruction doesn't happen
    @parameter
    if cluster_size[cluster_shape]() > 1:
        cluster_sync()


# ===----------------------------------------------------------------------=== #
# Entry Point and Dispatch
# ===----------------------------------------------------------------------=== #


fn grouped_matmul[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList, //,
](
    c: NDBuffer[mut=True, c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    expert_ids: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    alias is_expert_shape_static = b_shape.all_known[3]() and a_shape.has_value[
        1
    ]() and c_shape.has_value[1]()
    alias is_sm90_kernel_applicable = ctx.device_info is H100 and is_expert_shape_static

    @parameter
    if is_sm90_kernel_applicable:
        grouped_matmul_sm90(
            c,
            a,
            a_offsets,
            max_num_tokens_per_expert,
            b,
            expert_ids,
            num_active_experts,
            ctx,
        )
    else:
        naive_grouped_matmul(
            c,
            a,
            b,
            a_offsets,
            expert_ids,
            max_num_tokens_per_expert,
            num_active_experts,
            ctx,
        )
