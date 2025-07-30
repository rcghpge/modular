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

# TODO: Later PR: Partitioning of Tensor Memory for multiple consumers (is this even needed since only one core? potentially to pipeline the write out of tmem)

from sys import sizeof
from hashlib import default_comp_time_hasher
from math import ceildiv

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList

from gpu import WARP_SIZE, barrier
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.id import block_idx, thread_idx
from gpu.memory import AddressSpace, external_memory, fence_async_view_proxy
from gpu.sync import named_barrier
from gpu.mma_sm100 import *
from gpu.tcgen05 import *
from gpu.mma import st_matrix
from layout.layout_tensor import LayoutTensorIter
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import IntTuple
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    st_matrix_n_layout,
    tile_to_descriptor,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from gpu.cluster import block_rank_in_cluster
from layout.swizzle import make_swizzle
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
    PipelineState,
)
from linalg import vendor_blas

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

# Additional imports for testing
from internal_utils import (
    DeviceNDBuffer,
    HostNDBuffer,
    assert_almost_equal,
    random,
    zero,
)
from internal_utils._utils import ValOrDim, dynamic, static


@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
fn tma_umma_warp_specialized_gemm_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_threads: UInt = 128,
    num_pipeline_stages: UInt = 1,
](
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    num_iters: UInt,
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]  # BM #64
    alias MMA_N = mma_shape[1]  # BN #128
    alias MMA_K = mma_shape[2]  # 16
    alias num_m_mmas = BM // MMA_M  # 1
    alias num_n_mmas = BN // MMA_N  # 1
    alias num_k_mmas = BK // MMA_K  # 4

    alias TMA_BN = 64  # Using half of BN for 2-thread TMA stores

    var one_thread_in_kernel = (
        thread_idx.x == 0 and block_idx.x == 0 and block_idx.y == 0
    )

    var warp_id = thread_idx.x // WARP_SIZE
    var local_thread_id = thread_idx.x % WARP_SIZE

    var is_epilogue_wg = warp_id <= 3
    var is_mainloop_warp = warp_id >= 4

    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()
    alias accum_type = get_accum_type[a_type]()
    alias num_epilogue_warps = 4

    alias c_smem_tile_t = LayoutTensor[
        c_type,
        c_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    base_ptr_smem = rebind[
        UnsafePointer[
            Scalar[a_type], address_space = AddressSpace.SHARED, alignment=128
        ]
    ](
        external_memory[
            Scalar[a_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]()
    )  # pointer to first byte of scratchpad

    alias a_size = a_smem_layout.size()
    alias b_size = b_smem_layout.size()
    alias c_size = c_layout.size()

    constrained[
        ((a_size * sizeof[a_type]()) % 128) == 0, "preserve alignment"
    ]()
    constrained[((b_size * sizeof[b_type]()) % 16) == 0, "preserve alignment"]()
    constrained[
        ((c_size * sizeof[c_type]()) % 128) == 0, "preserve alignment"
    ]()
    var a_smem = LayoutTensorIter[
        a_type,  # dtype (first positional)
        a_smem_layout,  # layout
        MutableAnyOrigin,  # origin
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=True,
    ](
        base_ptr_smem.static_alignment_cast[128](),
        a_size * num_pipeline_stages,  # full ring size
    )

    var b_smem_base = (base_ptr_smem + a_size * num_pipeline_stages).bitcast[
        Scalar[b_type]
    ]()

    var b_smem = LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
        circular=True,
    ](b_smem_base.static_alignment_cast[128](), b_size * num_pipeline_stages)

    var c_smem = (
        (b_smem_base + b_size * num_pipeline_stages)
        .bitcast[Scalar[c_type]]()
        .static_alignment_cast[128]()
    )
    var c_smem_tile = c_smem_tile_t(c_smem)

    # Shared memory pointer to hold tensor memory address, after last smem pointer and expected smem size
    var ptr_tmem_addr = (
        (c_smem + c_size)
        .bitcast[UInt32]()
        .static_alignment_cast[alignment=16]()
    )

    # Arrays of *pointers* to SharedMemBarrier, one per pipeline stage.
    var full_barrier_base = (
        ptr_tmem_addr.bitcast[SharedMemBarrier]() + 1
    )  # ptr_tmem_addr is a pointer of 4 bytes. every addition is a 4 byte offset

    var empty_barrier_base = full_barrier_base + num_pipeline_stages

    var math_barrier_base = empty_barrier_base + num_pipeline_stages
    alias a_expected_bytes = a_size * sizeof[a_type]()
    alias b_expected_bytes = b_size * sizeof[b_type]()
    alias expected_bytes = a_expected_bytes + b_expected_bytes

    alias num_output_warps = 4

    if thread_idx.x == 0:

        @parameter
        for i in range(num_pipeline_stages):
            full_barrier_base[i].init()  # initialise first TMA barrier
            empty_barrier_base[i].init()  # initialise first MMA barrier

        math_barrier_base[].init()

    var producer_state = PipelineState[num_pipeline_stages](0, 1, 0)
    var consumer_state = PipelineState[num_pipeline_stages]()

    # Thread role identification for separate producer/consumer threads

    # something something vibe is, the threads that issue tma and umma must be lane 0 of their warp group
    var producer_warp = warp_id == 4
    var consumer_warp = warp_id == 5

    var elect_one_warp = (
        thread_idx.x // WARP_SIZE == 0
    )  # first warp in the block
    var elect_one_thread = thread_idx.x == 0  # first thread in the block
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    alias max_tmem_cols = 512

    # allocate all 2^18 bytes of smem for tcgen05, all 512 cols allocated
    if elect_one_warp:
        tcgen05_alloc[1](ptr_tmem_addr, max_tmem_cols)

    barrier()

    tmem_addr = ptr_tmem_addr[]

    # give me a tensor for matrices A and B, sliced down to the portion that this CTA is responsible for, and referring to original global tensor
    # so it's the 128 x 64 portion the CTA will copy from global tensor into smem tile A and B
    alias a_canonical_layout = tile_to_descriptor[a_type, a_smem_layout]()
    alias b_canonical_layout = tile_to_descriptor[
        b_type, b_smem_layout, is_k_major=transpose_b
    ]()
    alias aSBO = a_canonical_layout[0].stride[1].value() * sizeof[a_type]()
    alias aLBO = a_canonical_layout[1].stride[1].value() * sizeof[a_type]()
    alias b_stride01 = b_canonical_layout[0].stride[1].value()
    alias b_stride11 = b_canonical_layout[1].stride[1].value()
    alias bSBO = (b_stride01 if transpose_b else b_stride11) * sizeof[b_type]()
    alias bLBO = (b_stride11 if transpose_b else b_stride01) * sizeof[b_type]()

    adesc_base = MMASmemDescriptor.create[aSBO, aLBO, a_swizzle](a_smem[].ptr)
    bdesc_base = MMASmemDescriptor.create[bSBO, bLBO, b_swizzle](b_smem[].ptr)

    idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        accum_type,
        a_type,
        b_type,
        Index[dtype = DType.uint32](mma_shape[0], mma_shape[1]),
        transpose_b=transpose_b,
    ]()

    # ------------------------------------------------------------------
    # Producer CTA thread (threadIdx.x == 0)
    # ------------------------------------------------------------------
    if is_mainloop_warp:
        if producer_warp:
            if local_thread_id == 0:
                for i in range(num_iters):
                    var stage = producer_state.index()
                    var phase = producer_state.phase()
                    empty_barrier_base[stage].wait(phase)

                    # Tell the barrier how many bytes the upcoming TMA transfers will write so that
                    # the consumer can wait on completion.
                    full_barrier_base[stage].expect_bytes(expected_bytes)

                    a_smem_tile = a_smem.next(stage)[]
                    a_tma_op.async_copy(
                        a_smem_tile,
                        full_barrier_base[stage],
                        (UInt(i) * BK, block_idx.y * BM),
                    )

                    b_smem_tile = b_smem.next(stage)[]
                    b_tma_op.async_copy(
                        b_smem_tile,
                        full_barrier_base[stage],
                        (UInt(i) * BK, block_idx.x * BN) if transpose_b else (
                            block_idx.x * BN,
                            UInt(i) * BK,
                        ),
                    )

                    producer_state.step()  # this is i+1%num_pipeline_stages

        # ------------------------------------------------------------------
        # Consumer CTA thread (threadIdx.x == 1)
        # ------------------------------------------------------------------
        if consumer_warp:
            for i in range(num_iters):
                if local_thread_id == 0:
                    var stage = consumer_state.index()
                    var phase = consumer_state.phase()
                    var stage_offset_a = stage * a_expected_bytes
                    var stage_offset_b = stage * b_expected_bytes

                    var cur_adesc = adesc_base + stage_offset_a
                    var cur_bdesc = bdesc_base + stage_offset_b

                    full_barrier_base[stage].wait(
                        phase
                    )  # wait on barrier based on phase, then flip phase bit

                    # Issue UMMA instructions across the K dimension
                    @parameter
                    for j in range(num_k_mmas):
                        alias idx = IntTuple(0, MMA_K * j)
                        alias a_offset = a_smem_layout(idx) * sizeof[a_type]()
                        alias b_offset = b_smem_layout(idx) * sizeof[b_type]()

                        # use c_scale=0 for the first mma only on the first iteration to initialize
                        var c_scale_value: UInt32 = 0 if (
                            i == 0 and j == 0
                        ) else 1
                        mma(
                            cur_adesc + a_offset,
                            cur_bdesc + b_offset,
                            tmem_addr,
                            idesc,
                            c_scale=c_scale_value,
                        )

                    # Signal that UMMA operations consuming this stage have been issued.
                    mma_arrive(empty_barrier_base + stage)

                    # advance to next stage
                    consumer_state.step()
            if local_thread_id == 0:
                _ = math_barrier_base[].arrive()

    if is_epilogue_wg:
        # all threads in the epilogue warp wait for consumer warp to signal that it has finished eating
        math_barrier_base[].wait()
        # eventually all of c has been accumulated, so we load it from tmem_addr into c_frag registers using tcgen05_ld
        alias c_frag_size = MMA_M * MMA_N // 128  # MMA_M * MMA_N is the size of the accumulator, num_threads is the number of threads in the warp, c_frag_size is the num of elements in the accumulator per thread

        c_frag = tcgen05_ld[
            datapaths=16,
            bits=256,
            repeat = BN // 8,
            dtype=accum_type,
            pack=False,
            width=c_frag_size,
        ](tmem_addr)

        tcgen05_load_wait()  # wait for the load to finish

        # Store from tensor memory to smem using st_matrix with swizzling pattern
        warp_id = thread_idx.x // WARP_SIZE

        var st_matrix_rt_layout = RuntimeLayout[
            st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, 1](),
            element_type = DType.int32,
            linear_idx_type = DType.int32,
        ]()

        alias st_matrix_swizzle = make_swizzle[c_type, c_swizzle]()

        @parameter
        for tma_n in range(BN // TMA_BN):

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for i in range(TMA_BN // 16):
                    var d_reg = c_frag.slice[
                        8, offset = (i + tma_n * (TMA_BN // 16)) * 8
                    ]().cast[DType.bfloat16]()

                    var st_matrix_args = RuntimeTuple[
                        IntTuple(
                            UNKNOWN_VALUE,
                            IntTuple(
                                i,
                                m_mma,
                                UNKNOWN_VALUE,
                            ),
                        )
                    ](thread_idx.x, i, m_mma, 0)
                    var offset = c_smem_tile.ptr.offset(
                        st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                        + BM * TMA_BN * tma_n
                    )

                    var d_reg_f32_packed = bitcast[DType.float32, 4](d_reg)

                    st_matrix[simd_width=4](offset, d_reg_f32_packed)
        named_barrier[num_output_warps * WARP_SIZE]()

        # SMEM -> GMEM: Direct TMA store
        # UMMA (tensor memory) → registers → shared memory → global memory
        #           c_frag                   c_smem_tile      c_tma_op
        if elect_one_warp and thread_idx.x < BN // TMA_BN:
            fence_async_view_proxy()

            var smem_offset = c_smem_tile.ptr.offset(BM * TMA_BN * thread_idx.x)

            var c_tma_tile = LayoutTensor[
                c_type,
                c_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.SHARED,
                alignment=128,
            ](smem_offset)

            c_tma_op.async_store(
                c_tma_tile,
                (
                    (block_idx.x * BN + thread_idx.x * TMA_BN),
                    (block_idx.y * BM),
                ),
            )
            c_tma_op.commit_group()
            c_tma_op.wait_group[0]()

        if elect_one_warp:
            tcgen05_release_allocation_lock[1]()
            tcgen05_dealloc[1](tmem_addr, max_tmem_cols)


fn blackwell_matmul_tma_umma[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    num_pipeline_stages: Int = 8,
    block_dim: Int = 32 * 6,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    M: Int,
    N: Int,
    K: Int,
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    # hard coded 64 for BK

    # equivalent of cutlass tma atom a, it is a handle that is passed to async_copy, to accurately tell the TMA engine how to copy from global tensor a into smem tile A
    a_tma_op = create_tma_tile[
        a_type, 2, Index(BM, 64), swizzle_mode=a_swizzle
    ](ctx, a)
    b_tma_op = create_tma_tile[
        b_type,
        2,
        Index(BN, 64) if transpose_b else Index(64, BN),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b)
    c_tma_op = create_tma_tile[BM, 64, swizzle_mode=c_swizzle](ctx, c)

    # Dynamic shared memory usage per CTA:
    #   • A & B stage buffers
    #   • 2 × UInt32 placeholder used to record the allocated TMEM address (ptr_tmem_addr)
    #   • 3 SharedMemBarriers per pipeline stage (TMA, MMA, full)
    #   • 3 descriptors (24 B) kept from the original formula

    alias bytes_per_stage_buffers = (BM * BK * sizeof[a_type]()) + (
        BN * BK * sizeof[b_type]()
    )
    alias a_b_buffers_bytes = bytes_per_stage_buffers * num_pipeline_stages
    alias c_buffer_bytes = BM * BN * sizeof[c_type]()

    alias ptr_tmem_bytes = 2 * sizeof[
        UInt32
    ]()  # ptr_tmem_addr takes two UInt32 words (lane | col | align)
    alias barriers_per_stage = 2
    alias barrier_bytes = sizeof[SharedMemBarrier]() * (
        barriers_per_stage * num_pipeline_stages + 1
    )  # +1 for barrier between consumer and epilogue

    alias smem_use = a_b_buffers_bytes + c_buffer_bytes + ptr_tmem_bytes + barrier_bytes + 24  # 24 not necessary, just for buffer

    alias block_dim_val = block_dim

    alias kernel = tma_umma_warp_specialized_gemm_kernel[
        a_type,
        b_type,
        c_type,
        __type_of(a_tma_op).layout,
        __type_of(b_tma_op).layout,
        __type_of(c_tma_op).layout,
        __type_of(a_tma_op).desc_layout,
        __type_of(b_tma_op).desc_layout,
        __type_of(c_tma_op).desc_layout,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        num_threads=block_dim_val,
        num_pipeline_stages=num_pipeline_stages,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        K // BK,
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim_val),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )


fn get_dic_of_shapes(
    index: Int,
    dic_bro: Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher],
) -> Tuple[Int, Int, Int]:
    try:
        return dic_bro[index]
    except error:
        print("error")
        return (128, 128, 128)


fn make_dic_of_shapes() -> (
    Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher]
):
    var dic = Dict[Int, Tuple[Int, Int, Int], default_comp_time_hasher]()
    dic[0] = (8192, 2560, 8192)
    dic[1] = (4096, 2560, 8192)
    dic[2] = (512, 2560, 8192)
    dic[3] = (8192, 8192, 2048)
    dic[4] = (4096, 8192, 2048)
    dic[5] = (512, 8192, 2048)
    dic[6] = (8192, 14336, 8192)
    dic[7] = (4096, 14336, 8192)
    dic[8] = (512, 14336, 8192)
    dic[9] = (8192, 8192, 7168)
    dic[10] = (4096, 8192, 7168)
    dic[11] = (512, 8192, 7168)
    return dic


fn benchmark_blackwell_matmul(ctx: DeviceContext) raises:
    alias a_type = DType.bfloat16
    alias b_type = DType.bfloat16
    alias c_type = DType.bfloat16
    alias umma_shape = Index(64, 128, 16)
    alias transpose_b = True
    alias BK = 64
    alias block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

    alias dic_of_shapes = make_dic_of_shapes()

    print("Benchmarking blackwell_matmul_tma_umma_kernel")
    print("============================================")
    print("Shapes: [M, N, K]")
    print("Data types: a=", a_type, ", b=", b_type, ", c=", c_type)
    print("UMMA shape:", umma_shape[0], "x", umma_shape[1], "x", umma_shape[2])
    print("BK:", BK)
    print("transpose_b:", transpose_b)
    print()

    alias num_runs = 20
    alias num_warmup = 40

    @parameter
    for i in range(len(dic_of_shapes)):
        alias shape = get_dic_of_shapes(i, dic_of_shapes)
        try:
            print(
                "Benchmarking shape: [",
                shape[0],
                ",",
                shape[1],
                ",",
                shape[2],
                "]",
            )
            alias M = shape[0]
            alias N = shape[1]
            alias K = shape[2]

            alias static_a_shape = DimList(M, K)
            alias static_b_shape = DimList(N, K)
            alias static_c_shape = DimList(M, N)
            var dynamic_a_shape = DimList(M, K)
            var dynamic_b_shape = DimList(N, K)
            var dynamic_c_shape = DimList(M, N)

            var a_host = HostNDBuffer[a_type, 2, static_a_shape](
                dynamic_a_shape
            )
            var b_host = HostNDBuffer[b_type, 2, static_b_shape](
                dynamic_b_shape
            )
            var c_host = HostNDBuffer[c_type, 2, static_c_shape](
                dynamic_c_shape
            )
            var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](
                dynamic_c_shape
            )

            var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
                dynamic_a_shape, ctx=ctx
            )
            var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
                dynamic_b_shape, ctx=ctx
            )
            var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
                dynamic_c_shape, ctx=ctx
            )
            var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
                dynamic_c_shape, ctx=ctx
            )

            # Initialize matmul operands
            random(a_host.tensor)
            random(b_host.tensor)
            zero(c_host.tensor)
            zero(c_host_ref.tensor)

            # Move operands to the Device
            ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
            ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

            ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
            ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

            var a = from_ndbuffer_row_major(a_device.tensor)
            var b = from_ndbuffer_row_major(b_device.tensor)
            _ = from_ndbuffer_row_major(c_device.tensor)

            @parameter
            fn run_kernel(ctx: DeviceContext) raises:
                blackwell_matmul_tma_umma[
                    transpose_b=transpose_b,
                    umma_shape=umma_shape,  # 64, 128, 16
                    block_tile_shape=block_tile_shape,  # 64, 128, 64 (BM, BN, entirety of BK)
                ](
                    c_device.tensor,
                    a_device.tensor,
                    b_device.tensor,
                    M,
                    N,
                    K,
                    ctx,
                )
                ctx.synchronize()

            # Warmup
            for _ in range(num_warmup):
                run_kernel(ctx)
            ctx.synchronize()
            print("finished warmup")
            ### run the benchmark for this shape by enqueuing the kernel and calling it
            var nstime = ctx.execution_time[run_kernel](num_runs) / num_runs
            var sectime = nstime * 1e-9
            var TFlop = 2.0 * M * N * K * 1e-12

            print("  Average time: ", sectime * 1000, " ms")
            print("  Performance: ", TFlop / sectime, " TFLOPS")
            print()
        except e:
            print("Error: Failed to run benchmark for this shape")


def test_blackwell_matmul_tma_umma[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    umma_shape: IndexList[3],
    transpose_b: Bool = True,
    BK: Int = 64,
](ctx: DeviceContext, m: ValOrDim, n: ValOrDim, k: ValOrDim):
    var M = m.value
    var N = n.value
    var K = k.value

    print(
        M,
        "x",
        N,
        "x",
        K,
    )

    alias static_a_shape = DimList(m.dim, k.dim)
    alias static_b_shape = DimList(n.dim, k.dim) if transpose_b else DimList(
        k.dim, n.dim
    )
    alias static_c_shape = DimList(m.dim, n.dim)
    var dynamic_a_shape = DimList(m.value, k.value)
    var dynamic_b_shape = DimList(n.value, k.value) if transpose_b else DimList(
        k.value, n.value
    )
    var dynamic_c_shape = DimList(m.value, n.value)

    var a_host = HostNDBuffer[a_type, 2, static_a_shape](dynamic_a_shape)
    var b_host = HostNDBuffer[b_type, 2, static_b_shape](dynamic_b_shape)
    var c_host = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)
    var c_host_ref = HostNDBuffer[c_type, 2, static_c_shape](dynamic_c_shape)

    var a_device = DeviceNDBuffer[a_type, 2, static_a_shape](
        dynamic_a_shape, ctx=ctx
    )
    var b_device = DeviceNDBuffer[b_type, 2, static_b_shape](
        dynamic_b_shape, ctx=ctx
    )
    var c_device = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )
    var c_device_ref = DeviceNDBuffer[c_type, 2, static_c_shape](
        dynamic_c_shape, ctx=ctx
    )

    # Initialize matmul operands
    random(a_host.tensor)
    random(b_host.tensor)
    zero(c_host.tensor)
    zero(c_host_ref.tensor)

    # Move operands to the Device

    ctx.enqueue_copy(a_device.buffer, a_host.tensor.data)
    ctx.enqueue_copy(b_device.buffer, b_host.tensor.data)

    ctx.enqueue_copy(c_device.buffer, c_host.tensor.data)
    ctx.enqueue_copy(c_device_ref.buffer, c_host_ref.tensor.data)

    var a = from_ndbuffer_row_major(a_device.tensor)
    var b = from_ndbuffer_row_major(b_device.tensor)
    var c = from_ndbuffer_row_major(c_device.tensor)

    alias block_tile_shape = Index(umma_shape[0], umma_shape[1], BK)

    blackwell_matmul_tma_umma[
        transpose_b=transpose_b,
        umma_shape=umma_shape,
        block_tile_shape=block_tile_shape,
    ](
        c_device.tensor,
        a_device.tensor,
        b_device.tensor,
        M,
        N,
        K,
        ctx,
    )

    ctx.synchronize()

    vendor_blas.matmul(
        ctx,
        c_device_ref.tensor,
        a_device.tensor,
        b_device.tensor,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host.tensor.data, c_device.buffer)
    ctx.enqueue_copy(c_host_ref.tensor.data, c_device_ref.buffer)
    ctx.synchronize()
    alias rtol = 1e-2
    assert_almost_equal(
        c_host.tensor,
        c_host_ref.tensor,
        atol=0.0001,
        rtol=rtol,
    )
    print("🚀🚀🚀 Success! 🚀🚀🚀")

    _ = c_device
    _ = c_device_ref
    _ = a_host
    _ = b_host
    _ = c_host_ref
    _ = c_host
    _ = a_device
    _ = b_device

    _ = a
    _ = b
    _ = c


def main():
    with DeviceContext() as ctx:
        test_blackwell_matmul_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 128, 16),
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(128), static[128](), static[128]())

        test_blackwell_matmul_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 128, 16),
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(1024), static[2048](), static[2048]())

        test_blackwell_matmul_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 128, 16),
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(1024), static[2048](), static[2048]())

        test_blackwell_matmul_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 128, 16),
            transpose_b=True,
            BK=64,
        ](ctx, static[1024](), static[2048](), static[2048]())

        test_blackwell_matmul_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 128, 16),
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(100), static[512](), static[256]())

        test_blackwell_matmul_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 128, 16),
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(99), static[1024](), static[1024]())

        test_blackwell_matmul_tma_umma[
            DType.bfloat16,
            DType.bfloat16,
            DType.bfloat16,
            umma_shape = Index(64, 128, 16),
            transpose_b=True,
            BK=64,
        ](ctx, dynamic(201), static[2048](), static[256]())

        # Run the benchmark
        print("\n\n========== Running Benchmarks ==========\n")
        benchmark_blackwell_matmul(ctx)
