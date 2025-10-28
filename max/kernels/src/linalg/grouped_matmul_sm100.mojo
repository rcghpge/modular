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
from math import align_up, ceildiv
from memory import bitcast
from sys import align_of, simd_width_of, size_of

from bit import next_power_of_two, prev_power_of_two
from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, barrier
from gpu.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import B200
from gpu import block_id_in_cluster, block_idx, lane_id, thread_idx, grid_dim
from gpu import warp_id as get_warp_id
from gpu.memory import (
    AddressSpace,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from gpu.mma import st_matrix
from gpu.mma_sm100 import *
from gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
)
from gpu.tcgen05 import *
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.int_tuple import IntTuple
from layout.layout import blocked_product
from layout.layout_tensor import LayoutTensorIter
from layout.runtime_tuple import idx2crd, crd2idx
from layout.swizzle import Swizzle, make_ldmatrix_swizzle, make_swizzle
from layout.tensor_core_async import (
    st_matrix_n_layout,
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_to_descriptor,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMATensorTile,
    create_tma_tile,
)

from utils.fast_div import FastDiv
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from .arch.sm100 import MmaOpSM100_SS
from .utils import elementwise_epilogue_type
from .utils_gpu import MatmulConfig
from .grouped_matmul_tile_scheduler import TileScheduler, WorkInfo

from .matmul.gpu.sm100.matmul import _blackwell_matmul_tma_umma_warp_specialized


@fieldwise_init
@register_passable("trivial")
struct WarpRole(ImplicitlyCopyable, Movable):
    var _role: Int32

    alias Mma = Self(5)
    alias MainLoad = Self(4)
    alias Epilogue = Self(3)

    @always_inline
    fn __eq__(self, other: UInt) -> Bool:
        return self._role == other

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @always_inline
    fn __ge__(self, other: UInt) -> Bool:
        return self._role >= other

    @staticmethod
    @always_inline
    fn is_main_load() -> Bool:
        return Self.MainLoad == get_warp_id()

    @staticmethod
    @always_inline
    fn is_mma() -> Bool:
        return Self.Mma == get_warp_id()

    @staticmethod
    @always_inline
    fn is_epilogue() -> Bool:
        return Self.Epilogue >= get_warp_id()


@always_inline
fn load_AB[
    a_type: DType,
    b_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    num_pipeline_stages: UInt,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
](
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    a_smem: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    b_smem: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    mma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    tma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    producer_phase: PipelineState[Int(num_pipeline_stages)],
    peer_cta_coord: Tuple[UInt, UInt, UInt],
    work_tile_coord: Tuple[UInt, UInt],
    a_multicast_mask: UInt16,
    b_multicast_mask: UInt16,
    iter_idx: UInt32,
    elect_one_cta: Bool,
    scheduler: TileScheduler,
):
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]

    alias a_expected_bytes = a_smem_layout.size() * size_of[a_type]()
    alias b_expected_bytes = b_smem_layout.size() * size_of[b_type]()
    # Leader CTAs expect SMEM from itself and their peers
    alias expected_bytes = cta_group * (a_expected_bytes + b_expected_bytes)

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()

    var stage = producer_phase.index()
    var phase = producer_phase.phase()
    mma_mbar[stage].wait(phase)

    var a_gmem_slice_coord = (
        peer_cta_coord[2] * UInt(a_tma_rows)
        + work_tile_coord[0]
        + expert_ids[Int(scheduler.current_group_idx)] * scheduler.M
    )
    var b_gmem_slice_coord = (
        peer_cta_coord[1] * UInt(b_tma_rows)
        + peer_cta_coord[0] * UInt(BN)
        + work_tile_coord[1]
    )

    var a_smem_tile = a_smem.next(stage)[]
    var b_smem_tile = b_smem.next(stage)[]

    var a_smem_slice = type_of(a_smem_tile)(
        a_smem_tile.ptr + peer_cta_coord[2] * UInt(a_tma_load_size)
    )
    var b_smem_slice = type_of(b_smem_tile)(
        b_smem_tile.ptr + peer_cta_coord[1] * UInt(b_tma_load_size)
    )

    if elect_one_sync():
        if elect_one_cta:
            tma_mbar[stage].expect_bytes(expected_bytes)

        a_tma_op.async_multicast_load[cta_group](
            a_smem_slice,
            tma_mbar[stage],
            (UInt(iter_idx) * UInt(BK), UInt(a_gmem_slice_coord)),
            a_multicast_mask,
        )

        b_tma_op.async_multicast_load[cta_group](
            b_smem_slice,
            tma_mbar[stage],
            (UInt(iter_idx) * UInt(BK), UInt(b_gmem_slice_coord)),
            b_multicast_mask,
        )


@always_inline
fn consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
](
    tmem_addr: UInt32,
    a_smem_iter: LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    b_smem_iter: LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    mma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    tma_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    consumer_phase: PipelineState[pipeline_stages],
    mma_op: MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ],
    elect_one_warp: Bool,
    iter_idx: UInt32,
):
    var stage = consumer_phase.index()
    var phase = consumer_phase.phase()

    tma_mbar[stage].wait(phase)

    var a_smem_tile = a_smem_iter.next(stage)[]
    var b_smem_tile = b_smem_iter.next(stage)[]
    if elect_one_sync():
        mma_op.mma(
            a_smem_tile,
            b_smem_tile,
            tmem_addr,
            init_c=(iter_idx == 0),  # Initialize C on first iteration
        )

        mma_op.commit(mma_mbar + stage)


@always_inline
fn stsm_helper[
    swizzle: Swizzle,
    transpose_c: Bool = False,
](
    vec: SIMD[_, _],
    dst: LayoutTensor[_, _, address_space = AddressSpace.SHARED, *_, **_],
):
    # Number of elements in one row per stsmx4 tile, a row is 32B.
    alias stsmx4_row_size = 32 // size_of[dst.dtype]()
    # Number of elements owned by each lane, each lane has 16B
    alias stsmx4_lane_size = 16 // size_of[dst.dtype]()
    # TODO: constrain the shared memory layout to be 2D row-major.
    # E.g. dst layout can be (16, 16) : (32, 1), which is tiled from
    # row-major(16, 32). The map should use tile's stride to calculate
    # the dst row offset.
    alias stride0 = dst.layout.stride[0].value()
    alias stride1 = dst.layout.stride[1].value()
    constrained[stride1 == 1, "stride1 must be 1. Got: " + String(stride1)]()
    alias shape0 = dst.layout.shape[
        1
    ].value() if not transpose_c else dst.layout.shape[0].value()
    # the layout looks like
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b
    # but transposed and coalesced by 8 elements.
    alias trans_st_matrix_layout = Layout(
        IntTuple(8, 2, 2), IntTuple(stride0, 8 * stride1, 8 * stride0)
    )
    alias stsmx4_tile_offset = (
        stride0 if transpose_c else stride1
    ) * stsmx4_row_size

    var lane = lane_id()
    alias RLayout32Bits[layout: Layout] = RuntimeLayout[
        layout, element_type = DType.uint32, linear_idx_type = DType.uint32
    ]
    var stsm_lane_offset: UInt32 = (lane & 15) * UInt(stride0) + (
        lane >> 4
    ) * 8 if not transpose_c else RLayout32Bits[trans_st_matrix_layout]()(
        Int(lane)
    )

    # Helper function to slice a range of SIMD vector.
    # LLVM extract intrinsic generates bad code on GPU.
    @always_inline
    fn slice[offset: Int, size: Int](v: SIMD) -> SIMD[v.dtype, size]:
        var tmp = SIMD[v.dtype, size]()

        @parameter
        for i in range(size):
            tmp[i] = v[i + offset]
        return tmp

    # Assume the dst tile has 16 rows and only use stsm in N dim.
    @parameter
    for i in range(shape0 // stsmx4_row_size):
        alias n_offset = i * stsmx4_tile_offset
        var offset = swizzle(stsm_lane_offset + n_offset)
        var v = slice[i * stsmx4_lane_size, stsmx4_lane_size](vec).cast[
            dst.dtype
        ]()
        st_matrix[simd_width=4, transpose=transpose_c](
            dst.ptr + offset, bitcast[DType.float32, 4](v)
        )


@always_inline
fn multi_stage_store_C[
    c_type: DType,
    c_smem_layout: Layout,
    c_layout: Layout,
    c_tensor_layout: Layout,
    c_desc_layout: Layout,
    num_accum_pipeline_stages: UInt,
    /,
    *,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    stage_stride_cols: UInt,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 1,
    num_output_warps: UInt = 4,
    max_tmem_cols: UInt = 512,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    transpose_c: Bool = False,
](
    c_iter: LayoutTensorIter[
        c_type,
        c_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_tensor_layout, MutableAnyOrigin],
    accum_pipeline_consumer_state: PipelineState[
        Int(num_accum_pipeline_stages)
    ],
    accum_full_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    accum_empty_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    tmem_addr: UInt32,
    work_tile_coord: Tuple[UInt, UInt],
    group_end_idx: UInt32,
    elect_one_warp: Bool,
    M: UInt32,
    N: UInt32,
):
    # WAIT FOR MMA TO FINISH AND STORE RESULT
    # scheduler fetch next work
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]

    alias num_m_mmas = BM // (mma_shape[0] // cta_group)
    alias num_n_mmas = BN // (mma_shape[1] // cta_group)

    constrained[num_m_mmas == 1 and num_n_mmas == 1]()

    # assume N dimension is static
    alias simd_size = simd_width_of[c_type]()

    # we break down the output tile BM x MMA_N to BM x stageN tiles
    # and output one tile per stage.
    # stage N is 32
    alias N_dim = 0 if transpose_c else 1
    alias stageN = c_smem_layout.shape[N_dim].value()
    alias stage_contiguous_size = c_smem_layout.shape[1].value()
    # so num stages is usually 256 by 32 is 8
    # MMA Size will be larger than output tile shape. E.G. MMA_MxMMA_N = (128, 256); OUT_MxOUT_N = (128, 32)

    alias num_stages = MMA_N // stageN if (
        MMA_M == 256 or cta_group == 1
    ) else MMA_N // stageN // 2
    alias data_paths = 16  # same as lanes
    alias bits = 256
    # every element in tmem is 4 bytes, so bits being 256 means 8 elements stored across N
    # repeated 4 times is 8*4 = 32, enough to move elements into the width of our 128x32 tile
    alias rep = stageN // (bits // 32)  # repetitions per stage
    # typically repeated 4 times to get the desired 32 elements

    # stageN is how many elements we want to load at once

    # stmatrix related
    alias st_matrix_swizzle = c_swizzle
    alias swizzle = make_swizzle[c_type, st_matrix_swizzle]()

    var warp_id = get_warp_id()

    # lets keep track of the of the starting row and column in GMEM
    # var c_row = work_tile_coord[0] * BM
    # var c_col = work_tile_coord[1] * MMA_N
    var c_row = work_tile_coord[0]
    var c_col = work_tile_coord[1]

    # before i start the process of transferring over num_stages * stageN= MMA_N from tensor memory to global, i should wait
    # on the accum_full_mbar barrier
    var index = accum_pipeline_consumer_state.index()
    var phase = accum_pipeline_consumer_state.phase()
    accum_full_mbar[index].wait(phase)
    # this is the column offset for all the stages of THIS load, where one load takes (num_stages iterations)
    var tmem_offset = index * stage_stride_cols + tmem_addr

    @parameter
    for stage in range(num_stages):
        # column offset, moving right by 32 columns each time, since each num_stage stores two, 16 column submatrices
        # MMA has result in 32 rows per warp's data paths.
        # upper_frag is for rows 0-15, lower is for 16-31.
        var stage_tmem_addr = tmem_offset + (stage * stageN)
        var upper_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=rep,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr)

        var lower_frag = tcgen05_ld[
            datapaths=data_paths,
            bits=bits,
            repeat=rep,
            dtype=accum_type,
            pack=False,
        ](stage_tmem_addr + (16 << 16))

        tcgen05_load_wait()

        @parameter
        if stage == num_stages - 1:
            umma_arrive_leader_cta(accum_empty_mbar + index)

        # Assume double-buffer for shared memory packing
        var c_smem_tile = c_iter.next(stage % 2)[]

        @parameter
        if transpose_c:
            # if stage_contiguous_size is 128, we need to split the shared memory
            # into two stageNx64 row-major tiles due to the limitation of 128B TMA
            # swizzle. However, for easier programming, we reshape the tile
            # contiguous row_major(stageN, 16) chunks.
            var c_smem_warp_tile_upper = c_smem_tile.tile[
                stageN * 16 // stage_contiguous_size, stage_contiguous_size
            ](2 * Int(warp_id), 0).reshape[Layout.row_major(stageN, 16)]()
            var c_smem_warp_tile_lower = c_smem_tile.tile[
                stageN * 16 // stage_contiguous_size, stage_contiguous_size
            ](2 * Int(warp_id) + 1, 0).reshape[Layout.row_major(stageN, 16)]()

            # Pack the upper frag to shared memory
            stsm_helper[swizzle, transpose_c](
                upper_frag, c_smem_warp_tile_upper
            )
            stsm_helper[swizzle, transpose_c](
                lower_frag, c_smem_warp_tile_lower
            )

            # Guard the write to shared memory is done.
            named_barrier[num_output_warps * UInt(WARP_SIZE)]()

        else:
            var c_smem_warp_tile = c_smem_tile.tile[32, stageN](Int(warp_id), 0)

            var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                data_paths, stageN
            ](0, 0)
            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                data_paths, stageN
            ](1, 0)
            stsm_helper[swizzle, transpose_c](
                upper_frag, c_smem_warp_tile_upper
            )
            stsm_helper[swizzle, transpose_c](
                lower_frag, c_smem_warp_tile_lower
            )

            # Guard the write to shared memory is done.
            named_barrier[num_output_warps * UInt(WARP_SIZE)]()

        var lane = lane_id()

        alias TMA_BM = c_smem_tile.layout.shape[
            0
        ].value() if MMA_M == 256 or cta_group == 1 else BM

        var elect_one_warp = (
            warp_id == 0 if MMA_M == 256 or cta_group == 1 else warp_id % 2 == 0
        )
        # var coord_n_mma_m256 = work_tile_coord[1] * MMA_N + stage * stageN
        # var coord_n_mma_m128 = (
        #     work_tile_coord[1] * MMA_N + stage * stageN + BN * (warp_id // 2)
        # )
        var coord_n_mma_m256 = work_tile_coord[1] + UInt(stage * stageN)
        var coord_n_mma_m128 = (
            work_tile_coord[1]
            + UInt(stage * stageN)
            + UInt(BN * Int(warp_id // 2))
        )

        var coord_n = (
            coord_n_mma_m256 if MMA_M == 256
            or cta_group == 1 else coord_n_mma_m128
        )

        var n_inbound_size = group_end_idx - coord_n

        alias M = c_smem_tile.layout.shape[1].value()

        alias has_elementwise_lambda = Bool(elementwise_lambda_fn)

        if not has_elementwise_lambda and n_inbound_size >= stageN:
            if elect_one_warp and lane == 0:
                fence_async_view_proxy()

                @parameter
                if transpose_c:

                    @parameter
                    for i in range(M // 16):
                        var c_smem_warp_tile = c_smem_tile.tile[
                            stageN * 16 // stage_contiguous_size,
                            stage_contiguous_size,
                        ](i, 0).reshape[Layout.row_major(stageN, 16)]()
                        c_tma_op.async_store(
                            c_smem_warp_tile,
                            (
                                UInt(work_tile_coord[0] + UInt(i * 16)),
                                UInt(coord_n),
                            ),
                        )
                else:
                    var c_smem_coord_m = 0 if MMA_M == 256 else (warp_id // 2)
                    var c_smem_split = c_smem_tile.tile[TMA_BM, stageN](
                        Int(c_smem_coord_m), 0
                    )
                    c_tma_op.async_store(
                        c_smem_split,
                        (
                            UInt(coord_n),
                            # UInt(work_tile_coord[0] * BM),
                            UInt(work_tile_coord[0]),
                        ),
                    )
                c_tma_op.commit_group()

            # Keep one tma store in fly
            @parameter
            if stage < num_stages - 1:
                c_tma_op.wait_group[1]()
            # Last stage guard all tma store to finish
            else:
                c_tma_op.wait_group[0]()
        else:
            constrained[
                transpose_c, "Unaligned handling only supports transpose_c"
            ]()
            constrained[
                MMA_M == 256 or cta_group == 1,
                "Unaligned handling only supports MMA_M == 256 or cta_group =="
                " 1. Got "
                + String(MMA_M)
                + " and "
                + String(cta_group),
            ]()

            alias chunkM = st_matrix_swizzle.bytes() // size_of[c_type]()
            alias vec_chunkM = chunkM // simd_size
            alias chunk_num = M // chunkM
            alias logical_c_layout = Layout.row_major(
                chunk_num, stageN, vec_chunkM
            )
            alias thread_num = num_output_warps * UInt(WARP_SIZE)
            constrained[
                logical_c_layout.size() % Int(thread_num) == 0,
                "logical_c_layout.size() must be a multiple of thread_num. Got "
                + String(logical_c_layout.size())
                + ".",
            ]()
            alias value_shape = logical_c_layout.size() // Int(thread_num)
            alias cM = c.shape[1]()

            @parameter
            for v in range(value_shape):
                alias thread_offset = v * Int(thread_num)
                var thread_index = UInt32(thread_idx.x) + UInt32(thread_offset)
                # idx2crd but RuntimeTuple.idx2crd is too hard to use
                var vec_chunkM_idx = thread_index % vec_chunkM
                var rest = thread_index // vec_chunkM
                var n_idx = rest % stageN
                if n_idx >= min(n_inbound_size, stageN):
                    continue
                var src_idx = simd_size * thread_index
                var c_smem_idx = swizzle(src_idx)
                alias alignment = align_of[SIMD[c_type, simd_size]]()
                var val_vec = (c_smem_tile.ptr + c_smem_idx).load[
                    width=simd_size, alignment=alignment
                ]()
                var chunk_idx = rest // stageN
                var n = coord_n + n_idx
                var m = (
                    work_tile_coord[0]
                    + (chunk_idx * vec_chunkM + vec_chunkM_idx) * simd_size
                )
                if m < cM:

                    @parameter
                    if elementwise_lambda_fn:
                        alias elementwise_lambda = elementwise_lambda_fn.value()
                        elementwise_lambda[
                            c_type, simd_size, alignment=alignment
                        ](Index(n, m), val_vec)
                    else:
                        (c.ptr + n * cM + m).store[alignment=alignment](val_vec)

        @parameter
        if stage > 0 or stage == num_stages - 1:
            # Guard the tma read from shared memory is done.
            named_barrier[num_output_warps * UInt(WARP_SIZE)]()


fn zero_output[
    c_type: DType,
    c_layout: Layout,
    *,
    output_tile_shape: IndexList[2],
](
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    coord: Tuple[UInt32, UInt32],
    group_end_idx: UInt32,
):
    alias thread_num = 4 * WARP_SIZE
    alias simd_size = min(2, simd_width_of[c_type]())
    # unfortunately this doesn't just work
    # c_frag = c.vectorize[1, simd_size]().distribute[Layout.row_major(1, thread_num)](thread_idx.x)
    # _ = c_frag.fill(0.0)

    # This is an easy to implement but quite inefficient way to zero out the
    # output. Note that we are simply filling the output row by row and mask out
    # out of bound thread and the thread layout is simply one dimensional.

    # Note that output_tile_shape is always the proper C tile shape independent of transpose_c.
    alias output_N = output_tile_shape[1]
    alias stride = c.layout.stride[0].value()
    var ptr = c.ptr + coord[1] * stride + coord[0]
    constrained[
        thread_num * simd_size >= output_N,
        "output_N must be less than thread_num * simd_size. Got "
        + String(output_N)
        + ".",
    ]()
    alias row_thread_num = UInt32(output_N // simd_size)
    alias cN = c.shape[1]()
    var row_boundary = UInt32(cN - coord[0]) // simd_size
    alias alignment = align_of[SIMD[c_type, simd_size]]()
    var zero_vec = SIMD[c_type, simd_size](0.0)
    var M = group_end_idx - coord[1]
    if UInt32(thread_idx.x) < min(row_thread_num, row_boundary):
        for i in range(min(M, output_tile_shape[0])):
            (ptr + thread_idx.x * UInt(simd_size)).store[alignment=alignment](
                zero_vec
            )
            ptr += stride


# Important deviation from the normal SM100 matmul: The coordinate returned by
# the tile scheduler is not necessarily aligned to `MMA_N` because of group
# splitting. Thus, we simply take the `work_tile_coord` without scaling it
# in the callee; the scheduler returns element coordinates instead of tile
# coordinates.
@__llvm_metadata(`nvvm.cluster_dim`=cluster_shape)
@__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
fn blackwell_tma_umma_warp_specialized_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    expert_m: Int,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    c_tensor_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    num_pipeline_stages: UInt,
    num_accum_pipeline_stages: UInt,
    num_output_stages: UInt = 2,
    output_tile_shape: IndexList[2] = Index(128, 32),
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    cta_group: Int = 2,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    transpose_c: Bool = False,
](
    num_active_experts: Int,
    a_tma_op: TMATensorTile[a_type, a_layout, a_desc_layout],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    b_tma_op: TMATensorTile[b_type, b_layout, b_desc_layout],
    b_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    c_tma_op: TMATensorTile[c_type, c_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_tensor_layout, MutableAnyOrigin],
    mnk: StaticTuple[UInt32, 3],
):
    constrained[c_type is not DType.float32, "c_type cannot be float32"]()

    alias num_output_warps = 4

    alias SCHEDULER_THREADS = WARP_SIZE
    alias TMA_LOAD_THREADS = WARP_SIZE
    alias MMA_THREADS = WARP_SIZE
    alias EPILOGUE_THREADS = num_output_warps * WARP_SIZE
    alias CLUSTER_SIZE = cluster_shape[0] * cluster_shape[1]
    alias clc_producer_arv_count = 1
    alias clc_consumer_arv_count = SCHEDULER_THREADS + CLUSTER_SIZE * (
        TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
    )

    # For ld from TMEM, use same per-stage stride in column field.
    alias NUM_TMEM_COLS = 512
    alias stage_stride_cols = NUM_TMEM_COLS // Int(num_accum_pipeline_stages)

    alias clc_throttle_producer_arv_count = TMA_LOAD_THREADS
    alias clc_throttle_consumer_arv_count = SCHEDULER_THREADS

    alias accum_pipeline_producer_arv_count = 1
    alias accum_pipeline_consumer_arv_count = cta_group * EPILOGUE_THREADS

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]
    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias MMA_K = mma_shape[2]

    alias num_m_mmas = BM // (mma_shape[0] // cta_group)
    alias num_n_mmas = BN // (mma_shape[1] // cta_group)
    alias num_k_mmas = BK // mma_shape[2]

    alias CLUSTER_M = Int(cluster_shape[0])
    alias CLUSTER_N = Int(cluster_shape[1])

    alias a_tma_load_size = a_desc_layout.size()
    alias b_tma_load_size = b_desc_layout.size()
    alias a_tma_rows = a_desc_layout.shape[0].value()
    alias b_tma_rows = b_desc_layout.shape[0].value()

    # keep the physical SMEM buffer BM x MMA_N
    alias a_smem_layout = tile_layout_k_major[
        a_type, BM, BK, swizzle_mode=a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]() if transpose_b else tile_layout_mn_major[
        b_type, BN, BK, swizzle_mode=b_swizzle
    ]()

    base_ptr_smem = external_memory[
        Scalar[a_type],
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]()

    alias a_smem_size = a_smem_layout.size() * Int(num_pipeline_stages)
    alias b_smem_size = b_smem_layout.size() * Int(num_pipeline_stages)
    alias c_smem_size = output_tile_shape[0] * output_tile_shape[1] * Int(
        num_output_stages
    )

    var a_smem_base = base_ptr_smem
    var b_smem_base = (a_smem_base + a_smem_size).bitcast[Scalar[b_type]]()
    var c_smem_base = (b_smem_base + b_smem_size).bitcast[Scalar[c_type]]()

    var a_smem = LayoutTensorIter[
        a_type,
        a_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        a_smem_base,
        a_smem_size,
    )

    var b_smem = LayoutTensorIter[
        b_type,
        b_smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        b_smem_base,
        b_smem_size,
    )

    var c_smem_iter = LayoutTensorIter[
        c_type,
        Layout.row_major(output_tile_shape[0], output_tile_shape[1]),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](c_smem_base, c_smem_size)

    var smem_pool = (c_smem_base + c_smem_size).bitcast[Int64]()

    var tma_mbar_ptr = smem_pool
    var mma_mbar_ptr = tma_mbar_ptr + num_pipeline_stages
    var accum_full_mbar_ptr = mma_mbar_ptr + num_pipeline_stages
    var accum_empty_mbar_ptr = accum_full_mbar_ptr + num_accum_pipeline_stages

    var tmem_dealloc_mbar_ptr = (
        accum_empty_mbar_ptr + num_accum_pipeline_stages
    ).bitcast[Int64]()

    var ptr_tmem_addr = (tmem_dealloc_mbar_ptr + 1).bitcast[UInt32]()

    tma_mbar = tma_mbar_ptr.bitcast[SharedMemBarrier]()
    mma_mbar = mma_mbar_ptr.bitcast[SharedMemBarrier]()
    accum_full_mbar = accum_full_mbar_ptr.bitcast[SharedMemBarrier]()
    accum_empty_mbar = accum_empty_mbar_ptr.bitcast[SharedMemBarrier]()
    tmem_dealloc_mbar = tmem_dealloc_mbar_ptr.bitcast[SharedMemBarrier]()

    alias accum_type = get_accum_type[a_type]()

    var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
    var elect_one_thread = elect_one_sync_with_mask()
    var elect_one_cta = block_rank_in_cluster() % 2 == 0
    var is_first_cta_in_cluster = block_rank_in_cluster() == 0
    var warp_id = get_warp_id()
    alias max_tmem_cols = 512

    if elect_one_warp and elect_one_thread:
        a_tma_op.prefetch_descriptor()
        b_tma_op.prefetch_descriptor()
        c_tma_op.prefetch_descriptor()

        @parameter
        for i in range(num_pipeline_stages):
            tma_mbar[i].init()
            # we need to have 5 arrivals, 2 M, 4 N, top left M/N is shared
            mma_mbar[i].init(
                cluster_shape[0] // cta_group + cluster_shape[1] - 1
            )

        @parameter
        for i in range(num_accum_pipeline_stages):
            accum_full_mbar[i].init(accum_pipeline_producer_arv_count)
            accum_empty_mbar[i].init(accum_pipeline_consumer_arv_count)

        tmem_dealloc_mbar[].init(EPILOGUE_THREADS * cta_group)

    fence_mbarrier_init()
    cluster_sync()

    var consumer_phase = PipelineState[Int(num_pipeline_stages)]()
    var producer_phase = PipelineState[Int(num_pipeline_stages)](0, 1, 0)

    var accum_pipeline_producer_state = PipelineState[
        Int(num_accum_pipeline_stages)
    ](0, 1, 0)
    var accum_pipeline_consumer_state = PipelineState[
        Int(num_accum_pipeline_stages)
    ]()

    var mma_op = MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape = Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ]()

    var scheduler = TileScheduler[
        M=expert_m,
        cluster = Index(cluster_shape[0], cluster_shape[1], cluster_shape[2]),
        cta_group=cta_group,
        tile_shape = Index(
            block_tile_shape[0], block_tile_shape[1], block_tile_shape[2]
        ),
    ](num_active_experts, b_offsets)

    var work_info = scheduler.fetch_next_work()

    var rank_m = block_id_in_cluster.x
    var rank_n = block_id_in_cluster.y

    # (peer_id, mma_coord_m, mma_coord_n)
    var peer_cta_coord = (
        UInt(rank_m % UInt(cta_group)),
        UInt(rank_m // UInt(cta_group)),
        rank_n,
    )  # v,m,n

    var a_multicast_mask: UInt16 = 0x0
    var b_multicast_mask: UInt16 = 0x0

    # TODO: find a generic way to calculate multicast mask
    @parameter
    for i in range(CLUSTER_N):
        a_multicast_mask |= 1 << (i * CLUSTER_M)
    # they all have the same v and m, but different n,

    @parameter
    for i in range(CLUSTER_M // cta_group):
        b_multicast_mask |= 1 << (i * cta_group)

    a_multicast_mask <<= rank_m
    b_multicast_mask <<= peer_cta_coord[0]
    b_multicast_mask <<= rank_n * UInt(CLUSTER_M)

    var self_mask = 1 << Int(block_rank_in_cluster())
    var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
    var mma_complete_mask = self_mask | peer_mask

    var num_iters: UInt32 = ceildiv(mnk[2], BK)

    if WarpRole.is_main_load():
        while not work_info.is_done():
            if (
                not work_info.is_valid()
                or expert_ids[Int(scheduler.current_group_idx)] < 0
            ):
                work_info = scheduler.fetch_next_work()
                continue

            # DO TMA LOAD
            for i in range(num_iters):
                load_AB[
                    block_tile_shape=block_tile_shape,
                    mma_shape=mma_shape,
                    cta_group=cta_group,
                ](
                    expert_ids,
                    a_tma_op,
                    b_tma_op,
                    a_smem,
                    b_smem,
                    mma_mbar,
                    tma_mbar,
                    producer_phase,
                    peer_cta_coord,
                    (UInt(work_info.m), UInt(work_info.n)),
                    a_multicast_mask,
                    b_multicast_mask,
                    i,
                    elect_one_cta,
                    scheduler,
                )
                producer_phase.step()

            syncwarp()
            var next_work_info = scheduler.fetch_next_work()
            work_info = next_work_info

        @parameter
        for i in range(num_pipeline_stages):
            mma_mbar[producer_phase.index()].wait(producer_phase.phase())
            producer_phase.step()

    if WarpRole.is_mma():
        tcgen05_alloc[cta_group](ptr_tmem_addr, max_tmem_cols)
        syncwarp()
        # non blocking, arrives and proceeds
        named_barrier_arrive[MMA_THREADS + EPILOGUE_THREADS](1)

        tmem_addr = ptr_tmem_addr[0]

        while not work_info.is_done():
            if (
                not work_info.is_valid()
                or expert_ids[Int(scheduler.current_group_idx)] < 0
            ):
                work_info = scheduler.fetch_next_work()
                continue
            # scheduler fetch next work
            next_work_info = scheduler.fetch_next_work()
            # DO MMA
            if elect_one_cta:
                var accum_index = accum_pipeline_producer_state.index()
                var accum_phase = accum_pipeline_producer_state.phase()

                accum_empty_mbar[accum_index].wait(accum_phase)
                var tmem_offset = tmem_addr + (accum_index * stage_stride_cols)

                for i in range(num_iters):
                    consumer_main_loop[
                        block_tile_shape=block_tile_shape,
                        mma_shape=mma_shape,
                        cta_group=cta_group,
                        cluster_shape = Index(
                            cluster_shape[0], cluster_shape[1], cluster_shape[2]
                        ),
                    ](
                        tmem_offset,
                        a_smem,
                        b_smem,
                        mma_mbar,
                        tma_mbar,
                        consumer_phase,
                        mma_op,
                        elect_one_warp,
                        i,
                    )
                    consumer_phase.step()

                # mma arrive multicast will track completion of all mma prior to this barrier.
                if elect_one_sync():

                    @parameter
                    if cta_group == 2:
                        mma_arrive_multicast[cta_group](
                            accum_full_mbar + accum_index,
                            mma_complete_mask,
                        )
                    else:
                        mma_arrive(accum_full_mbar + accum_index)

                accum_pipeline_producer_state.step()
            work_info = next_work_info

        tcgen05_release_allocation_lock[cta_group]()

        # wait for epilogue to finish
        tmem_dealloc_mbar[].wait()

        tcgen05_dealloc[cta_group](tmem_addr, max_tmem_cols)

    if WarpRole.is_epilogue():
        named_barrier[MMA_THREADS + EPILOGUE_THREADS](1)
        tmem_addr = ptr_tmem_addr[0]

        # while work_info.is_valid():
        while not work_info.is_done():
            if not work_info.is_valid():
                work_info = scheduler.fetch_next_work()
                continue

            if expert_ids[Int(scheduler.current_group_idx)] < 0:
                zero_output[output_tile_shape=output_tile_shape](
                    c,
                    (UInt32(work_info.m), UInt32(work_info.n)),
                    scheduler.group_offsets[
                        Int(scheduler.current_group_idx + 1)
                    ],
                )
                work_info = scheduler.fetch_next_work()
                continue
            # WAIT FOR MMA TO FINISH AND STORE RESULT
            # scheduler fetch next work
            multi_stage_store_C[
                accum_type=accum_type,
                block_tile_shape=block_tile_shape,
                mma_shape=mma_shape,
                stage_stride_cols = UInt(stage_stride_cols),
                c_swizzle=c_swizzle,
                cta_group=cta_group,
                num_output_warps=num_output_warps,
                max_tmem_cols=max_tmem_cols,
                elementwise_lambda_fn=elementwise_lambda_fn,
                transpose_c=transpose_c,
            ](
                c_smem_iter,
                c_tma_op,
                c,
                accum_pipeline_consumer_state,
                accum_full_mbar,
                accum_empty_mbar,
                tmem_addr,
                work_tile_coord=(UInt(work_info.m), UInt(work_info.n)),
                group_end_idx=scheduler.group_offsets[
                    Int(scheduler.current_group_idx + 1)
                ],
                elect_one_warp=elect_one_warp,
                M=mnk[0],
                N=mnk[1],
            )
            accum_pipeline_consumer_state.step()

            next_work_info = scheduler.fetch_next_work()
            work_info = next_work_info

        @parameter
        if cta_group == 2:
            _ = tmem_dealloc_mbar[].arrive_cluster(block_rank_in_cluster() ^ 1)

        _ = tmem_dealloc_mbar[].arrive()


fn grouped_matmul_sm100_persistent[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    cta_group: Int = 1,
    num_pipeline_stages: Optional[UInt] = None,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: NDBuffer[c_type, 2, MutableAnyOrigin, c_shape],
    a: NDBuffer[a_type, 2, MutableAnyOrigin, a_shape],
    a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    max_num_tokens_per_expert: Int,
    b: NDBuffer[b_type, 3, MutableAnyOrigin, b_shape],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    # swapAB by default
    alias num_experts = b.shape.get[0]()
    alias M = b.shape.get[1]()
    alias K = b.shape.get[2]()

    a_tensor = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * M, K),
        MutableAnyOrigin,
        address_space = AddressSpace.GENERIC,
    ](b.data)

    b_tensor = from_ndbuffer_row_major(a)
    c_tensor = from_ndbuffer_row_major(c)

    b_offsets = a_offsets
    alias new_config = config.swapAB()

    _grouped_matmul_sm100_persistent[
        c_type=c_type,
        c_layout = type_of(c_tensor).layout,
        a_type=b_type,
        a_layout = type_of(a_tensor).layout,
        b_type=a_type,
        b_layout = type_of(b_tensor).layout,
        transpose_b=transpose_b,
        config=new_config,
        expert_m=M,
        cta_group=cta_group,
        num_pipeline_stages=num_pipeline_stages,
        transpose_c=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ](
        c_tensor,
        a_tensor,
        expert_ids,
        b_tensor,
        b_offsets,
        num_active_experts,
        ctx,
    )


fn _grouped_matmul_sm100_persistent[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    expert_m: Int,
    cta_group: Int = 1,
    num_pipeline_stages: Optional[UInt] = None,
    transpose_c: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c_device: LayoutTensor[c_type, c_layout, *_, **_],
    a_device: LayoutTensor[a_type, a_layout, *_, **_],
    expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
    b_device: LayoutTensor[b_type, b_layout, *_, **_],
    b_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    alias MMA_M = config.mma_shape[0]
    alias MMA_N = config.mma_shape[1]
    alias MMA_K = config.mma_shape[2]

    alias BM = MMA_M // cta_group
    alias BN = MMA_N // cta_group
    alias BK = config.block_tile_shape[2]

    constrained[
        (MMA_M != 128) or (MMA_N % 32 == 0),
        "if MMA_M is 128, then MMA_N must be a multiple of 32",
    ]()

    alias cluster_shape = config.cluster_shape

    var M = c_device.dim[0]()
    var N = c_device.dim[1]()
    var M_maybe_swapped = a_device.dim[0]()
    var N_maybe_swapped = b_device.dim[0]()
    var K = a_device.dim[1]()

    a_tma_op = create_tma_tile[
        Index(BM // cluster_shape[1], BK), swizzle_mode=a_swizzle
    ](ctx, a_device)

    b_tma_op = create_tma_tile[
        Index(
            BN // (cluster_shape[0] // cta_group), BK
        ) if transpose_b else Index(BK, BN // (cluster_shape[0] // cta_group)),
        is_k_major=transpose_b,
        swizzle_mode=b_swizzle,
    ](ctx, b_device)

    # If MMA_M is 256, the warps read the entire MMA_N.
    # That MMA_N to be multiple of 32 for me to use large N dim on C buf write out
    # If MMA_M is 128, the warps read 1/2 of MMA_N (BN), so now *that* has to be multiple of 32
    # Otherwise, we just use 16
    alias width = 32 if (MMA_M == 256 and MMA_N % 32 == 0) or (
        MMA_M == 128 and BN % 32 == 0
    ) else 16
    alias output_tile_shape = Index(128, width) if not transpose_c else Index(
        width, 128
    )
    alias split_tile_shape = Index(64, width) if not transpose_c else Index(
        width, 64
    )
    alias c_tma_tile_shape = output_tile_shape if MMA_M == 256 else split_tile_shape
    alias c_swizzle = TensorMapSwizzle.SWIZZLE_32B if transpose_c else (
        TensorMapSwizzle.SWIZZLE_64B if width
        == 32 else TensorMapSwizzle.SWIZZLE_32B
    )
    # transpose_c => MMA_M == 256 is the same as (not transpose_c) or MMA_M == 256
    constrained[
        (not (transpose_c and cta_group == 2)) or MMA_M == 256,
        "swapAB is only supported for MMA_M == 256",
    ]()
    var c_tma_op = create_tma_tile[
        c_tma_tile_shape if not transpose_c else Index(
            c_tma_tile_shape[0], c_swizzle.bytes() // size_of[c_type]()
        ),
        swizzle_mode=c_swizzle,
    ](ctx, c_device)

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    alias b200_smem = B200.shared_memory_per_multiprocessor - 1024
    alias a_smem_bytes_per_stage = BM * BK * size_of[a_type]()
    alias b_smem_bytes_per_stage = BN * BK * size_of[b_type]()
    # A and B per pipeline stage
    alias AB_smem_per_stage = a_smem_bytes_per_stage + b_smem_bytes_per_stage
    # Support double-buffer for output stages.
    alias num_output_stages = 2

    alias c_smem_bytes = output_tile_shape[0] * output_tile_shape[
        1
    ] * num_output_stages * size_of[c_type]()

    alias MBAR_BYTES = size_of[Int64]()  # 8 bytes per barrier
    alias CLC_RESPONSE_BYTES = size_of[Int128]()  # 16 bytes per response
    alias TMEM_ADDR_BYTES = size_of[
        Int32
    ]()  # 4 bytes or 32 bits for tensor memory address
    # the 'N' dimension of tensor memory is 512
    alias TMEM_N = 512
    # the maximum different number of mma's that can be run in parallel is TMEM_N/MMA_N
    alias max_accum_pipeline_stages = TMEM_N // next_power_of_two(MMA_N)
    # Mainloop barrier
    alias accum_full_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages
    alias accum_empty_mbar_bytes = MBAR_BYTES * max_accum_pipeline_stages

    alias tmem_addr_bytes = TMEM_ADDR_BYTES
    alias tmem_dealloc_mbar_bytes = MBAR_BYTES

    alias tmem_writeout_smem = c_smem_bytes + tmem_addr_bytes + tmem_dealloc_mbar_bytes
    alias accum_smem = accum_full_mbar_bytes + accum_empty_mbar_bytes
    alias smem_leftover = (b200_smem) - (accum_smem + tmem_writeout_smem)

    alias tma_mbar_bytes_per_stage = MBAR_BYTES
    alias mma_mbar_bytes_per_stage = MBAR_BYTES

    alias producer_consumer_smem_per_stage = (
        AB_smem_per_stage + tma_mbar_bytes_per_stage + mma_mbar_bytes_per_stage
    )

    alias max_pipeline_stages = UInt(
        smem_leftover // producer_consumer_smem_per_stage
    )

    constrained[
        max_pipeline_stages >= 1, "Max pipeline stages must be at least 1"
    ]()

    @parameter
    if num_pipeline_stages:
        constrained[
            num_pipeline_stages.value() <= max_pipeline_stages,
            "Pipeline stage must be less than or equal to max pipeline stages",
        ]()

    alias pipeline_stage = num_pipeline_stages.value() if num_pipeline_stages else max_pipeline_stages
    alias producer_consumer_smem = producer_consumer_smem_per_stage * Int(
        pipeline_stage
    )

    alias smem_size = (
        # clc_smem + accum_smem + producer_consumer_smem + tmem_writeout_smem
        accum_smem
        + producer_consumer_smem
        + tmem_writeout_smem
    )

    alias kernel = blackwell_tma_umma_warp_specialized_kernel[
        a_type,
        b_type,
        c_type,
        expert_m,
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op.layout,
        c_device.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
        config.block_tile_shape,
        config.mma_shape,
        transpose_b=transpose_b,
        cluster_shape = StaticTuple[Int32, 3](
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        cta_group=cta_group,
        num_pipeline_stages = UInt(pipeline_stage),
        num_accum_pipeline_stages = UInt(max_accum_pipeline_stages),
        num_output_stages = UInt(num_output_stages),
        output_tile_shape=output_tile_shape,
        transpose_c=transpose_c,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    constrained[
        cluster_shape[1] == 1,
        "cluster_shape[1] must be 1. Got " + String(cluster_shape[1]),
    ]()

    var grid_dim = (
        Int(B200.sm_count),
        1,
        1,
    )

    var mnk = StaticTuple[UInt32, 3](M, N, K)

    ctx.enqueue_function_checked[kernel, kernel](
        num_active_experts,
        a_tma_op,
        expert_ids,
        b_tma_op,
        b_offsets,
        c_tma_op,
        c_device,
        mnk,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 4 EPILOGUE warps
        block_dim=(32 * 6),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
    )
