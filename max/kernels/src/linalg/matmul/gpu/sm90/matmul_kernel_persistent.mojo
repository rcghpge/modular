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

from math import ceildiv

from gpu import MAX_THREADS_PER_BLOCK_METADATA
from gpu.globals import WARPGROUP_SIZE
from gpu.id import thread_idx
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from layout import Layout, LayoutTensor
from layout.tma_async import TMATensorTile

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.matmul.gpu.tile_scheduler import MatmulSchedule, TileScheduler
from .matmul_kernels import find_K_alignment_upto_16B, HopperMatmulSM90Kernel


__extension HopperMatmulSM90Kernel:
    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_persistent[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        c_tma_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        c_desc_layout: Layout,
        grid_shape: IndexList[2],
        schedule: MatmulSchedule,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
        problem_shape: IndexList[3],
    ):
        alias K = b_layout.shape[1].value()
        alias num_k_iters = ceildiv(K, Self.BK)

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        var smem = Self.SMem()

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        # Create ring buffer
        var ring_buffer = Self.build_ring_buffer(smem, warp_group_thread_idx)

        # Create TileLoaderTMA loaders
        var a_loader, b_loader = Self.build_tma_loaders(
            a_tma_op, b_tma_op, rank_m, rank_n
        )

        Self.pipeline_init()

        alias N = b_layout.shape[0].value()
        alias M = a_layout.shape[0].value()
        var scheduler = TileScheduler[
            Index(M, N, K), block_tile_shape, grid_shape, schedule=schedule
        ](problem_shape)
        var work_info = scheduler.get_current_work_info()

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            _ = Self.setup_producer()

            if warp_id == 0 and lane_predicate:
                while work_info.is_valid():
                    var m_coord = work_info.m
                    var n_coord = work_info.n

                    Self.producer_main_loop[num_k_iters=num_k_iters](
                        UInt(m_coord),
                        UInt(n_coord),
                        0,
                        a_loader,
                        b_loader,
                        ring_buffer,
                    )
                    work_info = scheduler.fetch_next_work()
        else:
            # Consumer warp groups
            var local_warp_group_idx, c_reg_tile, final_c_reg_tile = (
                Self.setup_consumer(warp_group_idx)
            )

            @parameter
            if a_type is DType.float8_e4m3fn:
                _ = final_c_reg_tile.fill(0.0)
            else:
                _ = c_reg_tile.fill(0.0)

            # Enter consumer mode
            with ring_buffer.consumer() as consumer:
                while work_info.is_valid():
                    Self.consumer_main_loop[num_k_iters=num_k_iters](
                        wgmma_op,
                        local_warp_group_idx,
                        final_c_reg_tile,
                        c_reg_tile,
                        consumer,
                    )

                    var block_y = UInt(ceildiv(work_info.m, Self.BM))
                    var block_x = UInt(ceildiv(work_info.n, Self.BN))
                    var output_reg_tile = (
                        final_c_reg_tile if a_type
                        is DType.float8_e4m3fn else c_reg_tile
                    )

                    Self.consumer_output(
                        c_tma_op,
                        c,
                        smem.c_tile,
                        output_reg_tile,
                        UInt(warp_group_thread_idx),
                        UInt(local_warp_group_idx),
                        thread_idx.x - UInt(WARPGROUP_SIZE),
                        Int(block_y),
                        Int(block_x),
                    )
                    work_info = scheduler.fetch_next_work()

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_unaligned[
        c_desc_layout: Layout,
        c_tma_layout: Layout,
    ](
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
        b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    ):
        """Kernel using cp.async for A/B loading when K alignment doesn't meet TMA requirements.
        """
        alias K = b_layout.shape[1].value()
        alias num_k_iters = ceildiv(K, Self.BK)

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        var smem = Self.SMem()

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        # Create RingBuffer for cp.async operations
        var ring_buffer = Self.build_ring_buffer[tma_transfer=False](
            smem, warp_group_thread_idx
        )

        # Create TileLoaderCPAsync loaders
        alias k_align = find_K_alignment_upto_16B(K * size_of[a_type]())
        var a_loader, b_loader = Self.build_cpasync_loaders[k_align](a, b)

        Self.pipeline_init()

        # Calculate block swizzle
        var block_idx_swizzle = Self.get_block_swizzle()

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            warpgroup_reg_dealloc[32]()

            Self.producer_main_loop[num_k_iters=num_k_iters](
                UInt(block_idx_swizzle[1]),
                UInt(block_idx_swizzle[0]),
                0,
                a_loader,
                b_loader,
                ring_buffer,
            )
        else:
            # Consumer warp groups
            constrained[
                Self.num_consumer <= 2, "Only support 1 or 2 consumer"
            ]()
            warpgroup_reg_alloc[232]()

            var local_warp_group_idx = warp_group_idx - 1
            var c_reg_tile = Self.AccumRegTileType.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

            # Enter consumer mode
            with ring_buffer.consumer() as consumer:
                Self.consumer_main_loop[num_k_iters=num_k_iters](
                    wgmma_op,
                    local_warp_group_idx,
                    final_c_reg_tile,
                    c_reg_tile,
                    consumer,
                )

            var output_reg_tile = (
                final_c_reg_tile if a_type
                is DType.float8_e4m3fn else c_reg_tile
            )

            Self.consumer_output(
                c_tma_op,
                c,
                smem.c_tile,
                output_reg_tile,
                UInt(warp_group_thread_idx),
                UInt(local_warp_group_idx),
                thread_idx.x - UInt(WARPGROUP_SIZE),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()
