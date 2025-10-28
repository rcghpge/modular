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

from sys import size_of

from gpu import barrier
from gpu.cluster import block_rank_in_cluster, cluster_sync
from gpu.host import DeviceContext, Dim
from gpu import block_idx, thread_idx
from gpu.memory import fence_mbarrier_init
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout.layout_tensor import copy_sram_to_dram
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tma_tile
from memory import stack_allocation
from testing import assert_equal


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_mcast_load_kernel[
    dtype: DType,
    layout: Layout,
    tile_layout: Layout,
    thread_layout: Layout,
    CLUSTER_M: UInt32,
    CLUSTER_N: UInt32,
](
    dst: LayoutTensor[dtype, layout, MutableAnyOrigin],
    tma_tile: TMATensorTile[dtype, tile_layout],
):
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias expected_bytes = tile_layout.size() * size_of[dtype]()

    var block_rank = block_rank_in_cluster()
    alias CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

    var rank_m = block_rank // CLUSTER_N
    var rank_n = block_rank % CLUSTER_N

    var tma_multicast_mask = (1 << CLUSTER_N) - 1

    tile = (
        LayoutTensor[
            dtype,
            tile_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]
        .stack_allocation()
        .fill(10)
    )

    barrier()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar[0].init()

    barrier()

    # we use cluster_sync() together with a mbarrier init fence to ensure cluster-wide visibility of the mbarrier initialization
    cluster_sync()
    fence_mbarrier_init()

    if thread_idx.x == 0:
        mbar[0].expect_bytes(expected_bytes)
        if rank_n == 0:
            var multicast_mask = tma_multicast_mask << (rank_m * CLUSTER_N)
            tma_tile.async_multicast_load(
                tile,
                mbar[0],
                (
                    UInt(block_idx.x * UInt(tileN)),
                    UInt(block_idx.y * UInt(tileM)),
                ),
                multicast_mask.cast[DType.uint16](),
            )

    barrier()

    # after this line, the TMA load is finished
    mbar[0].wait()

    # we use another cluster_sync() to ensure that one of the two CTAs in the cluster doesn’t exit prematurely while the other is still waiting for the multicast load to complete.
    cluster_sync()

    dst_tile = dst.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_multicast_load_row_major[
    src_layout: Layout,
    tile_layout: Layout,
    dst_layout: Layout,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
](ctx: DeviceContext):
    alias src_M = src_layout.shape[0].value()
    alias src_N = src_layout.shape[1].value()
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias dst_M = dst_layout.shape[0].value()
    alias dst_N = dst_layout.shape[1].value()

    var src = ManagedLayoutTensor[DType.float32, src_layout](ctx)
    var dst = ManagedLayoutTensor[DType.float32, dst_layout](ctx)

    arange(src.tensor(), 1)
    arange(dst.tensor(), 100001)
    var tma_tensor = create_tma_tile[tileM, tileN](ctx, src.device_tensor())
    ctx.synchronize()

    alias kernel = test_tma_mcast_load_kernel[
        type_of(tma_tensor).dtype,
        dst_layout,  # dst layout
        type_of(tma_tensor).layout,  # smem layout
        type_of(tma_tensor).layout,  # thread layout
        CLUSTER_M,
        CLUSTER_N,
    ]

    ctx.enqueue_function_checked[kernel, kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(dst_N // tileN, dst_M // tileM),
        block_dim=(tileN * tileM),
        cluster_dim=Dim(CLUSTER_N, CLUSTER_M, 1),
    )

    src_host = src.tensor()
    dst_host = dst.tensor()

    for m in range(dst_M):
        for n in range(dst_N):
            assert_equal(
                dst_host[m, n].cast[DType.float32](),
                src_host[m, n % src_N].cast[DType.float32](),
            )

    ctx.synchronize()
    _ = src^
    _ = dst^


# Test loading a single 2d tile.
@__llvm_arg_metadata(tma_tile, `nvvm.grid_constant`)
fn test_tma_sliced_multicast_load_kernel[
    dtype: DType,
    layout: Layout,
    tile_layout: Layout,
    thread_layout: Layout,
    CLUSTER_M: UInt32,
    CLUSTER_N: UInt32,
    tma_layout: Layout,
](
    dst: LayoutTensor[dtype, layout, MutableAnyOrigin],
    tma_tile: TMATensorTile[dtype, tma_layout],
):
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias expected_bytes = tile_layout.size() * size_of[dtype]()

    var block_rank = block_rank_in_cluster()
    alias CLUSTER_SIZE = CLUSTER_M * CLUSTER_N

    var rank_m = block_rank // CLUSTER_N
    var rank_n = block_rank % CLUSTER_N

    var tma_multicast_mask = (1 << CLUSTER_N) - 1

    tile = (
        LayoutTensor[
            dtype,
            tile_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]
        .stack_allocation()
        .fill(10)
    )

    barrier()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=8,
    ]()
    if thread_idx.x == 0:
        mbar[0].init()

    barrier()

    # we use cluster_sync() together with a mbarrier init fence to ensure cluster-wide visibility of the mbarrier initialization
    cluster_sync()
    fence_mbarrier_init()

    if thread_idx.x == 0:
        mbar[0].expect_bytes(expected_bytes)
        var slice_cord = Int(
            block_idx.y * UInt(tileM)
            + Int(block_rank % CLUSTER_N) * tileM // CLUSTER_N
        )
        var multicast_mask = tma_multicast_mask << (rank_m * CLUSTER_N)
        tma_tile.async_multicast_load(
            type_of(tile)(
                tile.ptr + (block_rank % CLUSTER_N) * tileM * tileN // CLUSTER_N
            ),
            mbar[0],
            (UInt(0), UInt(slice_cord)),
            multicast_mask.cast[DType.uint16](),
        )

    barrier()

    # after this line, the TMA load is finished
    mbar[0].wait()

    # we use another cluster_sync() to ensure that one of the two CTAs in the cluster doesn’t exit prematurely while the other is still waiting for the multicast load to complete.
    cluster_sync()

    dst_tile = dst.tile[tileM, tileN](block_idx.y, block_idx.x)
    copy_sram_to_dram[thread_layout](dst_tile, tile)


def test_tma_sliced_multicast_load_row_major[
    src_layout: Layout,
    tile_layout: Layout,
    dst_layout: Layout,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
](ctx: DeviceContext):
    alias src_M = src_layout.shape[0].value()
    alias src_N = src_layout.shape[1].value()
    alias tileM = tile_layout.shape[0].value()
    alias tileN = tile_layout.shape[1].value()
    alias dst_M = dst_layout.shape[0].value()
    alias dst_N = dst_layout.shape[1].value()

    var src = ManagedLayoutTensor[DType.float32, src_layout](ctx)
    var dst = ManagedLayoutTensor[DType.float32, dst_layout](ctx)

    arange(src.tensor(), 1)
    arange(dst.tensor(), 100001)
    var tma_tensor = create_tma_tile[tileM // CLUSTER_N, tileN](
        ctx, src.device_tensor()
    )
    ctx.synchronize()

    alias kernel = test_tma_sliced_multicast_load_kernel[
        type_of(tma_tensor).dtype,
        dst_layout,  # dst layout
        Layout.row_major(tileM, tileN),
        Layout.row_major(tileM, tileN),
        CLUSTER_M,
        CLUSTER_N,
        type_of(tma_tensor).layout,  # smem layout
    ]

    ctx.enqueue_function_checked[kernel, kernel](
        dst.device_tensor(),
        tma_tensor,
        grid_dim=(dst_N // tileN, dst_M // tileM),
        block_dim=(tileN * tileM),
        cluster_dim=Dim(CLUSTER_N, CLUSTER_M, 1),
    )

    src_host = src.tensor()
    dst_host = dst.tensor()

    for m in range(dst_M):
        for n in range(dst_N):
            assert_equal(
                dst_host[m, n].cast[DType.float32](),
                src_host[m, n % src_N].cast[DType.float32](),
            )

    ctx.synchronize()
    _ = src^
    _ = dst^


def main():
    with DeviceContext() as ctx:
        print("test_tma_multicast_load_row_major")
        test_tma_multicast_load_row_major[
            src_layout = Layout.row_major(8, 8),
            tile_layout = Layout.row_major(4, 8),
            dst_layout = Layout.row_major(8, 16),
            CLUSTER_M=1,
            CLUSTER_N=2,
        ](ctx)
        test_tma_multicast_load_row_major[
            src_layout = Layout.row_major(16, 8),
            tile_layout = Layout.row_major(4, 8),
            dst_layout = Layout.row_major(16, 16),
            CLUSTER_M=2,
            CLUSTER_N=2,
        ](ctx)

        print("test_tma_sliced_multicast_load_row_major")
        test_tma_sliced_multicast_load_row_major[
            src_layout = Layout.row_major(8, 16),
            tile_layout = Layout.row_major(4, 16),
            dst_layout = Layout.row_major(8, 32),
            CLUSTER_M=1,
            CLUSTER_N=2,
        ](ctx)
        test_tma_sliced_multicast_load_row_major[
            src_layout = Layout.row_major(16, 16),
            tile_layout = Layout.row_major(4, 16),
            dst_layout = Layout.row_major(16, 32),
            CLUSTER_M=2,
            CLUSTER_N=2,
        ](ctx)
        test_tma_sliced_multicast_load_row_major[
            src_layout = Layout.row_major(32, 16),
            tile_layout = Layout.row_major(4, 16),
            dst_layout = Layout.row_major(32, 32),
            CLUSTER_M=4,
            CLUSTER_N=2,
        ](ctx)
        test_tma_sliced_multicast_load_row_major[
            src_layout = Layout.row_major(32, 16),
            tile_layout = Layout.row_major(16, 16),
            dst_layout = Layout.row_major(32, 64),
            CLUSTER_M=2,
            CLUSTER_N=4,
        ](ctx)
