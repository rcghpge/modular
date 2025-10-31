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
from sys import simd_width_of, size_of

from gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import lane_id
from gpu.memory import fence_async_view_proxy
from gpu.mma import st_matrix
from gpu.sync import named_barrier
from layout import IntTuple, Layout, LayoutTensor
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout, RuntimeTuple
from layout.swizzle import Swizzle, make_ldmatrix_swizzle
from layout.tensor_core_async import st_matrix_n_layout
from layout.tma_async import TMATensorTile
from memory import bitcast
from stdlib.bit import log2_floor

from utils.index import IndexList

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....structuring import (
    SMemTileType,
    RegTileType,
)
from .tile_writer import (
    TileWriterTMA,
    TileWriterThreadwise,
    FragmentToSMemWriter,
    RegisterToGMemWriter,
    TileCoordinates,
    RegTileWriter,
)
import itertools


@register_passable("trivial")
struct MatmulTileWriter[
    dtype: DType,
    layout: Layout,
    address_space: AddressSpace,
    element_layout: Layout,
    layout_int_type: DType,
    linear_idx_type: DType,
    masked: Bool,
    alignment: Int,
    smem_tile_layout: Layout, //,
    /,
    *,
    BM: Int,
    BN: Int,
    swizzle: TensorMapSwizzle,
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
]:
    alias N = layout.shape[1].value()
    alias frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE
    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]
    alias num_consumer_threads = num_consumer * WARPGROUP_SIZE
    alias simd_size = simd_width_of[dtype]()

    # Layout dimensions
    alias WG_BM = smem_tile_layout.shape[0].value()
    alias WG_BN = smem_tile_layout.shape[1].value()

    alias CTensorType = LayoutTensor[
        dtype,
        layout,
        MutableAnyOrigin,
        address_space=address_space,
        element_layout=element_layout,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=masked,
        alignment=alignment,
    ]
    alias lambda_type = fn[dtype: DType, width: Int, *, alignment: Int = 1] (
        IndexList[2], mut SIMD[dtype, width]
    ) capturing -> None

    # Instance fields
    var tensor: Self.CTensorType
    var smem_tile: SMemTileType[dtype, smem_tile_layout, alignment=128]
    var warp_group_thread_idx: UInt
    var local_warp_group_idx: UInt
    var local_thread_idx: UInt
    var block_y: Int
    var block_x: Int

    @always_inline
    fn __init__(
        out self,
        tensor: Self.CTensorType,
        smem_tile: SMemTileType[dtype, smem_tile_layout, alignment=128],
        warp_group_thread_idx: UInt,
        local_warp_group_idx: UInt,
        local_thread_idx: UInt,
        block_y: Int,
        block_x: Int,
    ):
        self.tensor = tensor
        self.smem_tile = smem_tile
        self.warp_group_thread_idx = warp_group_thread_idx
        self.local_warp_group_idx = local_warp_group_idx
        self.local_thread_idx = local_thread_idx
        self.block_y = block_y
        self.block_x = block_x

    @always_inline
    fn _calculate_output_bounds(self) -> Tuple[UInt32, UInt32]:
        """Calculate valid output bounds for the current block's tile."""
        var rows = self.tensor.dim[0]()
        var max_row = min(UInt32((self.block_y + 1) * BM), UInt32(rows))
        var max_col = min(UInt32((self.block_x + 1) * BN), UInt32(Self.N))
        return max_row, max_col

    @always_inline
    fn _apply_epilogue[
        epilogue_fn: Self.lambda_type
    ](
        self,
        output_tile: LayoutTensor[dtype, _, MutableAnyOrigin, *_, **_],
        tile_row_offset: Int,
        tile_col_offset: Int,
        max_row: UInt32,
        max_col: UInt32,
    ):
        """Apply epilogue operations (bias, activation) to shared memory data.
        """
        alias epilogue = epilogue_fn
        alias smem_swizzle = make_ldmatrix_swizzle[dtype, Self.WG_BN]()
        alias thread_layout = Layout.row_major(
            Self.num_consumer_threads // (Self.WG_BN // Self.simd_size),
            Self.WG_BN // Self.simd_size,
        )

        var output_fragment, fragment_offsets, _ = output_tile.vectorize[
            1, Self.simd_size
        ]().distribute_with_offset[thread_layout](self.local_thread_idx)
        var row_coord = tile_row_offset + fragment_offsets[0]
        var col_coord = tile_col_offset + fragment_offsets[1] * Self.simd_size

        var shared_fragment = self.smem_tile.vectorize[
            1, Self.simd_size
        ]().distribute[thread_layout, swizzle=smem_swizzle](
            self.local_thread_idx
        )

        alias num_elements_per_thread = output_fragment.layout.size()

        @parameter
        for i in range(num_elements_per_thread):
            alias smem_idx = shared_fragment.layout(i)
            alias output_idx = output_fragment.layout(i)
            alias row_offset = output_idx // Self.N
            alias col_offset = output_idx % Self.N
            var row = UInt32(row_coord + row_offset)
            var col = UInt32(col_coord + col_offset)

            if row < max_row and col < max_col:
                epilogue(
                    IndexList[2](Int(row), Int(col)),
                    shared_fragment[i, 0],
                )

    @always_inline
    fn _write_tile_to_gmem[
        accum_type: DType,
        reg_tile_layout: Layout, //,
        check_runtime_bounds: Bool = False,
    ](self, reg_tile: RegTileType[accum_type, reg_tile_layout]):
        """Write from registers to global memory."""
        var output_tile, tile_origin, _ = self.tensor.tile_with_offset[BM, BN](
            self.block_y, self.block_x
        )
        var consumer_tile, consumer_coords, _ = output_tile.tile_with_offset[
            BM // Self.num_consumer, BN
        ](Int(self.local_warp_group_idx), 0)

        var tile_coords: OptionalReg[TileCoordinates] = None
        var max_row: OptionalReg[UInt32] = None

        @parameter
        if (
            elementwise_lambda_fn is not None
            or elementwise_compute_lambda_fn is not None
        ):
            tile_coords = TileCoordinates(
                IndexList[2](tile_origin[0], tile_origin[1]),
                IndexList[2](consumer_coords[0], consumer_coords[1]),
            )
            max_row = UInt32(self.tensor.dim[0]())

        var reg_writer = RegisterToGMemWriter[
            wgmma_shape=wgmma_shape,
            num_consumer=num_consumer,
            N = Self.N,
            epilogue_fn=elementwise_lambda_fn,
            compute_lambda_fn=elementwise_compute_lambda_fn,
            check_runtime_bounds=check_runtime_bounds,
        ](
            consumer_tile,
            self.warp_group_thread_idx,
            Self.num_m_mmas,
            tile_coords,
            max_row,
        )

        @parameter
        for row_tile, col_tile in itertools.product(
            range(Self.num_m_mmas), range(Self.num_n_mmas)
        ):
            reg_writer.write_tile(
                reg_tile,
                (UInt(row_tile), UInt(col_tile)),
            )

    @always_inline
    fn _write_tile_stmatrix[
        tma_layout: Layout,
        desc_layout: Layout,
        accum_type: DType,
        reg_tile_layout: Layout, //,
    ](
        self,
        tma_op: TMATensorTile[dtype, tma_layout, desc_layout],
        reg_tile: RegTileType[accum_type, reg_tile_layout],
        output_tile: LayoutTensor[dtype, _, MutableAnyOrigin, *_, **_],
        tile_origin: Self.CTensorType.CornerCoordsType,
    ):
        """Use st.matrix instructions for optimized bf16 output."""
        var max_row, max_col = self._calculate_output_bounds()

        alias TMA_BN = tma_layout.shape[
            1
        ].value() if use_tma_store else Self.WG_BN
        alias needs_x2 = BN % Self.WG_BN != 0

        constrained[
            needs_x2 == (Self.frag_size % 4 == 0 and Self.frag_size % 8 != 0),
            "stmatrix and wgmma register count conflict: needs_x2 = "
            + String(needs_x2)
            + " frag_size ="
            + String(Self.frag_size),
        ]()

        alias fragment_writer_type[
            sub_wg_id: Int, half_tile: Bool
        ] = FragmentToSMemWriter[
            tile_n_size=TMA_BN,
            num_m_mmas = Self.num_m_mmas,
            num_consumer=num_consumer,
            half_tile=half_tile,
            WG_BM = Self.WG_BM,
            WG_BN = Self.WG_BN,
            sub_wg_id=sub_wg_id,
        ]

        alias num_column_tiles = ceildiv(BN, Self.WG_BN)
        alias last_tile = BN // Self.WG_BN

        @parameter
        for tile_idx in range(num_column_tiles):
            alias is_partial_tile = needs_x2 and tile_idx == last_tile

            # Write fragments to shared memory
            var fragment_writer = fragment_writer_type[
                tile_idx, is_partial_tile
            ](
                self.smem_tile,
                self.warp_group_thread_idx,
                self.local_warp_group_idx,
            )

            @parameter
            for tma_chunk in range(Self.WG_BN // TMA_BN):
                fragment_writer.write_tile(reg_tile, (UInt(0), UInt(tma_chunk)))

            named_barrier[Self.num_consumer_threads](10)

            var workgroup_tile, tile_coords, _ = output_tile.tile_with_offset[
                BM, Self.WG_BN
            ](0, tile_idx)
            var global_coords = (
                rebind[Self.CTensorType.CornerCoordsType](tile_coords)
                + tile_origin
            )

            @parameter
            fn apply_epilogue[lambda_fn: Self.lambda_type]():
                self._apply_epilogue[lambda_fn](
                    workgroup_tile,
                    global_coords[0],
                    global_coords[1],
                    max_row,
                    max_col,
                )

            @parameter
            if elementwise_compute_lambda_fn:
                alias compute_fn = elementwise_compute_lambda_fn.value()

                @parameter
                fn _compute[
                    dtype: DType, width: Int, *, alignment: Int = 1
                ](
                    index: IndexList[2], mut val: SIMD[dtype, width]
                ) capturing -> None:
                    val = compute_fn[alignment=alignment](index, val)

                apply_epilogue[_compute]()
                named_barrier[Self.num_consumer_threads](10)

            @parameter
            if elementwise_lambda_fn:
                alias epilogue_fn = elementwise_lambda_fn.value()

                @parameter
                fn _epilogue[
                    dtype: DType, width: Int, *, alignment: Int = 1
                ](
                    index: IndexList[2], mut val: SIMD[dtype, width]
                ) capturing -> None:
                    _ = epilogue_fn[alignment=alignment](index, val)

                apply_epilogue[_epilogue]()
            else:

                @parameter
                if use_tma_store and not is_partial_tile:
                    var tma_writer = TileWriterTMA(Pointer(to=tma_op))

                    if self.local_thread_idx < UInt(Self.WG_BN // TMA_BN):
                        var smem_offset = self.smem_tile.ptr.offset(
                            Self.WG_BM * TMA_BN * Int(self.local_thread_idx)
                        )
                        var tma_tile = SMemTileType[
                            dtype, tma_layout, alignment=128
                        ](smem_offset)

                        var tma_coords = (
                            UInt(
                                self.block_x * BN
                                + tile_idx * Self.WG_BN
                                + Int(self.local_thread_idx * UInt(TMA_BN))
                            ),
                            UInt(self.block_y * BM),
                        )

                        tma_writer.write_tile(tma_tile, tma_coords)
                else:
                    alias thread_layout = Layout.row_major(
                        Self.num_consumer_threads
                        // (Self.WG_BN // Self.simd_size),
                        Self.WG_BN // Self.simd_size,
                    )

                    var threadwise_writer = TileWriterThreadwise[
                        thread_layout=thread_layout,
                        simd_size = Self.simd_size,
                        half_tile=is_partial_tile,
                    ](workgroup_tile, self.local_thread_idx)

                    threadwise_writer.write_tile(
                        self.smem_tile, (UInt(0), UInt(0))
                    )

            named_barrier[Self.num_consumer_threads](10)

    @always_inline
    fn write_tile[
        tma_layout: Layout,
        desc_layout: Layout,
        accum_type: DType,
        reg_tile_layout: Layout, //,
    ](
        self,
        tma_op: TMATensorTile[dtype, tma_layout, desc_layout],
        reg_tile: RegTileType[accum_type, reg_tile_layout],
    ):
        """Write output from registers to global memory.

        Selects optimized st.matrix path for bf16 when constraints are met,
        otherwise uses general register-to-global path.
        """
        var output_tile, tile_origin, _ = self.tensor.tile_with_offset[BM, BN](
            self.block_y, self.block_x
        )

        alias TMA_BN = tma_layout.shape[
            1
        ].value() if use_tma_store else Self.WG_BN
        alias row_size_aligned = Self.N * size_of[dtype]() % 16 == 0

        # Check if st.matrix optimization can be used
        # fmt: off
        alias can_use_stmatrix = (
            accum_type is DType.float32 and dtype is DType.bfloat16  # F32→BF16
            and Self.frag_size % 4 == 0                               # Register count
            and BM % wgmma_shape[0] == 0                              # M alignment
            and Self.WG_BN % 16 == 0                                  # Shared memory
            and num_consumer <= 2                                     # Thread limit
            and BN == wgmma_shape[1]                                  # Tile size
            and BM == Self.WG_BM                                      # Block size
            and row_size_aligned                                      # Row alignment
        )
        # fmt: on

        @parameter
        if can_use_stmatrix:
            self._write_tile_stmatrix(
                tma_op,
                reg_tile,
                output_tile,
                tile_origin,
            )
        else:
            self._write_tile_to_gmem[check_runtime_bounds = (Self.N % BN != 0)](
                reg_tile
            )
