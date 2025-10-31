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
from math import ceildiv, exp2, recip, align_up, align_down, gcd
from math.constants import log2e
from sys import align_of, simd_width_of, size_of, env_get_int
import gpu.warp as warp
from algorithm.functional import unswitch
from bit import prev_power_of_two, pop_count
from buffer import NDBuffer
from collections import OptionalReg
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
)
from gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from gpu.cluster import elect_one_sync
from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory
from gpu.mma import MMAOperandDescriptor
from gpu.mma_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma,
    mma_arrive,
)
from gpu.sync import named_barrier
from gpu.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
from gpu.primitives.warp import _vote_nvidia_helper
from layout.int_tuple import IntTuple, UNKNOWN_VALUE
from layout.layout import Layout, blocked_product
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMANestedTensorTile,
)
from memory import bitcast
from nn.mha_fa3_utils import (
    _get_position,
    MHAPosition,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    output_reg_to_smem,
    output_gmem_to_smem_STMatrix,
    Pack,
    produce,
    q_out_tma,
    QTMATile,
)
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
    TransientScheduler,
)
from nn.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from nn.softmax import (
    _online_softmax_correction,
    _rowmax_online_softmax,
    _rowsum,
)
from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor

from pathlib import Path

alias LocalTensor[
    dtype: DType, layout: Layout, element_layout: Layout = Layout(1, 1)
] = LayoutTensor[
    dtype,
    layout,
    MutableAnyOrigin,
    address_space = AddressSpace.LOCAL,
    element_layout=element_layout,
]
alias SharedMemTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype,
    layout,
    MutableAnyOrigin,
    address_space = AddressSpace.SHARED,
    layout_int_type = DType.int32,
    linear_idx_type = DType.int32,
    alignment=128,
]
alias SharedMemPointer[type: AnyType] = UnsafePointer[
    type, address_space = AddressSpace.SHARED
]
alias MBarType = SharedMemPointer[SharedMemBarrier]


fn extract_power_of_two(N: Int, i: Int) -> Int:
    pt = prev_power_of_two(N)
    rem = N
    for _ in range(i):
        rem -= pt
        pt = prev_power_of_two(rem)
    return pt


fn cumulative_power_of_two(N: Int, i: Int) -> Int:
    acc = 0
    rem = N
    for _ in range(i):
        pt = prev_power_of_two(rem)
        acc += pt
        rem -= pt
    return acc


# Final call is with `pow_two == 0` (which isn't a power of 2)
# to enable use of this function with pipelining.
@always_inline("nodebug")
fn break_into_powers_of_two[
    origins: OriginSet, //,
    func: fn[pow_two: Int, offset: Int] () capturing [origins] -> None,
    N: Int,
    *,
    max_value: Int = 128,
]():
    alias power_of_two = prev_power_of_two(min(max_value, N))

    @parameter
    for offset in range(0, N, power_of_two):
        alias iter_size = min(N - offset, power_of_two)

        @parameter
        if iter_size == power_of_two:
            func[power_of_two, offset]()
        else:

            @parameter
            for j in range(pop_count(iter_size)):
                alias pow_two = extract_power_of_two(iter_size, j)
                alias coffset = offset + cumulative_power_of_two(iter_size, j)
                func[pow_two, coffset]()
    # final call for possible pipeline cleanup
    func[0, N]()


@register_passable("trivial")
struct STMatrixLayout[
    BM: Int,
    BN: Int,
    *,
    num_threads: Int,
    accum_type_size: Int,
]:
    """
    Layout for using `st_matrix` for writing the final accumulator to smem.
    """

    # We have a BM x BN tile
    #
    # The st_matrix layout wants to map it to threads in 16x8 blocks
    # shape  (2,8), (2,4)
    # stride (0,4), (0,1)
    # Layout = ((2,8),(2,4)):((0,4),(0,1))
    # Where `0` stride indicates that the same thread is repeated across these.
    # We also need a layout for this local memory, which we define here.
    # That first `2` is
    alias num_row_blocks_per_mma = 2
    # The second `2` is
    alias frag_simdwidth: Int = 2

    alias thread_cols = 4
    # When using tcgen05 ld/st we must repeat across all columns:
    alias repeat = BN // (Self.thread_cols * Self.frag_simdwidth)

    alias num_warpgroups = ceildiv(num_threads, 128)
    # 2 = 32 // 16, i.e. we need to load 2 sets of 16
    alias num_m_tiles_total = ceildiv(2 * BM, 128)
    alias num_m_tiles = Self.num_m_tiles_total // Self.num_warpgroups

    alias frag_size = BN * Self.num_row_blocks_per_mma // Self.thread_cols

    # layout of local memory
    # alias local_layout: Layout = Layout(
    #     IntTuple(IntTuple(Self.num_row_blocks_per_mma, Self.num_m_tiles),IntTuple(Self.frag_simdwidth, Self.repeat)),
    #     IntTuple(IntTuple(Self.frag_simdwidth, Self.frag_size),IntTuple(1, Self.num_row_blocks_per_mma*Self.frag_simdwidth)),
    # )
    alias elements_per_repeat = Self.frag_simdwidth * Self.num_row_blocks_per_mma

    alias vec_local_layout: Layout = Layout(
        IntTuple(
            IntTuple(Self.num_row_blocks_per_mma, Self.num_m_tiles),
            IntTuple(Self.repeat),
        ),
        IntTuple(
            IntTuple(Self.frag_simdwidth, Self.frag_size),
            IntTuple(Self.num_row_blocks_per_mma * Self.frag_simdwidth),
        ),
    )
    alias element_layout: Layout = Layout.row_major(1, Self.frag_simdwidth)
    alias TensorType[dtype: DType] = LocalTensor[
        dtype, Self.vec_local_layout, Self.element_layout
    ]
    alias row_of_frags_layout: Layout = Layout.row_major(
        Self.num_m_tiles, Self.frag_size
    )

    alias bits_per_byte = 8
    alias bits = Self.bits_per_byte * Self.frag_simdwidth * Self.thread_cols * accum_type_size

    @always_inline
    fn __init__(out self):
        pass


@register_passable("trivial")
struct STMatrixOffsets[
    BM: Int,
    BN: Int,
    *,
    num_threads: Int,
    accum_type_size: Int,
    curr_repeat: Int,
    cumulative_repeat: Int,
    m_mma: Int,
]:
    alias STLayout = STMatrixLayout[
        BM, BN, num_threads=num_threads, accum_type_size=accum_type_size
    ]

    alias tmem_col_offset = cumulative_repeat * Self.STLayout.frag_simdwidth * Self.STLayout.thread_cols
    alias tmem_row_offset = 16 * m_mma
    alias tmem_offset = (Self.tmem_row_offset << 16) + Self.tmem_col_offset
    alias b32_per_repeat = Self.STLayout.elements_per_repeat * accum_type_size // 4
    alias local_frag_size_b32 = curr_repeat * Self.b32_per_repeat
    alias ptr_offset = Self.b32_per_repeat * (
        Self.STLayout.repeat * m_mma + cumulative_repeat
    )

    @always_inline
    fn __init__(out self):
        pass


@always_inline
fn _tmem_offset(dtype_size: Int, *, MMA_N: Int, m_mma: Int, n_mma: Int) -> Int:
    row = 16 * m_mma
    col = (MMA_N * n_mma * dtype_size) // 4
    return (row << 16) + col


@always_inline
fn _tmem_offset[dtype: DType, *, MMA_N: Int, m_mma: Int, n_mma: Int]() -> Int:
    alias linear = _tmem_offset(
        dtype.size_of(), MMA_N=MMA_N, m_mma=m_mma, n_mma=n_mma
    )
    return linear


@register_passable("trivial")
struct TMemTile[
    dtype_: DType,
    BM: Int,
    BN: Int,
]:
    alias dtype: DType = dtype_
    alias dtype_size = Self.dtype.size_of()
    # alias layout_t = STMatrixLayout[
    #     BM, BN, num_threads= num_threads
    # ]
    # alias vec_output_layout = Self.layout_t.vec_local_layout
    # alias element_layout = Self.layout_t.element_layout
    alias num_m_tiles = BM // 64

    var tmem_addr: UInt32

    @always_inline
    fn __init__(out self, tmem_addr: UInt32):
        self.tmem_addr = tmem_addr

    @always_inline
    fn __getitem__(self, i: UInt32) -> Self:
        return {self.tmem_addr + i * BN}

    @always_inline
    fn offset[m_mma: Int, n_mma: Int](self) -> UInt32:
        @parameter
        if m_mma == 0 and n_mma == 0:
            return self.tmem_addr
        else:
            alias linear = _tmem_offset[
                Self.dtype, MMA_N=BN, m_mma=m_mma, n_mma=n_mma
            ]()

            return self.tmem_addr + linear

    @staticmethod
    @always_inline
    fn allocate_register_tile[
        *, num_threads: Int
    ](
        out res: STMatrixLayout[BM, BN, num_threads=num_threads].TensorType[
            Self.dtype
        ],
    ):
        res = type_of(res).stack_allocation()

    @always_inline
    fn store_async[
        *, num_threads: Int
    ](
        self,
        src: STMatrixLayout[BM, BN, num_threads=num_threads].TensorType[
            Self.dtype
        ],
    ):
        constrained[Self.dtype_size <= 4]()
        ptr = src.ptr.bitcast[UInt32]()
        alias st_mat_layout = STMatrixLayout[
            BM, BN, num_threads=num_threads, accum_type_size = Self.dtype_size
        ]
        constrained[st_mat_layout.bits == 128 or st_mat_layout.bits == 256]()

        @parameter
        @always_inline
        fn store_fn[pow_two: Int, offset: Int]():
            # pow_two is current repeat, offset total so far
            @parameter
            if pow_two > 0:

                @parameter
                for m_mma in range(st_mat_layout.num_m_tiles):
                    alias offsets = STMatrixOffsets[
                        BM,
                        BN,
                        num_threads=num_threads,
                        accum_type_size = Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=offset,
                        m_mma=m_mma,
                    ]()
                    tmem = self.tmem_addr + offsets.tmem_offset
                    frag = ptr.load[width = offsets.local_frag_size_b32](
                        offsets.ptr_offset
                    )
                    # 16 x 256b results in repeated 8x4 matrix of <1,2> vector pattern
                    tcgen05_st[
                        datapaths=16,  # first dimension of the shape
                        bits = st_mat_layout.bits,  # second dimension of the shape
                        repeat=pow_two,
                        pack=False,
                    ](tmem, frag)

        alias max_value = 64 if st_mat_layout.bits == 128 else 32
        break_into_powers_of_two[
            func=store_fn, N = st_mat_layout.repeat, max_value=max_value
        ]()

    @always_inline
    fn store[
        *, num_threads: Int
    ](
        self,
        src: STMatrixLayout[BM, BN, num_threads=num_threads].TensorType[
            Self.dtype
        ],
    ):
        self.store_async[num_threads=num_threads](src)
        tcgen05_store_wait()
        named_barrier[num_threads]()

    @always_inline
    fn load_async_st_matrix[
        *, num_threads: Int
    ](
        self,
        out dst: STMatrixLayout[BM, BN, num_threads=num_threads].TensorType[
            Self.dtype
        ],
    ):
        constrained[
            Self.dtype_size <= 4,
            "Loading for st matrix requires elements to be <= 4 bytes.",
        ]()
        alias st_mat_layout = STMatrixLayout[
            BM, BN, num_threads=num_threads, accum_type_size = Self.dtype_size
        ]()
        constrained[
            (st_mat_layout.num_m_tiles == 1)
            or (st_mat_layout.num_m_tiles == 2),
            "Only 1 or 2 m tiles are supported, but"
            " st_mat_layout.num_m_tiles == "
            + String(st_mat_layout.num_m_tiles),
        ]()
        alias bits_per_byte = 8
        alias bits = st_mat_layout.bits
        alias repeat = st_mat_layout.repeat
        alias frag_size_b32 = st_mat_layout.frag_size * Self.dtype_size // 4

        dst = type_of(dst).stack_allocation()
        alias load_dtype = DType.uint32
        # alias load_dtype = Self.dtype if Self.dtype_size == 4 else DType.uint32
        var ptr: UnsafePointer[
            Scalar[load_dtype], address_space = AddressSpace.LOCAL
        ]

        @parameter
        if load_dtype == DType.uint32:
            ptr = rebind[type_of(ptr)](dst.ptr)
        else:
            ptr = rebind[type_of(ptr)](dst.ptr.bitcast[UInt32]())

        constrained[
            st_mat_layout.num_m_tiles, "this is just a check we'll drop"
        ]()

        @parameter
        @always_inline
        fn load_fn[pow_two: Int, offset: Int]():
            constrained[pow_two + offset <= repeat]()

            @parameter
            if pow_two > 0:

                @parameter
                for m_mma in range(st_mat_layout.num_m_tiles):
                    alias offsets = STMatrixOffsets[
                        BM,
                        BN,
                        num_threads=num_threads,
                        accum_type_size = Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=offset,
                        m_mma=m_mma,
                    ]()
                    tmem = self.tmem_addr + offsets.tmem_offset
                    frag = tcgen05_ld[
                        datapaths=16,  # first dimension of the shape
                        bits = st_mat_layout.bits,  # second dimension of the shape
                        repeat=pow_two,
                        dtype=load_dtype,
                        pack=False,
                        width = offsets.local_frag_size_b32,
                    ](tmem)
                    ptr.store(offsets.ptr_offset, frag)

        alias max_value = 64 if st_mat_layout.bits == 128 else 32
        break_into_powers_of_two[func=load_fn, N=repeat, max_value=max_value]()

    @always_inline
    fn load_async(
        self,
        out dst: LocalTensor[Self.dtype, Layout.row_major(BN)],
    ):
        dst = type_of(dst).stack_allocation()
        alias repeat = Self.dtype_size * BN // 4
        alias dtype = Self.dtype if Self.dtype_size == 4 else DType.uint32

        @parameter
        @always_inline
        fn load_fn[pow_two: Int, offset: Int]():
            @parameter
            if pow_two > 0:

                @parameter
                if dtype == Self.dtype:
                    frag0 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype = Self.dtype,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + offset)
                    dst.ptr.store(offset, frag0)
                else:
                    frag1 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype = DType.uint32,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + offset)
                    dst.ptr.bitcast[UInt32]().store[width=pow_two](
                        offset, frag1
                    )

        break_into_powers_of_two[func=load_fn, N=repeat, max_value=128]()

    @always_inline
    fn store_async[
        src_type: DType
    ](self, src: LocalTensor[src_type, Layout.row_major(BN)]):
        @parameter
        @always_inline
        fn store_fn[pow_two: Int, offset: Int]():
            @parameter
            if pow_two > 0:
                var frag: SIMD[DType.uint32, pow_two * Self.dtype_size // 4]

                @parameter
                if src_type == Self.dtype:
                    frag = src.ptr.bitcast[UInt32]().load[
                        width = pow_two * Self.dtype_size // 4
                    ](offset)
                else:
                    # if thread_idx.x == 9:
                    #     print("pow_two =", pow_two)
                    #     print("offset =", offset)
                    alias src_offset = offset
                    alias src_frag = pow_two
                    frag = bitcast[
                        DType.uint32, pow_two * Self.dtype_size // 4
                    ](
                        src.ptr.load[width=src_frag](src_offset).cast[
                            Self.dtype
                        ]()
                    )
                tcgen05_st[
                    datapaths=32,  # first dimension of the shape
                    bits=32,  # second dimension of the shape
                    repeat = pow_two * Self.dtype_size // 4,
                    pack=False,
                ](self.tmem_addr + offset * Self.dtype_size // 4, frag)

        break_into_powers_of_two[func=store_fn, N=BN, max_value=128]()

    @always_inline
    fn store[
        src_type: DType
    ](self, src: LocalTensor[src_type, Layout.row_major(BN)]):
        self.store_async(src)
        tcgen05_store_wait()


@register_passable("trivial")
struct SM100TensorAccumulatorSS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BK: Int,
    *,
    swizzle_a: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    cta_group: Int = 1,
    num_stages: Int = 1,
]:
    # This performs C = A @ B
    # where A is BM x BK and B is BN x BK if k major, else BK x BN.
    # `BK` is broken into `num_stages` and pipelined.
    #
    # The complete multiplication of all stages produces an unweighted
    # score, which is the input of the `softmax`.
    # The benefit of setting `stages > 1` is that this can hide latency.
    alias operand_t: DType = operand_type
    alias accum_t: DType = accum_type
    alias MMA_K = 16
    alias num_k_mmas = BK // Self.MMA_K
    alias swizzle_granularity = max(
        swizzle_a.bytes(), swizzle_b.bytes()
    ) // operand_type.size_of()
    alias padded_BK = align_up(BK, Self.swizzle_granularity)
    alias num_k_blocks = Self.padded_BK // Self.MMA_K
    alias num_k_blocks_per_stage = Self.num_k_blocks // num_stages

    alias a_layout = tile_layout_k_major[
        Self.operand_t, align_up(MMA_M, 8), Self.padded_BK, swizzle_a
    ]()
    alias b_layout = tile_layout_k_major[
        Self.operand_t, MMA_N, Self.padded_BK, swizzle_b
    ]() if transpose_b else tile_layout_mn_major[
        Self.operand_t, MMA_N, Self.padded_BK, swizzle_b
    ]()

    alias idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype = DType.uint32](MMA_M, MMA_N),
        transpose_b=transpose_b,
    ]()

    alias AType = MMASmemDescriptor
    alias BType = MMASmemDescriptor
    alias CType = TMemTile[Self.accum_t, MMA_M, MMA_N]

    @staticmethod
    @always_inline("nodebug")
    fn mma[
        *, c_scale: UInt32, stage_idx: Int = 0
    ](a: Self.AType, b: Self.BType, c: UInt32):
        alias k_start = Self.num_k_blocks_per_stage * stage_idx
        alias k_stop = min(
            Self.num_k_blocks_per_stage + k_start, Self.num_k_mmas
        )

        @parameter
        for k_mma in range(k_start, k_stop):
            alias a_offset = Self.a_layout(IntTuple(0, Self.MMA_K * k_mma))
            alias a_offset_bytes = a_offset * size_of[Self.operand_t]()
            a_desc = a + a_offset_bytes

            alias b_offset = Self.b_layout(
                IntTuple(0, Self.MMA_K * k_mma)
            ) * size_of[Self.operand_t]()
            b_desc = b + b_offset

            if elect_one_sync():

                @parameter
                if k_mma == 0:
                    mma[cta_group, c_scale=c_scale](
                        a_desc,
                        b_desc,
                        c,
                        Self.idesc,
                    )
                else:
                    mma[cta_group, c_scale=1](a_desc, b_desc, c, Self.idesc)

    @staticmethod
    @always_inline
    fn mma[
        *, stage_idx: Int = 0
    ](a: Self.AType, b: Self.BType, c: UInt32, c_scale: UInt32,):
        @parameter
        if stage_idx != 0:
            Self.mma[c_scale=1](a, b, c)
        else:
            alias k_stop = min(Self.num_k_blocks_per_stage, Self.num_k_mmas)

            @parameter
            for k_mma in range(k_stop):
                alias a_offset = Self.a_layout(IntTuple(0, Self.MMA_K * k_mma))
                alias a_offset_bytes = a_offset * size_of[Self.operand_t]()
                a_desc = a + a_offset_bytes

                alias b_offset = Self.b_layout(
                    IntTuple(0, Self.MMA_K * k_mma)
                ) * size_of[Self.operand_t]()
                b_desc = b + b_offset

                if elect_one_sync():

                    @parameter
                    if k_mma == 0:
                        mma[cta_group](
                            a_desc,
                            b_desc,
                            c,
                            Self.idesc,
                            c_scale,
                        )
                    else:
                        mma[cta_group, c_scale=1](a_desc, b_desc, c, Self.idesc)


@register_passable("trivial")
struct SM100TensorAccumulatorTS[
    operand_type: DType,
    accum_type: DType,
    MMA_M: Int,
    MMA_N: Int,
    BK: Int,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    *,
    transpose_b: Bool = True,
    cta_group: Int = 1,
    num_stages: Int = 1,
    padded_BK: Int = BK,
]:
    alias operand_t: DType = operand_type
    alias accum_t: DType = accum_type

    alias operand_t_size = operand_type.size_of()
    alias swizzle_granularity = swizzle_b.bytes() // Self.operand_t_size
    # alias MMA_N_padded = align_up(MMA_N, Self.swizzle_granularity)
    # BN here is depth
    alias b_layout = tile_layout_k_major[
        Self.operand_t, MMA_N, BK, swizzle_b
    ]() if transpose_b else tile_layout_mn_major[
        Self.operand_t, MMA_N, BK, swizzle_b
    ]()

    alias MMA_K = 16
    alias num_k_mmas = BK // Self.MMA_K
    alias num_k_blocks = padded_BK // Self.MMA_K
    alias num_k_blocks_per_stage = Self.num_k_blocks // num_stages

    alias AType = TMemTile[operand_type, MMA_M, BK]
    alias BType = MMASmemDescriptor
    alias CType = TMemTile[Self.accum_t, MMA_M, MMA_N]

    # B's descriptor contains stride info, so we should be
    # able to use `BN` here instead of `BN_padded`
    alias idesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype = DType.uint32](MMA_M, MMA_N),
        transpose_b=transpose_b,
    ]()

    @staticmethod
    @always_inline
    fn descriptor_a(a_tmem: UInt32) -> Self.AType:
        return {a_tmem}

    @staticmethod
    @always_inline
    fn mma[
        *, stage_idx: Int = 0
    ](a: UInt32, b: Self.BType, c: UInt32, c_scale: UInt32,):
        @parameter
        if stage_idx != 0:
            Self.mma[c_scale=1, stage_idx=stage_idx](a, b, c)
        else:
            alias k_stop = min(Self.num_k_blocks_per_stage, Self.num_k_mmas)

            @parameter
            for k_mma in range(k_stop):
                alias k = Self.MMA_K * k_mma
                alias tmem_offset = (k * Self.operand_t_size) // 4
                a_tmem = a + tmem_offset

                alias b_offset = Self.b_layout(
                    IntTuple(0, k)
                ) * Self.operand_t_size
                b_desc = b + b_offset

                if elect_one_sync():

                    @parameter
                    if k_mma == 0:
                        mma[cta_group](
                            a_tmem,
                            b_desc,
                            c,
                            Self.idesc,
                            c_scale,
                        )
                    else:
                        mma[cta_group, c_scale=1](a_tmem, b_desc, c, Self.idesc)

    @staticmethod
    @always_inline
    fn mma[
        *, c_scale: UInt32, stage_idx: Int = 0
    ](a: UInt32, b: Self.BType, c: UInt32):
        alias k_start = Self.num_k_blocks_per_stage * stage_idx
        alias k_stop = min(
            Self.num_k_blocks_per_stage + k_start, Self.num_k_mmas
        )

        @parameter
        for k_mma in range(k_start, k_stop):
            alias k = Self.MMA_K * k_mma
            alias tmem_offset = (k * Self.operand_t_size) // 4

            a_tmem = a + tmem_offset

            alias b_offset = Self.b_layout(IntTuple(0, k)) * Self.operand_t_size
            b_desc = b + b_offset

            if elect_one_sync():

                @parameter
                if k_mma == 0:
                    mma[cta_group, c_scale=c_scale](
                        a_tmem,
                        b_desc,
                        c,
                        Self.idesc,
                    )
                else:
                    mma[cta_group, c_scale=1](a_tmem, b_desc, c, Self.idesc)


@register_passable("trivial")
struct FA4Config:
    var MMA_M: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var depth: Int
    var padded_depth: Int
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    alias TMEM_S0: Int = 0
    var TMEM_S1: Int
    var TMEM_O0: Int
    var TMEM_O1: Int
    var TMEM_P0: Int
    var TMEM_P1: Int
    var TMEM_C0: Int
    var TMEM_C1: Int
    var tmem_used: Int
    var num_kv_stages: Int
    var num_mma_stages: Int
    var smem_used: Int
    var dtype_size: Int
    alias num_threads: Int = 512  # 2x softmax, 1x correction, 1x other
    var split_m: Bool
    var swizzle_mode: TensorMapSwizzle

    alias MMA_K = 16
    alias sm100_smem_carveout = B200.shared_memory_per_multiprocessor - 1024
    alias sm100_tmem_cols = 512
    alias mbar_size = DType.int64.size_of()  # 8
    alias num_correction_cols = 1

    @always_inline
    fn num_qo(self) -> Int:
        return 2

    fn __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        depth: Int,
        dtype_size: Int,
        swizzle_mode: TensorMapSwizzle,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.depth = depth
        self.split_m = depth > 128
        if self.split_m:
            self.BM = 128
            self.MMA_M = 64
        else:
            self.BM = 256
            self.MMA_M = 128
        self.dtype_size = dtype_size
        self.swizzle_mode = swizzle_mode
        swizzle_elems = swizzle_mode.bytes() // dtype_size
        self.padded_depth = align_up(depth, swizzle_elems)

        var smem_use = 4  # tmem
        if self.split_m:
            self.BN = min(
                256, align_down(Self.sm100_tmem_cols - depth, Self.MMA_K)
            )
            self.TMEM_P0 = Self.TMEM_S0
            self.TMEM_O0 = Self.TMEM_S0 + self.BN
            self.TMEM_C0 = Self.TMEM_S0 + self.BN // 2

            self.TMEM_S1 = Self.TMEM_S0 + 16 << 16
            self.TMEM_P1 = self.TMEM_P0 + 16 << 16
            self.TMEM_O1 = self.TMEM_O0 + 16 << 16
            self.TMEM_C1 = self.TMEM_C0 + 16 << 16
            self.tmem_used = self.TMEM_O1 + depth
        else:
            # we use two q and o
            # determine BN via tmem:
            # 2*BN + 2*depth <= 512 -> BN + depth <= 256
            self.BN = min(
                256,
                align_down(
                    (Self.sm100_tmem_cols // 2 - self.padded_depth), Self.MMA_K
                ),
            )
            self.TMEM_S1 = Self.TMEM_S0 + self.BN
            self.TMEM_P0 = Self.TMEM_S0
            self.TMEM_P1 = self.TMEM_S1
            self.TMEM_C0 = self.TMEM_P0 + self.BN // 2
            self.TMEM_C1 = self.TMEM_P1 + self.BN // 2
            self.TMEM_O0 = self.TMEM_S1 + self.BN
            self.TMEM_O1 = self.TMEM_O0 + self.padded_depth
            self.tmem_used = self.TMEM_O1 + self.padded_depth

        # We have the following resources that need smem barriers:
        # KV: num_kv_stages
        # S: 2
        # C: 2
        # O: 2
        # softmax order: 2
        # q: 1, for Q1 synchronization
        # 4 for `o_pipeline` (2 consumer + 2 producer)
        smem_use += (FA4MiscMBars.size + 4) * Self.mbar_size

        # We use the gcd here to ensure that both
        # depth//swizzle_elems and BN//MMA_K
        # can be evenly divided by the mma stages
        # TODO: Allow setting num_mma_stages > 1 and benchmark
        # self.num_mma_stages = gcd(
        #     self.padded_depth // swizzle_elems, self.BN // Self.MMA_K
        # )
        self.num_mma_stages = 1
        self.BK0 = self.padded_depth // self.num_mma_stages
        self.BK1 = self.BN // self.num_mma_stages
        # smem use is (NOTE: smem uses padded depth):
        # BM*depth*dtype_size + num_kv_stages*(2*mbar_size + BN*depth*dtype_size) <= smem_remaining
        # num_kv_stages <= (smem_remaining - 2*BM*depth*dtype_size) // (2*mbar_size + BN*depth*dtype_size)
        smem_use += self.BM * self.padded_depth * dtype_size
        smem_per_kv = (
            self.BN * self.padded_depth * dtype_size
            + 2 * Self.mbar_size * self.num_mma_stages
        )
        self.num_kv_stages = (
            Self.sm100_smem_carveout - smem_use
        ) // smem_per_kv
        # example values of (num_kv_stages * num_mma_stages)
        # depth= 64: (8 * 1) =  8
        # depth= 80: (3 * 2) =  6
        # depth=128: (5 * 2) = 10
        # depth=256: (1 * 4) =  4
        # The product gives the total number of stages
        smem_use += self.num_kv_stages * smem_per_kv
        self.smem_used = smem_use

    fn supported(self) -> Bool:
        return (
            self.depth >= 64
            and self.BN >= 64
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )

    fn use_tmem_for_correction(self) -> Bool:
        # if Self.TMEM_S0 == self.TMEM_P0, then we can place the correction
        # starting at `self.TMEM_P0 + self.BN//2`.
        # Otherwise, we check if we can place it after `O1`.
        return (Self.TMEM_S0 == self.TMEM_P0) or (
            (self.TMEM_O1 + self.depth + Self.num_correction_cols) <= 512
        )

    fn correction_smem_elements(self) -> Int:
        return (
            0 if self.use_tmem_for_correction() else self.BM
            * Self.num_correction_cols
        )

    fn num_active_warps_per_group(self) -> Int:
        return 4

    fn num_active_threads_per_group(self) -> Int:
        return WARP_SIZE * self.num_active_warps_per_group()


fn maximum[
    dtype: DType, BN: Int, //, *, width: Int = 8
](x: LocalTensor[dtype, Layout.row_major(BN)]) -> SIMD[dtype, width]:
    constrained[BN % width == 0]()
    vx = x.vectorize[width]()
    acc = vx[0]

    # unroll (using SIMD) to break up dependency chain
    @parameter
    for i in range(1, BN // width):
        acc = max(acc, vx[i])
    # return acc.reduce_max()
    return rebind[SIMD[dtype, width]](acc)


fn maximum[
    dtype: DType,
    BN: Int,
    width: Int, //,
](
    x: LocalTensor[dtype, Layout.row_major(BN)], init: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    constrained[BN % width == 0]()
    vx = x.vectorize[width]()
    acc = rebind[vx.element_type](init)

    # unroll (using SIMD) to break up dependency chain
    @parameter
    for i in range(BN // width):
        acc = max(acc, vx[i])
    return rebind[SIMD[dtype, width]](acc)


fn sum[
    dtype: DType, BN: Int, //, *, width: Int = 8
](x: LocalTensor[dtype, Layout.row_major(BN)]) -> SIMD[dtype, 2]:
    constrained[BN % width == 0]()
    vx = x.vectorize[width]()
    acc = vx[0]

    # unroll (using SIMD) to break up dependency chain
    @parameter
    for i in range(1, BN // width):
        acc += vx[i]

    return acc.reduce_add[size_out=2]()
    # return rebind[SIMD[dtype,width]](acc)


@always_inline
fn mha_sm100_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme, //,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
](
    output: UnsafePointer[Scalar[output_type]],
    q_arg: UnsafePointer[Scalar[q_type]],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask: MaskType,
    score_mod: ScoreModType,
    valid_length: UnsafePointer[UInt32],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin
        ]
    ],
    batch_size_arg: Int,
    partition: PartitionType,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[q_type, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin]
    ],
) raises:
    constrained[
        config.dtype == KVType.dtype and config.dtype == q_type,
        "config, kv, and q types must all match for FA3.",
    ]()
    alias decoding: Bool = _is_decoding[MaxPromptLenType]()
    constrained[not decoding, "this implementation does not support decoding"]()
    alias fa4_config = FA4Config(
        num_q_heads=Int(config.num_heads),
        group=group,
        depth=Int(config.depth),
        dtype_size=q_type.size_of(),
        swizzle_mode=config.swizzle_mode,
    )
    alias swizzle_mode = fa4_config.swizzle_mode
    alias BM = fa4_config.BM
    alias BK = fa4_config.padded_depth
    constrained[
        BK % 64 == 0,
        "B200 requires BK%64 as it uses 128B swizzles, but BK==",
        String(BK),
    ]()
    alias BN = fa4_config.BN
    alias num_threads = fa4_config.num_threads
    q = rebind[UnsafePointer[Scalar[KVType.dtype]]](q_arg)

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)
    var max_prompt_len: UInt32 = max_prompt_len_arg.as_uint32()
    var max_num_prompt_tiles: UInt32 = ceildiv(max_prompt_len, BM)
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()

    q_tma_op = q_out_tma[
        swizzle_mode,
        BM = BM // 2,
        depth = fa4_config.depth,
        padded_depth = fa4_config.BK0,
        q_num_heads = fa4_config.num_q_heads,
        group = fa4_config.group,
        decoding=False,
    ](ctx, q, num_rows_q)
    k_tma_op = k.create_tma_tile[
        BN, fa4_config.BK0, swizzle_mode, is_k_major=True
    ](ctx)
    v_tma_op = v.create_tma_tile[
        fa4_config.BK1, fa4_config.padded_depth, swizzle_mode, is_k_major=False
    ](ctx)
    constrained[BM == 256]()
    alias SchedulerType = TransientScheduler[BM, fa4_config.num_q_heads]
    var scheduler: SchedulerType = SchedulerType()

    @parameter
    if sink:
        alias SinkType = NonNullPointer[KVType.dtype]
        var sink_ptr: SinkType = {
            rebind[UnsafePointer[Scalar[KVType.dtype]]](
                sink_weights.value().ptr
            )
        }
        _mha_sm100_kv_input_row_offset_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVType,
            output_type=output_type,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=fa4_config,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            output,
            k,
            scale,
            batch_size,
            max_prompt_len_arg,
            max_cache_valid_length,
            valid_length,
            kv_input_row_offsets,
            sink_ptr,
            partition,
            mask,
            score_mod,
            ctx,
        )
    else:
        alias SinkType = NullPointer[KVType.dtype]
        alias sink_ptr: SinkType = {}
        _mha_sm100_kv_input_row_offset_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVType,
            output_type=output_type,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=fa4_config,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            output,
            k,
            scale,
            batch_size,
            max_prompt_len_arg,
            max_cache_valid_length,
            valid_length,
            kv_input_row_offsets,
            sink_ptr,
            partition,
            mask,
            score_mod,
            ctx,
        )


@always_inline
fn _mha_sm100_kv_input_row_offset_dispatch[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ragged: Bool,
    SinkType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = config.BM // 2,
        depth = config.BK0,
        group = config.group,
        decoding=False,
    ],
    k_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BN,
        config.BK0,
        swizzle_mode,
        is_k_major=True,
    ],
    v_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BK1,
        config.padded_depth,
        swizzle_mode,
        is_k_major=False,
    ],
    o_ptr_arg: UnsafePointer[Scalar[output_type]],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: UnsafePointer[UInt32],
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[
            DType.uint32, Layout.row_major(UNKNOWN_VALUE), MutableAnyOrigin
        ]
    ],
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
) raises:
    alias KVRowOffsetsNonNull = NonNullPointer[DType.uint32]
    alias KVRowOffsetsNull = NullPointer[DType.uint32]
    if kv_input_row_offsets:
        var kv_row_offsets: KVRowOffsetsNonNull = {
            kv_input_row_offsets.value().ptr
        }
        _mha_sm100_valid_length_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=config,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsType=KVRowOffsetsNonNull,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_row_offsets,
            sink_weights,
            partition,
            mask,
            score_mod,
            ctx,
        )
    else:
        var kv_row_offsets: KVRowOffsetsNull = {}
        _mha_sm100_valid_length_dispatch[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=config,
            use_score_mod=use_score_mod,
            ragged=ragged,
            SinkType=SinkType,
            KVRowOffsetsType=KVRowOffsetsNull,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_row_offsets,
            sink_weights,
            partition,
            mask,
            score_mod,
            ctx,
        )


@always_inline
fn _mha_sm100_valid_length_dispatch[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ragged: Bool,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = config.BM // 2,
        depth = config.BK0,
        group = config.group,
        decoding=False,
    ],
    k_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BN,
        config.BK0,
        swizzle_mode,
        is_k_major=True,
    ],
    v_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BK1,
        config.padded_depth,
        swizzle_mode,
        is_k_major=False,
    ],
    o_ptr_arg: UnsafePointer[Scalar[output_type]],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: UnsafePointer[UInt32],
    kv_input_row_offsets: KVRowOffsetsType,
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
) raises:
    @parameter
    if ragged:
        alias ValidLengthType = NonNullPointer[DType.uint32]
        var valid_len: ValidLengthType = {valid_length}
        _mha_sm100_enqueue[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=config,
            use_score_mod=use_score_mod,
            SinkType=SinkType,
            ValidLengthType=ValidLengthType,
            KVRowOffsetsType=KVRowOffsetsType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_len,
            kv_input_row_offsets,
            sink_weights,
            partition,
            mask,
            score_mod,
            ctx,
        )
    else:
        alias ValidLengthType = NullPointer[DType.uint32]
        var valid_len: ValidLengthType = {}
        _mha_sm100_enqueue[
            SchedulerType=SchedulerType,
            KVLUTType=KVLUTType,
            output_type=output_type,
            MaxSeqLenType=MaxSeqLenType,
            PartitionType=PartitionType,
            MaskType=MaskType,
            ScoreModType=ScoreModType,
            config=config,
            use_score_mod=use_score_mod,
            SinkType=SinkType,
            ValidLengthType=ValidLengthType,
            KVRowOffsetsType=KVRowOffsetsType,
            _is_cache_length_accurate=_is_cache_length_accurate,
            swizzle_mode=swizzle_mode,
        ](
            scheduler,
            q_tma_op,
            k_tma_op,
            v_tma_op,
            o_ptr_arg,
            kv_lut,
            scale,
            batch_size,
            max_seq_len,
            num_keys_arg,
            valid_len,
            kv_input_row_offsets,
            sink_weights,
            partition,
            mask,
            score_mod,
            ctx,
        )


@always_inline
fn _mha_sm100_enqueue[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        swizzle_mode,
        BM = config.BM // 2,
        depth = config.BK0,
        group = config.group,
        decoding=False,
    ],
    k_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BN,
        config.BK0,
        swizzle_mode,
        is_k_major=True,
    ],
    v_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.BK1,
        config.padded_depth,
        swizzle_mode,
        is_k_major=False,
    ],
    o_ptr_arg: UnsafePointer[Scalar[output_type]],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: ValidLengthType,  # OptionalPointer[DType.uint32]
    kv_input_row_offsets: KVRowOffsetsType,  # OptionalPointer[DType.uint32],
    sink_weights: SinkType,
    partition: PartitionType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
) raises:
    # the pack contains all possibly 0-sized objects
    alias PackType = Pack[
        MaskType,
        ScoreModType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxSeqLenType,
        PartitionType,
    ]
    var pack: PackType = {
        mask,
        score_mod,
        scheduler,
        valid_length,
        sink_weights,
        kv_input_row_offsets,
        max_seq_len,
        partition,
    }

    var max_num_prompt_tiles: UInt32 = ceildiv(
        max_seq_len.as_uint32(), config.BM
    )
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()
    alias num_threads = config.num_threads
    alias smem_use = config.smem_used
    # print(
    #     "smem_used =",
    #     smem_use,
    #     "\nnum_threads =",
    #     num_threads,
    #     "\nbatch_size =",
    #     batch_size,
    #     "\nblock_x =",
    #     block_x,
    # )
    gd = SchedulerType.grid_dim(batch_size, block_x)
    # print("grid_dim =", gd[0], gd[1], gd[2])
    alias kernel = SM100MHA2Q[
        KVLUTType,
        output_type,
        MaskType,
        ScoreModType,
        SchedulerType,
        config,
        use_score_mod,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        _is_cache_length_accurate,
        MaxSeqLenType,
        PartitionType,
    ].kernel
    ctx.enqueue_function[kernel](
        # ctx.enqueue_function[kernel, dump_llvm=Path("pr3.ll")](
        q_tma_op,
        k_tma_op,
        v_tma_op,
        o_ptr_arg,
        kv_lut,
        scale,
        batch_size,
        num_keys_arg,
        pack,
        grid_dim=SchedulerType.grid_dim(batch_size, block_x),
        block_dim=(Int(num_threads), 1, 1),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
        # func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
        #     B200.shared_memory_per_multiprocessor - 1024
        # ),
    )


@register_passable("trivial")
struct KVPipeline[num_kv_stages: Int, num_mma_stages: Int]:
    """
    KVPipeline has `num_kv_stages * num_mma_stages` stages.
    `num_kv_stages` refers to how many `K` and `V` tiles we pipeline
    for performing the `S = Q@K'` and `O += P@V` MMAs.
    Each of these MMAs is broken up into `num_mma_stages` pipelined
    MMAs. We set `step=False` for all but the last MMA that completes
    the operation.
    An alternative implementation would separate the two, and potentially
    allow for more overall stages at the cost of slightly more bookkeeping.
    """

    alias num_stages: Int = num_kv_stages * num_mma_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[num_kv_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    fn init(self):
        # Producer mbars: arrived by 1 thread performing TMA
        @parameter
        for i in range(Self.num_stages):
            self.mbar[i].init(1)

        # Consumer mbars: arrived on by the MMA warp
        @parameter
        for i in range(Self.num_stages, 2 * Self.num_stages):
            self.mbar[i].init(WARP_SIZE)

    @always_inline
    fn producer_mbar[mma_stage: Int](self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.mbar + num_mma_stages * idx + mma_stage

    @always_inline
    fn consumer_mbar[mma_stage: Int](self, idx: UInt32) -> MBarType:
        alias const_offset = mma_stage + Self.num_stages
        return self.mbar + num_mma_stages * idx + const_offset

    @always_inline
    fn consumer_mbar[mma_stage: Int](self) -> MBarType:
        return self.consumer_mbar[mma_stage](self.state.index())

    @always_inline("nodebug")
    fn producer_acquire[mma_stage: Int = num_mma_stages - 1](self):
        """
        Returns the dynamic pipe idx.
        """
        self.consumer_mbar[mma_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_wait[mma_stage: Int = num_mma_stages - 1](self):
        self.producer_mbar[mma_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_release[mma_stage: Int = num_mma_stages - 1](mut self):
        _ = self.consumer_mbar[mma_stage]()[].arrive()

        @parameter
        if mma_stage == num_mma_stages - 1:
            self.state.step()

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return 2 * num_mma_stages * num_kv_stages


@register_passable("trivial")
struct TMADestination[dtype: DType, layout: Layout]:
    var mbar: MBarType
    var smem: SharedMemTensor[dtype, layout]

    @always_inline
    fn __init__(out self, mbar: MBarType, smem: SharedMemTensor[dtype, layout]):
        self.mbar = mbar
        self.smem = smem


@register_passable("trivial")
struct KVProducerPipeline[dtype: DType, config: FA4Config]:
    alias KType = SharedMemTensor[
        dtype,
        tile_layout_k_major[
            dtype, config.BN, config.BK0, config.swizzle_mode
        ](),
    ]
    alias VType = SharedMemTensor[
        dtype,
        tile_layout_mn_major[
            dtype,
            config.padded_depth,
            config.BK1,
            config.swizzle_mode,
        ](),
    ]
    alias KPairType = TMADestination[dtype, Self.KType.layout]
    alias VPairType = TMADestination[dtype, Self.VType.layout]
    alias kv_elements = Self.KType.layout.size()
    alias kv_bytes = Self.kv_elements * dtype.size_of()
    alias SMemType = SharedMemPointer[Scalar[dtype]]

    var kv_pipeline: KVPipeline[config.num_kv_stages, config.num_mma_stages]
    var smem: Self.SMemType

    @always_inline
    fn __init__(
        out self,
        mbar: MBarType,
        smem: Self.SMemType,
    ):
        constrained[config.padded_depth % config.num_mma_stages == 0]()
        constrained[config.BN % config.num_mma_stages == 0]()
        constrained[Self.kv_elements == Self.VType.layout.size()]()
        self.kv_pipeline = {mbar}
        self.smem = smem
        self.kv_pipeline.state._phase = 1

    @always_inline
    fn __init__(
        out self,
        kv_pipeline: KVPipeline[config.num_kv_stages, config.num_mma_stages],
        smem: Self.SMemType,
    ):
        constrained[config.padded_depth % config.num_mma_stages == 0]()
        constrained[config.BN % config.num_mma_stages == 0]()
        constrained[Self.kv_elements == Self.VType.layout.size()]()
        self.kv_pipeline = kv_pipeline
        self.smem = smem
        self.kv_pipeline.state._phase = 1

    @always_inline
    fn init(self):
        """
        Only one of the producer or consumer should call `init()`.
        """
        self.kv_pipeline.init()

    @always_inline
    fn get_kv_smem[*, mma_stage: Int](self) -> Self.SMemType:
        alias stage_offset = mma_stage * config.padded_depth * config.BN
        var dyn_offset: UInt32 = (
            Self.kv_elements * self.kv_pipeline.state.index()
        )
        return self.smem + stage_offset + dyn_offset

    @always_inline
    fn get_k[*, mma_stage: Int, expect: Bool = True](self) -> Self.KPairType:
        p_mbar = self.kv_pipeline.producer_mbar[mma_stage=mma_stage]()

        @parameter
        if expect:
            p_mbar[].expect_bytes(Self.kv_bytes)
        return {p_mbar, {self.get_kv_smem[mma_stage=mma_stage]()}}

    @always_inline
    fn get_v[*, mma_stage: Int](self) -> Self.VPairType:
        p_mbar = self.kv_pipeline.producer_mbar[mma_stage=mma_stage]()
        p_mbar[].expect_bytes(Self.kv_bytes)
        return {p_mbar, {self.get_kv_smem[mma_stage=mma_stage]()}}

    @always_inline
    fn acquire_kv[*, mma_stage: Int = config.num_mma_stages - 1](self):
        self.kv_pipeline.producer_acquire[mma_stage]()

    @always_inline
    fn commit_kv_step(mut self):
        """
        Step the kv pipeline. The does not perform the commit on the mbars;
        that should be handled by the `tma_op.async_copy`.
        """
        self.kv_pipeline.state.step()


@register_passable("trivial")
struct KVConsumerPipeline[dtype: DType, config: FA4Config]:
    """
    Pipeline for managing the consumption of K and V.
    This follows the order of Tri Dao and Cutlass implementations
    (modulo any rotation of the ops through the iterations).

    We consume/produce in the following order:
        0. S0 <- Q0 @ Kn'
        1. O1 <- O1 + P1 @ V{n-1}
        2. S1 <- Q1 @ Kn'
        3. O0 <- O0 + P0 @ Vn

    Note that we have two MMA between calculating Si and consuming Pi,
    maximizing the overlap between MMAs and softmax calculation.
    Oi + Pi @ V also depends on the correction, which is computed
    asynchronously with the softmax in a correction warpgroup (as soon
    as the softmax writes the correction factor).

    # wait on K0
    S0 <- Q0 @ K0'
    S1 <- Q1 @ K0'
    # release K0
    # wait on V0
    O0 <- P0 @ V0
    for n in range(1,num_iters):
        # wait on Kn
        S0 <- Q0 @ Kn'
        O1 <- O1 + P1@V{n-1}
        # release V{n-1}
        S1 <- Q1 @ Kn'
        # release Kn
        # wait on Vn
        O0 <- P0 @ Vn
    O1 <- O1 + P1@V{num_iters-1}

    wK0, rK0, wV0
    wK1, rV0, rK1, wV1
    wK2, rV1, rK2, wV2
    wK3, rV2, rK3, wV3

    wKn(state)
    wK0(0), rK0(0), wV0(1)
    wK1(2), rV0(1), rK1(2), wV1(3)
    wK2(4), rV1(3), rK2(4), wV2(5)
    wK3(6), rV2(5), rK3(6), wV3(7)

    Rules:
        wK backs up and increments prior to waiting, except K0
        rK increments after releasing
        rV uses backup

    wK0(0), rK0(0), wV0(1)
    wK1(2), rV0(1), rK1(2), wV1(3)
    wK2(4), rV1(3), rK2(4), wV2(5)
    rV2(5)
    """

    alias full_kv_bytes = config.BN * config.padded_depth * dtype.size_of()
    alias mma_kv_bytes = config.BN * config.BK0 * dtype.size_of()

    var kv_pipeline: KVPipeline[config.num_kv_stages, config.num_mma_stages]
    var k_smem_descriptor: MMASmemDescriptor
    var v_smem_descriptor: MMASmemDescriptor
    var v_pipeline_release_index: UInt32

    @always_inline
    fn __init__(
        out self,
        kv_pipeline: KVPipeline[config.num_kv_stages, config.num_mma_stages],
        smem: SharedMemPointer[Scalar[dtype]],
    ):
        self.kv_pipeline = kv_pipeline
        self.k_smem_descriptor = smem_descriptor[
            BMN = config.BN,
            BK = config.BK0,
            swizzle_mode = config.swizzle_mode,
            is_k_major=True,
        ](smem)
        self.v_smem_descriptor = smem_descriptor[
            BMN = config.padded_depth,
            BK = config.BK1,
            swizzle_mode = config.swizzle_mode,
            is_k_major=False,
        ](smem)
        self.v_pipeline_release_index = 0

    @always_inline
    fn __init__(
        out self,
        mbar: MBarType,
        smem: SharedMemPointer[Scalar[dtype]],
    ):
        return self.__init__(
            KVPipeline[config.num_kv_stages, config.num_mma_stages](mbar), smem
        )

    @always_inline
    fn init(self):
        """
        Only one of the producer or consumer should call `init()`.
        """
        self.kv_pipeline.init()

    @always_inline("nodebug")
    fn wait[*, mma_stage: Int](self) -> UInt32:
        """
        Wait on `k` from the producer, and return the `k` smem descriptor.
        """

        alias stage_offset = mma_stage * Self.mma_kv_bytes
        var dyn_offset: UInt32 = (
            Self.full_kv_bytes * self.kv_pipeline.state.index()
        )
        self.kv_pipeline.consumer_wait[mma_stage]()
        return dyn_offset + stage_offset

    @always_inline("nodebug")
    fn wait_k[
        *,
        mma_stage: Int = config.num_mma_stages - 1,
        pre_increment: Bool = True,
    ](mut self) -> MMASmemDescriptor:
        """
        Wait on `k` from the producer, and return the `k` smem descriptor.
        If `pre-increment` is true.
        """

        @parameter
        if pre_increment and (mma_stage == 0):
            self.v_pipeline_release_index = self.kv_pipeline.state.index()
            self.kv_pipeline.state.step()
        return self.k_smem_descriptor + Int(self.wait[mma_stage=mma_stage]())

    @always_inline("nodebug")
    fn wait_v[
        *, mma_stage: Int = config.num_mma_stages - 1
    ](self) -> MMASmemDescriptor:
        return self.v_smem_descriptor + Int(self.wait[mma_stage=mma_stage]())

    @always_inline("nodebug")
    fn release_k[*, mma_stage: Int = config.num_mma_stages - 1](mut self):
        """
        Must call `producer_commit` on the tmem resource before calling
        `consumer_release`.
        `release_k` does increment the pipeline step.
        """
        self.kv_pipeline.consumer_release[mma_stage]()

    @always_inline("nodebug")
    fn release_v[*, mma_stage: Int = config.num_mma_stages - 1](self):
        """
        Must call `producer_commit` on the tmem resource before calling
        `consumer_release`.
        `release_v` does not increment the pipeline step.
        """
        _ = self.kv_pipeline.consumer_mbar[mma_stage](
            self.v_pipeline_release_index
        )[].arrive()


@register_passable("trivial")
struct ProducerPipeline[number_of_stages: Int]:
    alias num_stages: Int = number_of_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        # Behavior:
        # mbar - initially phase 0
        # Producer - phase 1
        # Consumer - phase 0
        #
        # A `wait(phase)` blocks so long as
        # `mbar.phase != phase`.
        # Memory barriers are initialized with `mbar.phase = 0`.
        # Memory barrier phases flip after init-count arrivals.
        # Example with `num_stages = 1`.
        #
        # Producer:
        # p0. consumer_mbar.wait(phase=1)  # 1 != 0: falls through
        # p1. producer_mbar.commit()       # producer_mbar.phase=1
        # p2. step()                       # phase = 0
        # p3. consumer_mbar.wait(phase=0)  # 0 == 0: blocked until c1
        # p4. producer_mbar.commit()       # producer_mbar.phase=0
        # p5. step()
        # p6. consumer_mbar.wait(phase=1)
        # p7. producer_mbar.commit()       # producer_mbar.phase=1
        #
        # Consumer:
        # c0. producer_mbar.wait(phase=0)  # 0 == 0: blocked until p1
        # c1. consumer.release()           # consumer_mbar.phase=1
        # c2. step()                       # phase = 1
        # c3. producer_mbar.wait(phase=1)  # blocked until p4
        # c4. consumer.release()           # consumer_mbar.phase=0
        # c5. step()
        # c6. producer_mbar.wait(phase=0)
        # c7. consumer.release()           # consumer_mbar.phase=1
        #
        # The order of blocking/unblocking can be visualized as:
        # p0, p1, p2
        #     \-> c0, c1, c2
        #              \-> p3, p4, p5
        #                       \-> c3, c4, c5
        #                                \-> p6, p7
        #                                         \-> c6, c7
        #
        # The producer initializes phase to `1`
        # Thus, initial producer `wait`s fall through; only after
        # `number_of_stages` steps will we reset to `phase = 0`,
        # and thus begin waiting on the first set of `consumer` releases.
        self.state = {}  # {0, 1, 0}
        self.state._phase = 1

    @always_inline
    fn producer_mbar(self) -> MBarType:
        return self.mbar + self.state.index()

    @always_inline
    fn consumer_mbar(self) -> MBarType:
        return self.mbar + number_of_stages + self.state.index()

    @always_inline("nodebug")
    fn acquire(self):
        self.consumer_mbar()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn commit(mut self):
        _ = self.producer_mbar()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    fn commit_mma(self):
        mbar = self.producer_mbar()
        if elect_one_sync():
            mma_arrive(mbar)

    @always_inline("nodebug")
    fn step(mut self):
        self.state.step()


@register_passable("trivial")
struct ConsumerPipeline[number_of_stages: Int]:
    alias num_stages: Int = number_of_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}
        # Consumer phase is initialized to `0`.
        # Producer phase is initialized to `1`.
        # See `ProducerPipeline.__init__` for details.

    @always_inline
    fn producer_mbar(self) -> MBarType:
        return self.mbar + self.state.index()

    @always_inline
    fn consumer_mbar(self) -> MBarType:
        return self.mbar + number_of_stages + self.state.index()

    @always_inline("nodebug")
    fn wait(self):
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn release(mut self):
        _ = self.consumer_mbar()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    fn step(mut self):
        self.state.step()


@register_passable("trivial")
struct MBarPipeline[number_of_stages: Int]:
    alias num_stages: Int = number_of_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    fn init[*, num_producer: UInt32 = 1, num_consumer: UInt32 = 1](self):
        @parameter
        for i in range(number_of_stages):
            self.mbar[i].init(Int(num_producer))

        @parameter
        for i in range(number_of_stages):
            self.mbar[i + number_of_stages].init(Int(num_consumer))

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return 2 * number_of_stages


@always_inline
fn apply_mask[
    dtype: DType,
    BN: Int,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait, //,
    *,
    use_score_mod: Bool,
    decoding: Bool = False,
](
    srow: LocalTensor[dtype, Layout.row_major(BN)],
    mask: MaskType,
    score_mod: ScoreModType,
    scale_log2e: Scalar[dtype],
    mask_status: TileMaskStatus,  # encoding-only
    *,
    prompt_idx: UInt32,
    q_head_idx: UInt32,
    kv_tile_start_row: UInt32,
    seq_len: UInt32,
    max_seq_len: UInt32,
    num_keys: UInt32,
    row: UInt32,  # encoding-only
    start_pos: UInt32,  # encoding-only
):
    alias simd_size = simd_width_of[dtype]()
    vs = srow.vectorize[simd_size]()
    var score_row: UInt32 = row
    var score_row_with_start_pos: UInt32 = score_row + start_pos

    @parameter
    @always_inline
    fn _apply_mask_capture[masked: Bool]():
        @parameter
        for n in range(BN // simd_size):
            # score_col = mask_frag_col + j * 8
            s = vs[n]
            alias frag_col = simd_size * n
            var score_col: UInt32 = kv_tile_start_row + frag_col

            @parameter
            if masked:
                s = mask.mask(
                    IndexList[4, element_type = DType.uint32](
                        Int(prompt_idx),
                        Int(q_head_idx),
                        Int(score_row_with_start_pos),
                        Int(score_col),
                    ),
                    s * scale_log2e,
                )
            else:  # if MaskType.apply_log2e_after_mask, this is scale only
                s *= scale_log2e

            @parameter
            if use_score_mod:
                s = (
                    score_mod.score_mod(
                        IndexList[4, element_type = DType.uint32](
                            Int(prompt_idx),
                            Int(q_head_idx),
                            Int(score_row_with_start_pos),
                            Int(score_col),
                        ),
                        s,
                        Int(max_seq_len),
                    )
                    * log2e
                )
            elif MaskType.apply_log2e_after_mask:
                s *= log2e

            var bound: IndexList[2, element_type = DType.uint32]

            @parameter
            if decoding:
                bound = IndexList[2, element_type = DType.uint32](
                    Int(num_keys),
                    Int(
                        min(
                            BN + kv_tile_start_row,
                            num_keys,
                        )
                    ),
                )
                s = _kernel_mask(
                    IndexList[2, element_type = DType.uint32](
                        Int(score_row), Int(score_col)
                    ),
                    bound,
                    s,
                )
            elif masked:
                bound = IndexList[2, element_type = DType.uint32](
                    Int(seq_len),
                    Int(num_keys),
                )
                s = _kernel_mask(
                    IndexList[2, element_type = DType.uint32](
                        Int(score_row), Int(score_col)
                    ),
                    bound,
                    s,
                )
            vs[n] = s

    unswitch[_apply_mask_capture](
        (mask_status == TileMaskStatus.PARTIAL_MASK)
        # NOTE: mask_status should be either PARTIAL_MASK or NO_MASK at
        # this point.
        # In the NO_MASK case, we still need to mask out the scores for the
        # last tile, which goes beyond num_keys (for num_keys % 128 != 0).
        or (BN + kv_tile_start_row > num_keys)
    )


@always_inline
fn scale_write_output[
    BM: Int,
    BN: Int,
    depth: Int,
    padded_depth: Int,
    q_num_heads: Int,
    group: Int,
    decoding: Bool,
    accum_type: DType,
    output_type: DType, //,
    config: FA4Config,
](
    local_row: UInt32,
    inv_row_sum: Scalar[accum_type],
    o_ptr_arg: UnsafePointer[Scalar[output_type]],
    o_smem: SharedMemPointer[Scalar[output_type]],
    o_tmem: TMemTile[accum_type, config.BM // 2, config.padded_depth],
    local_warp_group_idx: UInt32,
    position: MHAPosition[
        BM, BN, depth, padded_depth, q_num_heads, group, decoding
    ],
    consumer_mbar: MBarType,
):
    o = o_tmem.load_async_st_matrix[num_threads=WARPGROUP_SIZE]()
    alias num_rows = o.layout[0].size()
    inv_row_sums = LocalTensor[
        accum_type, Layout.row_major(num_rows)
    ].stack_allocation()
    lane = local_row % 32
    lane_row = lane // 4

    #  0  1  2  3
    #  4  5  6  7
    #  8  9 10 11
    # 12 13 14 15
    # 16 17 18 19
    # 20 21 22 23
    # 24 25 26 27
    # 28 29 30 31
    # lane 0 needs to get
    alias tid_to_print: UInt = UInt(env_get_int["TID_PRINT", -1]())

    @parameter
    for i in range(num_rows):
        # lane // 4, lane // 4 + 8, lane // 4 + 16, lane // 4 + 24
        inv_row_sums[i] = warp.shuffle_idx(inv_row_sum, lane_row + 8 * i)

    tcgen05_load_wait()
    tcgen05_fence_before()
    _ = consumer_mbar[].arrive()

    @parameter
    for i in range(num_rows):
        irs = o.element_type(rebind[Scalar[accum_type]](inv_row_sums[i]))

        @parameter
        for j in range(o.layout[1].size()):
            o[i, j] *= irs

    alias swizzle = make_swizzle[
        num_rows=8, row_size = config.padded_depth, access_size=8
    ]()
    alias ST = STMatrixLayout[
        config.BM // 2, config.padded_depth, num_threads=WARPGROUP_SIZE
    ]
    output_gmem_tile = position.split_out_gmem_tensor(
        o_ptr_arg, local_warp_group_idx
    )
    accum_smem_tile = LayoutTensor[
        output_type,
        Layout.row_major(BM // 2, padded_depth),
        address_space = AddressSpace.SHARED,
    ](o_smem)
    var warpy = local_row // 32

    @parameter
    for i in range(2):
        rows_of_o_frags = LocalTensor[
            accum_type,
            layout = Layout.row_major(1, ST.frag_size),
        ](o.ptr + i * ST.frag_size)
        accum_smem_warp_tile = accum_smem_tile.tile[16, config.padded_depth](
            Int(2 * warpy + i), Int(0)
        )

        output_gmem_to_smem_STMatrix[
            BM=16, padded_depth=padded_depth, swizzle=swizzle, num_consumer=1
        ](
            lane,
            local_warp_group_idx=0,
            output_reg_tile=rows_of_o_frags,
            accum_smem_tile=rebind[
                LayoutTensor[
                    output_type,
                    Layout.row_major(16, padded_depth),
                    MutableAnyOrigin,
                    address_space = AddressSpace.SHARED,
                ]
            ](accum_smem_warp_tile),
        )
    named_barrier[WARPGROUP_SIZE](Int32(local_warp_group_idx))
    alias simd_size = simd_width_of[output_type]()
    copy_sram_to_dram[
        thread_layout = Layout.row_major(
            WARPGROUP_SIZE * simd_size // config.depth,
            config.depth // simd_size,
        ),
        swizzle=swizzle,
    ](
        output_gmem_tile.vectorize[1, simd_size](),
        accum_smem_tile.vectorize[1, simd_size](),
    )


@register_passable("trivial")
struct FA4MiscMBars:
    var mbar_base: MBarType
    alias S0_offset = 0
    alias S1_offset = 2
    alias C0_offset = 4
    alias C1_offset = 6
    alias order_offset = 8
    alias Q1SyncIdx = 10
    alias size = Self.Q1SyncIdx + 1

    @always_inline
    fn __init__(out self, mbar_base: MBarType):
        self.mbar_base = mbar_base

    @always_inline
    fn init(self):
        # [0] producer 0
        # [1] consumer 0
        # [2] producer 1
        # [3] consumer 1
        @parameter
        for wg_idx in range(2):
            # S producer, produced by 1 UMMA
            self.mbar_base[2 * wg_idx].init(1)
            # S consumer, consumed by 128 softmax threads
            self.mbar_base[2 * wg_idx + 1].init(128)
            # C producer, produced by 128 softmax threads
            self.mbar_base[2 * wg_idx + Self.C0_offset].init(128)
            # C consumer, consumed by 128 correction threads
            self.mbar_base[2 * wg_idx + 1 + Self.C0_offset].init(128)
            # ordering is done by 128 softmax threads
            self.mbar_base[wg_idx + Self.order_offset].init(128)

        self.mbar_base[Self.Q1SyncIdx].init(1)

    @always_inline
    fn producer_s0(self) -> ProducerPipeline[1]:
        return {self.mbar_base}

    @always_inline
    fn producer_s1(self) -> ProducerPipeline[1]:
        return {self.mbar_base + Self.S1_offset}

    @always_inline
    fn consumer_s(self, wg_idx: UInt32) -> ConsumerPipeline[1]:
        return {self.mbar_base + 2 * wg_idx}

    @always_inline
    fn consumer_c0(self) -> ConsumerPipeline[1]:
        return {self.mbar_base + Self.C0_offset}

    @always_inline
    fn consumer_c1(self) -> ConsumerPipeline[1]:
        return {self.mbar_base + Self.C1_offset}

    @always_inline
    fn producer_c(self, wg_idx: UInt32) -> ProducerPipeline[1]:
        return {self.mbar_base + Self.C0_offset + 2 * wg_idx}

    @always_inline
    fn pipeline_order_wait(self, wg_idx: UInt32) -> MBarType:
        return {self.mbar_base + Self.order_offset + wg_idx}

    @always_inline
    fn pipeline_order_arrive(self, wg_idx: UInt32) -> MBarType:
        return {self.mbar_base + (Self.order_offset + 1) - wg_idx}

    @always_inline
    fn q1_wait_mbar(
        self,
    ) -> ref [
        self.mbar_base.origin, self.mbar_base.address_space
    ] SharedMemBarrier:
        return self.mbar_base[Self.Q1SyncIdx]

    @always_inline
    fn end(self) -> MBarType:
        return self.mbar_base + Self.size


@register_passable("trivial")
struct SM100MHA2Q[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
]:
    alias qkv_type = KVLUTType.dtype
    alias accum_type = get_accum_type[Self.qkv_type]()
    alias simd_size: Int = simd_width_of[Self.qkv_type]()

    alias cta_group = 1  # TODO: support 2
    alias BM = config.BM
    alias BN = config.BN
    alias depth = config.depth
    alias padded_depth = config.padded_depth
    alias num_q_heads = config.num_q_heads
    alias group = config.group
    alias ragged = not ValidLengthType.is_null

    alias num_m_mmas = 2
    alias MMA_M = config.BM // Self.num_m_mmas
    alias qo_elements = Self.padded_depth * Self.MMA_M
    alias qkv_dt_size = Self.qkv_type.size_of()

    alias OPipelineType = MBarPipeline[2]  # x1 -> 4 barriers

    alias num_mma_stages = config.num_mma_stages

    # First MMA is
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    alias UMMA0Type = SM100TensorAccumulatorSS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,  # generally 128
        MMA_N = Self.BN,
        BK = Self.depth,  # BK in memory depth
        swizzle_a = config.swizzle_mode,
        swizzle_b = config.swizzle_mode,
        transpose_b=True,
        num_stages = Self.num_mma_stages,
    ]
    # Second MMA is
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    alias UMMA1Type = SM100TensorAccumulatorTS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,
        MMA_N = config.padded_depth,
        BK = Self.BN,
        swizzle_b = config.swizzle_mode,
        transpose_b=False,
        num_stages = Self.num_mma_stages,
    ]

    alias swizzle_granularity = config.swizzle_mode.bytes() // Self.qkv_dt_size
    alias k_elements: UInt32 = Self.swizzle_granularity * config.BN
    alias qo_bytes: UInt32 = Self.qkv_dt_size * Self.qo_elements
    alias k_bytes: UInt32 = Self.qkv_dt_size * Self.k_elements
    alias MMA_K = 16
    alias v_bytes_per_mma: UInt32 = Self.qkv_dt_size * Self.MMA_K * config.padded_depth

    alias KVPipelineType = KVPipeline[
        config.num_kv_stages, config.num_mma_stages
    ]
    alias PositionType = MHAPosition[
        config.BM,
        config.BN,
        config.depth,
        config.padded_depth,
        config.num_q_heads,
        config.group,
        _is_decoding[MaxSeqLenType](),
    ]

    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads)
    )
    fn kernel(
        q_tma_op: QTMATile[
            KVLUTType.dtype,
            config.swizzle_mode,
            BM = config.BM // 2,
            depth = config.BK0,
            group = config.group,
            decoding=False,
        ],
        k_tma_op: TMANestedTensorTile[
            KVLUTType.dtype,
            config.BN,
            config.padded_depth,
            config.swizzle_mode,
            is_k_major=True,
        ],
        v_tma_op: TMANestedTensorTile[
            KVLUTType.dtype,
            config.BN,
            config.padded_depth,
            config.swizzle_mode,
            is_k_major=False,
        ],
        o_ptr_arg: UnsafePointer[Scalar[output_type]],
        kv_lut: KVLUTType,
        scale: Float32,
        batch_size: UInt32,
        num_keys_arg: UInt32,
        pack: Pack[
            MaskType,
            ScoreModType,
            SchedulerType,
            ValidLengthType,
            SinkType,
            KVRowOffsetsType,
            MaxSeqLenType,
            PartitionType,
        ],
    ):
        constrained[Self.MMA_M == 64 or Self.MMA_M == 128]()
        constrained[_is_decoding[MaxSeqLenType]() == False]()
        constrained[
            config.supported(),
            "depth = "
            + String(config.depth)
            + "\nBN = "
            + String(config.BN)
            + "\nnum_kv_stages = "
            + String(config.num_kv_stages)
            + "\ntmem_used = "
            + String(config.tmem_used)
            + "\nsmem_used = "
            + String(config.smem_used),
        ]()
        constrained[
            not SchedulerType.may_advance,
            "Persistent kernels not yet supported with FA4",
        ]()
        constrained[Self.UMMA0Type.num_stages == Self.UMMA1Type.num_stages]()

        mask = pack.mask
        score_mod = pack.score_mod
        scheduler = pack.scheduler
        valid_length = pack.valid_length
        sink_weights = pack.sink_weights
        kv_input_row_offsets = pack.kv_input_row_offsets
        max_seq_len = pack.max_seq_len
        partition = pack.partition

        alias num_qo = config.num_qo()
        # TODO: We may want to support num_qo>2 for depth=64?
        constrained[
            num_qo == 1 or num_qo == 2,
            "Currently only support num_qo == 1 or 2",
        ]()
        q_smem = external_memory[
            Scalar[Self.qkv_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        kv_smem = q_smem + config.BM * config.padded_depth
        alias kv_total_stages = config.num_kv_stages * config.num_mma_stages
        alias kv_smem_total_bytes = config.padded_depth * config.BN * kv_total_stages
        var correction_smem: SharedMemPointer[Scalar[Self.accum_type]] = (
            kv_smem + kv_smem_total_bytes
        ).bitcast[Scalar[Self.accum_type]]()
        var mbar_base: MBarType

        @parameter
        if config.use_tmem_for_correction():
            mbar_base = correction_smem.bitcast[SharedMemBarrier]()
        else:
            mbar_base = (
                correction_smem + config.correction_smem_elements()
            ).bitcast[SharedMemBarrier]()

        kv_pipeline = Self.KVPipelineType(mbar_base)
        mbar_base += Self.KVPipelineType.num_mbars()
        # O += P@V -> correction
        o_mbar = mbar_base  # 2, UMMA
        mbar_base += Self.OPipelineType.num_mbars()
        var misc_mbars: FA4MiscMBars = {mbar_base}
        # S = Q@K' -> softmax 0/1
        # softmax 0/1 -> correction
        # 4s (2 consumer, 2 producer)
        # 4c (2 consumer, 2 producer)
        # 2 softmax-order
        ptr_tmem_addr = misc_mbars.end().bitcast[UInt32]()

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        alias num_reg_softmax = 184
        alias num_reg_correction = 96
        alias num_reg_other = 48

        var tile_summary = MHATileSummary[ValidLengthType](
            batch_size,
            ceildiv(max_seq_len.as_uint32(), config.BM)
            * partition.num_partitions(),
            valid_length,
            max_seq_len.as_uint32(),
        )
        var state: MHATileState = scheduler.initial_state(
            SharedMemPointer[UInt32](),
            tile_summary,
        )
        initial_seq_info = scheduler.unsafe_seq_info(tile_summary, state)

        if not initial_seq_info.is_valid():
            return

        var tid: UInt32 = thread_idx.x

        constrained[
            not PartitionType.do_partition,
            (
                "Neither partitioning nor decoding are supported by the 2-q"
                " implementation."
            ),
        ]()
        if tid < 32:
            tcgen05_alloc[Self.cta_group](ptr_tmem_addr, config.sm100_tmem_cols)
        elif tid < 64 and elect_one_sync():
            kv_pipeline.init()

            # o produced by 1 MMA, consumed by 128 correction
            @parameter
            for i in range(2):
                o_mbar[i].init(1)  # producer
                o_mbar[i + 2].init(WARPGROUP_SIZE)  # consumer
            misc_mbars.init()

        @parameter
        @always_inline
        fn get_position(seq_info: SeqInfo) -> Self.PositionType:
            return _get_position[
                config.BM,
                config.BN,
                config.depth,
                config.padded_depth,
                config.num_q_heads,
                config.group,
                Self.ragged,
                _is_cache_length_accurate,
            ](
                seq_info,
                kv_lut,
                max_seq_len,
                num_keys_arg,
                kv_input_row_offsets,
            )

        var position: Self.PositionType = get_position(initial_seq_info)
        startend = position.get_start_and_end_for_partitions(partition)
        var kv_tile_start_row: UInt32 = startend[0]
        var end: UInt32 = startend[1]

        var warp_group_idx: UInt32 = warp.broadcast(tid // 128)
        barrier()

        # warp group partitioning
        # Two QO:
        if warp_group_idx < 2:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()

            Self.softmax(
                ptr_tmem_addr[0],
                warp_group_idx,
                misc_mbars,
                o_mbar,
                position,
                tid,
                mask,
                kv_tile_start_row,
                end,
                scale.cast[Self.accum_type](),
                score_mod,
                max_seq_len.as_uint32(),
                o_ptr_arg,
                q_smem.bitcast[Scalar[output_type]](),
                sink_weights,
            )

        elif warp_group_idx == 2:
            # correction
            # warpgroup_reg_alloc[num_reg_correction]()
            warpgroup_reg_dealloc[num_reg_correction]()
            Self.correction(
                ptr_tmem_addr[0],
                misc_mbars,
                o_mbar,
                position,
                kv_tile_start_row,
                end,
                mask,
            )
        else:
            warpgroup_reg_dealloc[num_reg_other]()
            warp_id = tid // 32
            if warp_id == 13:  # produce
                Self.load(
                    misc_mbars,
                    kv_pipeline,
                    position,
                    kv_tile_start_row,
                    end,
                    mask,
                    q_tma_op,
                    k_tma_op,
                    v_tma_op,
                    kv_lut,
                    q_smem,
                )

            elif warp_id == 12:  # Q @ K', P @ V
                Self.mma(
                    ptr_tmem_addr[0],
                    misc_mbars,
                    kv_pipeline,
                    o_mbar,
                    position,
                    kv_tile_start_row,
                    end,
                    mask,
                    q_smem,
                )

            # elif tid == 448:  # P @ V
            #     pass
            # elif tid == 480:
            #     pass

    @staticmethod
    @always_inline
    fn softmax(
        tmem_addr: UInt32,
        warp_group_idx: UInt32,
        mbars: FA4MiscMBars,
        o_mbar: MBarType,
        position: Self.PositionType,
        tid: UInt32,
        mask: MaskType,
        kv_tile_start_row: UInt32,
        end: UInt32,
        scale: Scalar[Self.accum_type],
        score_mod: ScoreModType,
        max_seq_len: UInt32,
        o_ptr_arg: UnsafePointer[Scalar[output_type]],
        o_smem: SharedMemPointer[Scalar[output_type]],
        sink_weights: SinkType,
    ):
        # FIXME: for depth 256
        var s_tmem: UInt32 = tmem_addr + config.TMEM_S0

        @parameter
        if config.split_m:
            # split-M: second S is (+16 rows) in st-matrix space
            s_tmem += (16 << 16) * warp_group_idx
        else:
            # 2-Q path: S1 is at +BN columns
            s_tmem += config.BN * warp_group_idx

        p_tmem = s_tmem
        c_tmem = p_tmem + config.BN // 2
        s_tile = Self.UMMA0Type.CType(s_tmem)
        p_tile = Self.UMMA1Type.AType(p_tmem)

        pipeline_s = mbars.consumer_s(warp_group_idx)
        pipeline_c = mbars.producer_c(warp_group_idx)
        # TODO: order_s_wait/arrive
        order_s_wait = mbars.pipeline_order_wait(warp_group_idx)
        order_s_arrive = mbars.pipeline_order_arrive(warp_group_idx)
        var order_phase: UInt32 = 0

        q_head_idx = position.head_idx
        row = tid % 128
        var kv_row: UInt32 = kv_tile_start_row
        # Peel first iter, as there is no need for a correction
        # TODO: add sink
        var mask_status: TileMaskStatus = position.mask_status(mask, kv_row)
        while mask_status == TileMaskStatus.FULL_MASK:
            kv_row += config.BN
            mask_status = position.mask_status(mask, kv_row)
        var scale_log2e: Scalar[Self.accum_type] = (
            scale if use_score_mod
            or MaskType.apply_log2e_after_mask else scale * log2e
        )

        @parameter
        @always_inline
        fn mask_row[
            BN: Int, //,
        ](
            s: LocalTensor[Self.accum_type, Layout.row_major(BN)],
            mask_status: TileMaskStatus,
            kv_gmem_row: UInt32,
        ):
            apply_mask[decoding=False, use_score_mod=use_score_mod](
                s,
                mask,
                score_mod,
                scale_log2e,
                mask_status,
                prompt_idx=position.prompt_idx,
                q_head_idx=q_head_idx,
                kv_tile_start_row=kv_gmem_row,
                seq_len=position.seq_len,
                max_seq_len=max_seq_len,
                num_keys=position.num_keys,
                row=position.prompt_offset + tid,
                start_pos=position.start_pos,
            )

        pipeline_s.wait()
        tcgen05_fence_after()
        s = LocalTensor[
            Self.accum_type, Layout.row_major(config.BN)
        ].stack_allocation()

        @parameter
        @always_inline
        fn load_mask_max(kv_row: UInt32) -> Scalar[Self.accum_type]:
            # break up into sets of 32
            # minimize wait time by using smallest first
            alias BM = config.BM // 2
            alias batch_size = 32
            alias has_remainder = (config.BN % batch_size) != 0
            alias first_cols = (
                config.BN % batch_size
            ) if has_remainder else batch_size
            s0 = TMemTile[Self.accum_type, BM, first_cols](s_tmem).load_async()
            tcgen05_load_wait()
            s1 = TMemTile[Self.accum_type, BM, batch_size](
                s_tmem + first_cols
            ).load_async()
            mask_row(s0, mask_status, kv_row)
            vrow_max = maximum[width = Self.simd_size](s0)

            s.ptr.store(s0.ptr.load[width=first_cols]())
            # i = 0
            # offset0 = first_cols
            # offset1 = first_cols + batch_size
            # offset2 = first_cols + 2*batch_size
            # i = 1
            # offset0 = first_cols + 2*batch_size
            # offset1 = first_cols + 3*batch_size
            # offset2 = first_cols + 4*batch_size
            # i = 2
            # offset0 = first_cols + 4*batch_size
            # offset1 = first_cols + 5*batch_size
            # offset2 = first_cols + 6*batch_size
            alias cols = config.BN - first_cols + batch_size

            # Examples:
            # BN = 80, first_cols = 16, batch_size = 32
            # cols = 64; cols//64 = 1
            # (80-16+32)//64 = 1
            # 80 // 64 = 1
            # offsets = (16, 48, 80)
            #
            # BN = 96, first_cols = 32, batch_size = 32
            # cols = 64; cols//64 = 1
            # (96-32+32)//64 = 1
            # 96 // 64 = 1
            # offsets = (32, 64, 96)
            #
            # BN = 112, first_cols = 16, batch_size = 32
            # cols = 96; cols//64 = 1
            # (112-16+32)//64 = 2
            # 112 // 64 = 1
            # offsets = (16, 48, 80)
            # offsets = (80, 112, 144)
            #
            # BN = 128, first_cols = 32, batch_size = 32
            # cols = 96; cols//64 = 1
            # (128-32+32)//64 = 2
            # 128 // 64 = 2
            # offsets = (32, 64, 96)
            #
            # BN = 144, first_cols = 16, batch_size = 32
            # cols = 128; cols//64 = 2
            # (144-16+32)//64 = 2
            # 144 // 64 = 2
            # offsets = (16, 48, 80)
            # offsets = (80, 112, 144)
            #
            # BN = 160, first_cols = 32, batch_size = 32
            # cols = 128; cols//64 = 2
            # (160-32+32)//64 = 2
            # 160 // 64 = 2
            # offsets = (32, 64, 96)
            # offsets = (96, 128, 160)
            #
            # BN = 176, first_cols = 16, batch_size = 32
            # cols = 160; cols//64 = 2
            # (176-16+32)//64 = 3
            # 176 // 64 = 2
            # offsets = (16, 48, 80)
            # offsets = (80, 112, 144)
            # offsets = (144, 176, 208)
            @parameter
            for i in range(cols // (2 * batch_size)):
                alias offset0 = first_cols + batch_size * (2 * i)
                alias offset1 = first_cols + batch_size * (2 * i + 1)
                alias offset2 = first_cols + batch_size * (2 * i + 2)

                tcgen05_load_wait()

                @parameter
                if offset1 >= config.BN:
                    mask_row(s1, mask_status, kv_row + offset0)
                    vrow_max = maximum(s1, vrow_max)
                    s.ptr.store(offset0, s1.ptr.load[width=batch_size]())
                else:
                    s2 = TMemTile[Self.accum_type, BM, batch_size](
                        s_tmem + offset1
                    ).load_async()
                    mask_row(s1, mask_status, kv_row + offset0)
                    vrow_max = maximum(s1, vrow_max)
                    # if thread_idx.x == 0:
                    #     print("s1 =", s1.vectorize[batch_size]()[0])
                    s.ptr.store(offset0, s1.ptr.load[width=batch_size]())
                    tcgen05_load_wait()

                    @parameter
                    if offset2 < config.BN:
                        s1 = TMemTile[Self.accum_type, BM, batch_size](
                            s_tmem + offset2
                        ).load_async()
                    mask_row(s2, mask_status, kv_row + offset1)
                    vrow_max = maximum(s2, vrow_max)
                    # if thread_idx.x == 0:
                    #     print("s2 =", s2.vectorize[batch_size]()[0])
                    s.ptr.store(offset1, s2.ptr.load[width=batch_size]())

            return vrow_max.reduce_max()

        row_max = load_mask_max(kv_row)
        var sink_weights_ptr = UnsafePointer[Scalar[Self.qkv_type]]()
        var sink_weight: Scalar[Self.accum_type]

        @parameter
        if not SinkType.is_null:
            sink_weights_ptr = rebind[UnsafePointer[Scalar[Self.qkv_type]]](
                sink_weights.value()
            )
            var head_idx = position.head_idx
            sink_weight = (
                sink_weights_ptr[head_idx].cast[Self.accum_type]() * log2e
            )
            row_max = max(row_max, sink_weight)
        else:
            sink_weights_ptr = UnsafePointer[Scalar[Self.qkv_type]]()
            sink_weight = 0.0

        @parameter
        @always_inline
        fn store_exp(
            row_max: Scalar[Self.accum_type],
        ) -> SIMD[Self.accum_type, 2]:
            alias exp_simd = 2
            alias vs_len = config.BN // exp_simd  # 128 // 2 = 64
            alias batch_size = 32
            alias num_batch_iters = vs_len // batch_size
            alias remainder = vs_len % batch_size
            constrained[num_batch_iters > 0]()
            alias BatchTileType = TMemTile[
                Self.qkv_type, config.BM // 2, batch_size * exp_simd
            ]
            alias RemainderTileType = TMemTile[
                Self.qkv_type, config.BM // 2, remainder * exp_simd
            ]
            constrained[(config.BN % exp_simd) == 0]()

            vs = s.vectorize[exp_simd]()
            # We batch stores, e.g. use `tcgen_05.st.x32`.
            # If we have BN = 128, we would perform two such stores
            # (storing 64 elements as 32x bf16x2)
            #
            # Let `x` be the number of elements we add prior to storing.
            # If `x < 64`, with BN = 128, we have these live counts at
            # the two `tcgen_05.st.x32`:
            # 0. (BN - x) + 32
            # 1. (BN - x) + 32
            #
            # Thus, we can sum the first 32 elements, leaving the remaining 96
            # in registers until after we write.
            # The optimal solution for the number to do in advance is also
            # independent of the number of batches.
            alias AccType = SIMD[Self.accum_type, exp_simd]
            var acc: AccType = exp2(rebind[AccType](vs[0]) - row_max)
            vs[0] = rebind[vs.element_type](acc)

            @parameter
            for i in range(1, batch_size // 2):
                vsi = exp2(rebind[AccType](vs[i]) - row_max)
                vs[i] = rebind[vs.element_type](vsi)
                acc += vsi

            # at this point, we need 32 fewer fp32 registers but 16 more u32
            @parameter
            for i in range(batch_size // 2, batch_size):
                vs[i] = exp2(vs[i] - row_max)

            BatchTileType(p_tmem).store(
                LocalTensor[
                    Self.accum_type, Layout.row_major(batch_size * exp_simd)
                ](s.ptr)
            )

            @parameter
            for b in range(1, num_batch_iters):
                alias offset = batch_size * b

                @parameter
                for i in range(offset, offset + batch_size):
                    vs[i] = exp2(vs[i] - row_max)

                alias el_offset = offset * exp_simd
                alias tmem_offset = (
                    el_offset * Self.qkv_type.size_of()
                ) // Self.accum_type.size_of()
                BatchTileType(p_tmem + tmem_offset).store(
                    LocalTensor[
                        Self.accum_type, Layout.row_major(batch_size * exp_simd)
                    ](s.ptr + el_offset)
                )

            @parameter
            if remainder > 0:
                alias offset = batch_size * num_batch_iters

                @parameter
                for i in range(offset, offset + remainder):
                    vs[i] = exp2(vs[i] - row_max)

                alias el_offset = offset * exp_simd
                alias tmem_offset = (
                    el_offset * Self.qkv_type.size_of()
                ) // Self.accum_type.size_of()
                RemainderTileType(p_tmem + tmem_offset).store(
                    LocalTensor[
                        Self.accum_type, Layout.row_major(remainder * exp_simd)
                    ](s.ptr + el_offset)
                )

            tcgen05_store_wait()
            tcgen05_fence_before()
            pipeline_s.release()
            # now we can sum the remaining elements of `acc`
            acc0 = vs[batch_size // 2]
            acc1 = vs[batch_size // 2 + 1]
            acc2 = vs[batch_size // 2 + 2] + vs[batch_size // 2 + 3]

            @parameter
            for i in range(batch_size // 2 + 4, vs_len, 4):
                acc += rebind[AccType](vs[i])
                acc0 += vs[i + 1]
                acc1 += vs[i + 2]
                acc2 += vs[i + 3]
            return (acc + rebind[AccType](acc0)) + rebind[AccType](acc1 + acc2)

        var row_sum: SIMD[Self.accum_type, 2] = store_exp(row_max)

        var o_phase: UInt32 = 0  # initial wait is phase 0

        @parameter
        if not SinkType.is_null:
            row_sum[0] += exp2(sink_weight - row_max)

        # TODO: add ordering barriers to prevent overlap
        # between the two softmax warpgroups
        while True:
            kv_row += config.BN
            if kv_row >= end:
                break
            mask_status = position.mask_status(mask, kv_row)
            if mask_status == TileMaskStatus.FULL_MASK:
                continue
            pipeline_s.wait()
            # calculate rowmax
            old_max = row_max
            row_max = max(old_max, load_mask_max(kv_row))
            correction = exp2(old_max - row_max)
            tcgen05_st[
                datapaths=32,
                bits=32,
                repeat=1,
                pack=False,
            ](c_tmem, correction)
            pipeline_c.commit()
            # update s->p
            local_rowsum = store_exp(row_max)
            row_sum = row_sum.fma(correction, local_rowsum)
            o_phase ^= 1
        # Do the final correction and write
        inv_row_sum = recip(row_sum.reduce_add())
        o_tile = Self.UMMA1Type.CType(
            tmem_addr + config.TMEM_O0 + warp_group_idx * config.padded_depth
        )
        # wait on the o_pipeline producer
        o_mbar[warp_group_idx].wait(o_phase)  # consumer wait
        tcgen05_fence_after()  # example 1
        # TODO: pass in a dedicated barrier that a q-writer can wait on in a persistent kernel?
        constrained[output_type.size_of() == Self.qkv_type.size_of()]()
        scale_write_output[config=config](
            row,
            inv_row_sum,
            o_ptr_arg,
            o_smem + warp_group_idx * config.BM // 2 * config.padded_depth,
            o_tile,
            warp_group_idx,
            position,
            o_mbar + 2 + warp_group_idx,  # consumer arrive
        )
        named_barrier[2 * WARPGROUP_SIZE](2)
        if thread_idx.x < 32:
            tcgen05_release_allocation_lock[Self.cta_group]()
            tcgen05_dealloc[Self.cta_group](tmem_addr, config.sm100_tmem_cols)

    @staticmethod
    @always_inline
    fn correction(
        tmem_addr: UInt32,
        mbars: FA4MiscMBars,
        o_mbar: MBarType,
        position: Self.PositionType,
        kv_tile_start_row: UInt32,
        end: UInt32,
        mask: MaskType,
    ):
        constrained[Self.accum_type.size_of() == 4]()

        o0_tmem = tmem_addr + config.TMEM_O0
        o1_tmem = tmem_addr + config.TMEM_O1
        c0_tmem = tmem_addr + config.TMEM_C0
        c1_tmem = tmem_addr + config.TMEM_C1

        pipeline_c0 = mbars.consumer_c0()
        pipeline_c1 = mbars.consumer_c1()
        pipeline_o = ConsumerPipeline[2](o_mbar)

        var kv_row: UInt32 = kv_tile_start_row
        while position.mask_status(mask, kv_row) == TileMaskStatus.FULL_MASK:
            kv_row += config.BN

        alias batch_size = 16
        # output is BM x depth
        alias load_iters = config.depth // (2 * batch_size)
        alias load_remainder = config.depth % (2 * batch_size)

        while True:
            kv_row += config.BN
            if kv_row >= end:
                return
            if position.mask_status(mask, kv_row) == TileMaskStatus.FULL_MASK:
                continue

            @parameter
            for i in range(2):
                var c_tmem: UInt32

                @parameter
                if i == 0:
                    c_tmem = c0_tmem
                    pipeline_c0.wait()
                else:
                    c_tmem = c1_tmem
                    pipeline_c1.wait()

                # correct
                c_scalar = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    dtype = Self.accum_type,
                    pack=False,
                    width=1,
                ](c_tmem)
                tcgen05_load_wait()

                @parameter
                if i == 0:
                    pipeline_c0.release()
                else:
                    pipeline_c1.release()

                change = _vote_nvidia_helper(c_scalar != 1) != 0
                pipeline_o.wait()
                if change:
                    # TODO: experiment with different batch sizes.
                    # The idea here is to both pipeline, and reduce peak register use.
                    constrained[load_iters > 1]()
                    constrained[config.depth % batch_size == 0]()

                    var o_tmem: UInt32

                    @parameter
                    if i == 0:
                        o_tmem = o0_tmem
                    else:
                        o_tmem = o1_tmem

                    var o_b0: SIMD[Self.accum_type, batch_size]
                    var o_b1: SIMD[Self.accum_type, batch_size]
                    o_b0 = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=batch_size,
                        dtype = Self.accum_type,
                        pack=False,
                        width=batch_size,
                    ](o_tmem)

                    @parameter
                    for b in range(load_iters):
                        tcgen05_load_wait()  # ob0 loaded
                        # BN=64 or BN=80, load_iters=2
                        # b=0
                        # b0_offset0=0
                        # b1_offset =16
                        # b0_offset1=32
                        # b=1
                        # b0_offset0=32
                        # b1_offset =48
                        # b0_offset1=64
                        alias b0_offset0 = 2 * b * batch_size
                        alias b1_offset = b0_offset0 + batch_size
                        alias b0_offset1 = b1_offset + batch_size
                        o_b1 = tcgen05_ld[  # 0b1 start
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            dtype = Self.accum_type,
                            pack=False,
                            width=batch_size,
                        ](o_tmem + b1_offset)
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + b0_offset0, o_b0 * c_scalar)
                        tcgen05_load_wait()  # ob1 loaded

                        @parameter
                        if b0_offset1 + batch_size <= config.depth:
                            o_b0 = tcgen05_ld[  # 0b0 start
                                datapaths=32,
                                bits=32,
                                repeat=batch_size,
                                dtype = Self.accum_type,
                                pack=False,
                                width=batch_size,
                            ](o_tmem + b0_offset1)
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + b1_offset, o_b1 * c_scalar)

                    @parameter
                    if load_remainder > 0:
                        tcgen05_load_wait()  # ob1 loaded
                        alias offset = 2 * batch_size * load_iters
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=load_remainder,
                            pack=False,
                        ](o_tmem + offset, o_b0 * c_scalar)
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                pipeline_o.release()

    @staticmethod
    @always_inline
    fn load(
        mbars: FA4MiscMBars,
        kv_pipeline_arg: Self.KVPipelineType,
        position: Self.PositionType,
        kv_tile_start_row: UInt32,
        end: UInt32,
        mask: MaskType,
        q_tma_op: QTMATile[
            KVLUTType.dtype,
            config.swizzle_mode,
            BM = config.BM // 2,
            depth = config.BK0,
            group = config.group,
            decoding=False,
        ],
        k_tma_op: TMANestedTensorTile[
            KVLUTType.dtype,
            config.BN,
            config.padded_depth,
            config.swizzle_mode,
            is_k_major=True,
        ],
        v_tma_op: TMANestedTensorTile[
            KVLUTType.dtype,
            config.BN,
            config.padded_depth,
            config.swizzle_mode,
            is_k_major=False,
        ],
        kv_lut: KVLUTType,
        q_smem: SharedMemPointer[Scalar[KVLUTType.dtype]],
    ):
        alias KVPipeType = KVProducerPipeline[KVLUTType.dtype, config]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        alias SMemTensor[layout: Layout] = SharedMemTensor[
            KVLUTType.dtype, layout
        ]
        alias QType = SMemTensor[type_of(q_tma_op).layout]
        alias KType = SMemTensor[type_of(k_tma_op).layout]
        alias VType = SMemTensor[type_of(v_tma_op).layout]
        constrained[
            QType.layout
            == tile_layout_k_major[
                Self.qkv_type, config.BM // 2, config.BK0, config.swizzle_mode
            ]()
        ]()
        constrained[KType.layout == KVPipeType.KType.layout]()
        constrained[VType.layout == KVPipeType.VType.layout]()

        var kv_col: UInt32 = kv_lut.col_idx(position.kv_head_idx())

        alias q_elements = (config.BM // 2) * config.BK0
        alias q_bytes = Self.qkv_type.size_of() * q_elements

        kv_smem = q_smem + config.BM * config.padded_depth
        var pipeline_kv: KVPipeType = {kv_pipeline_arg, kv_smem}

        var mbark0: KVPipeType.KPairType
        elect = elect_one_sync()

        mbark0 = pipeline_kv.get_k[mma_stage=0, expect=False]()  # no wait
        # copy q0
        if elect:
            # Q0
            mbark0.mbar[].expect_bytes(pipeline_kv.kv_bytes + q_bytes)
            q_tma_op.async_copy(
                QType(q_smem),
                mbark0.mbar[],
                (UInt(position.q_col), UInt(position.q_row)),
            )
        var kv_row: UInt32 = kv_tile_start_row
        while position.mask_status(mask, kv_row) == TileMaskStatus.FULL_MASK:
            kv_row += config.BN
        var kv_gmem_row: UInt32 = kv_lut.row_idx(position.prompt_idx, kv_row)
        # copy k0
        if elect:
            # K0
            k_tma_op.async_copy(
                mbark0.smem,
                mbark0.mbar[],
                (UInt(kv_col), UInt(kv_gmem_row)),
            )
        pipeline_kv.commit_kv_step()
        if elect:
            ref q1_mbar = mbars.q1_wait_mbar()
            q1_mbar.expect_bytes(q_bytes)
            # Q1
            q_tma_op.async_copy(
                QType(q_smem + q_elements),
                q1_mbar,
                (UInt(position.q_col), UInt(position.q_row + config.BM // 2)),
            )
        # copy v0
        if elect:
            mbarv0 = pipeline_kv.get_v[mma_stage=0]()
            v_tma_op.async_copy(
                mbarv0.smem,
                mbarv0.mbar[],
                (UInt(kv_col), UInt(kv_gmem_row)),
            )
        pipeline_kv.commit_kv_step()
        # kv producer loop
        while True:
            kv_row += config.BN
            if kv_row >= end:
                break
            if position.mask_status(mask, kv_row) == TileMaskStatus.FULL_MASK:
                continue
            kv_gmem_row = kv_lut.row_idx(position.prompt_idx, kv_row)
            # produce k
            pipeline_kv.acquire_kv()
            if elect:
                mbarkn = pipeline_kv.get_k[mma_stage=0]()
                k_tma_op.async_copy(
                    mbarkn.smem,
                    mbarkn.mbar[],
                    (UInt(kv_col), UInt(kv_gmem_row)),
                )
            pipeline_kv.commit_kv_step()
            pipeline_kv.acquire_kv()
            if elect:
                mbarvn = pipeline_kv.get_v[mma_stage=0]()
                v_tma_op.async_copy(
                    mbarvn.smem,
                    mbarvn.mbar[],
                    (UInt(kv_col), UInt(kv_gmem_row)),
                )
            pipeline_kv.commit_kv_step()

    @staticmethod
    @always_inline
    fn descriptor_q(
        q_smem: SharedMemPointer[Scalar[Self.qkv_type]],
    ) -> MMASmemDescriptor:
        return smem_descriptor[
            BMN = config.BM // 2,
            BK = config.BK0,
            swizzle_mode = config.swizzle_mode,
            is_k_major=True,
        ](q_smem)

    @staticmethod
    @always_inline
    fn mma(
        tmem_addr: UInt32,
        mbars: FA4MiscMBars,
        kv_pipeline_arg: Self.KVPipelineType,
        o_mbar: MBarType,
        position: Self.PositionType,
        kv_tile_start_row: UInt32,
        end: UInt32,
        mask: MaskType,
        q_smem: SharedMemPointer[Scalar[KVLUTType.dtype]],
    ):
        alias KVPipeType = KVConsumerPipeline[KVLUTType.dtype, config]

        s0_tmem = tmem_addr + config.TMEM_S0
        s1_tmem = tmem_addr + config.TMEM_S1
        o0_tmem = tmem_addr + config.TMEM_O0
        o1_tmem = tmem_addr + config.TMEM_O1

        pipeline_s0 = mbars.producer_s0()  # phase = 1
        pipeline_s1 = mbars.producer_s1()  # phase = 1
        pipeline_o = ProducerPipeline[2](o_mbar)  # phase = 1

        alias q0_size = (config.BM // 2) * config.padded_depth
        alias q0_bytes = q0_size * KVLUTType.dtype.size_of()
        q0 = Self.descriptor_q(q_smem)
        q1 = q0 + q0_bytes
        kv_smem = q_smem + 2 * q0_size

        var pipeline_kv: KVPipeType = {kv_pipeline_arg, kv_smem}

        # We peel the first iteration, as we want to wait on q1
        # First, increment the mask
        var kv_row: UInt32 = kv_tile_start_row
        while position.mask_status(mask, kv_row) == TileMaskStatus.FULL_MASK:
            kv_row += config.BN
        var iter_count: UInt32 = 0
        while True:
            kv_row += config.BN
            if kv_row >= end:
                break
            if position.mask_status(mask, kv_row) == TileMaskStatus.FULL_MASK:
                continue
            iter_count += 1

        # Q_0 @ K_0'
        k0 = pipeline_kv.wait_k[mma_stage=0, pre_increment=False]()  # [kv0]
        Self.UMMA0Type.mma[c_scale=0](q0, k0, s0_tmem)
        pipeline_s0.commit_mma()
        pipeline_s0.step()  # pipline_s0.phase = 0

        # Q_1 @ K_0'
        # pipeline_s1.producer_acquire()
        mbars.q1_wait_mbar().wait()  # wait on Q1
        # we don't need to wait on s1
        Self.UMMA0Type.mma[c_scale=0](q1, k0, s1_tmem)
        pipeline_s1.commit_mma()

        pipeline_kv.release_k()  # [kv0]->kv1
        pipeline_s1.step()

        vlatest = pipeline_kv.wait_v[mma_stage=0]()  # [kv1]
        # For the first V tile in the current KV stage buffer:
        # Use the SAME base pointer you used for K (no manual offset).
        pipeline_s0.acquire()  # wait(phase=0), waits on first consumer release
        Self.UMMA1Type.mma[c_scale=0](s0_tmem, vlatest, o0_tmem)
        pipeline_o.commit_mma()
        pipeline_o.step()

        var c_scale: UInt32 = 0

        while iter_count != 0:
            iter_count -= 1
            # Q_0 @ K_n'
            kn = pipeline_kv.wait_k[mma_stage=0]()  # kv_{2n-1}->[kv_{2n}]
            Self.UMMA0Type.mma[c_scale=0](q0, kn, s0_tmem)
            pipeline_s0.commit_mma()
            pipeline_s0.step()

            # O_1 + P_1 @ V_{n-1}
            pipeline_o.acquire()
            pipeline_s1.acquire()
            Self.UMMA1Type.mma(s1_tmem, vlatest, o1_tmem, c_scale=c_scale)
            pipeline_o.commit_mma()
            pipeline_o.step()
            c_scale = 1
            pipeline_kv.release_v()  # [kv_{2n-1}]

            # Q_1 @ K_n'
            Self.UMMA0Type.mma[c_scale=0](q1, kn, s1_tmem)
            pipeline_s1.commit_mma()
            pipeline_s1.step()

            pipeline_kv.release_k()  # [kv_{2n}]->kv_{2n+1}

            # O_0 + P_0 @ V_n
            vlatest = pipeline_kv.wait_v[mma_stage=0]()  # [kv_{2n+1}]
            pipeline_o.acquire()
            pipeline_s0.acquire()
            Self.UMMA1Type.mma[c_scale=1](s0_tmem, vlatest, o0_tmem)
            pipeline_o.commit_mma()
            pipeline_o.step()

        pipeline_o.acquire()
        pipeline_s1.acquire()
        Self.UMMA1Type.mma(s1_tmem, vlatest, o1_tmem, c_scale=c_scale)
        pipeline_o.commit_mma()

        # pipeline_o.step()
        # pipeline_o.acquire()
        # pipeline_o.step()
        # pipeline_o.acquire()
        # tcgen05_release_allocation_lock[Self.cta_group]()
        # tcgen05_dealloc[Self.cta_group](tmem_addr, config.sm100_tmem_cols)
