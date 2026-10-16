# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""The persistent large-T `mega_mhc` kernel for B200, after DeepGEMM's SM100
`mega_mhc` schedule: TMA loads of 64x64 tiles, a TF32 tcgen05 projection with
its A operand in tensor memory, split-K partials, and warp-specialized roles.
"""

from std.math import ceildiv
from std.sys._assembly import inlined_assembly
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma_arrive,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import mma as umma
from max.gpu.compute.arch.tcgen05 import (
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
from max.gpu.host.nvidia.tma import TMADescriptor, TensorMapSwizzle
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from max.gpu.memory import (
    CacheEviction,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
    fence_async_view_proxy,
)
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.primitives.grid_controls import wait_on_dependent_grids
import max.gpu.primitives.warp as warp
from max.gpu.sync import (
    barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    named_barrier,
    syncwarp,
)

from .common import (
    _F32,
    _HC,
    _NMIX,
    _NUM_SMS,
    _Shared,
    _ld_relaxed,
    _pack2,
    _red_add_release,
    _unpack2,
)
from .large_t_epilogue import (
    _LT_BM,
    _LT_RQ,
    _mix_producer,
    _mix_task,
    _normal_consumer,
    _shifted_norm_worker,
)

comptime _LT_BK = 64
comptime _LT_NT = 768
# Ring depths: residual + x tiles, projection tiles, A stages in tensor
# memory, accumulators, gate coefficients.
comptime _LT_IO = 4
comptime _LT_FS = 2
comptime _LT_AS = 4
comptime _LT_ACC = 2
comptime _LT_CS = 2
comptime _LT_RES_STAGE = _HC * _LT_BM * _LT_BK * 2  # 32 KB
comptime _LT_X_STAGE = _LT_BM * _LT_BK * 2  # 8 KB
comptime _LT_FN_ATOM = _HC * _NMIX * 32 * 4  # 12 KB: 32 hidden x 24 outputs x 4 routes
comptime _LT_FN_STAGE = 2 * _LT_FN_ATOM
comptime _LT_X0 = _LT_IO * _LT_RES_STAGE
comptime _LT_FN0 = _LT_X0 + _LT_IO * _LT_X_STAGE
comptime _LT_CO0 = _LT_FN0 + _LT_FS * _LT_FN_STAGE  # post | comb | pre per coeff stage
comptime _LT_CO_STAGE = _LT_BM * 24 * 4
comptime _LT_ST0 = _LT_CO0 + _LT_CS * _LT_CO_STAGE  # hc sums [2][64], x1 sums [2][64]
comptime _LT_NS0 = _LT_ST0 + 4 * _LT_BM * 4  # normal scratch: 24 warp sums, 6 tokens
comptime _LT_MB0 = _LT_NS0 + 256

# mbarrier indices from _LT_MB0, 8 bytes each.
comptime _B_FULL_IO = 0
comptime _B_EMPTY_IO = _B_FULL_IO + _LT_IO
comptime _B_READY_IO = _B_EMPTY_IO + _LT_IO
comptime _B_FULL_FN = _B_READY_IO + _LT_IO
comptime _B_EMPTY_FN = _B_FULL_FN + _LT_FS
comptime _B_FULL_A = _B_EMPTY_FN + _LT_FS
comptime _B_EMPTY_A = _B_FULL_A + _LT_AS
comptime _B_FULL_ACC = _B_EMPTY_A + _LT_AS
comptime _B_EMPTY_ACC = _B_FULL_ACC + _LT_ACC
comptime _B_FULL_CO = _B_EMPTY_ACC + _LT_ACC
comptime _B_EMPTY_CO = _B_FULL_CO + _LT_CS
comptime _B_FULL_STATS = _B_EMPTY_CO + _LT_CS
comptime _B_EMPTY_STATS = _B_FULL_STATS + 1
comptime _B_MIX = _B_EMPTY_STATS + 1
comptime _LT_NMB = _B_MIX + 1

comptime _LT_MISC0 = _LT_MB0 + 8 * _LT_NMB  # u32: tmem base, ticket, epoch
comptime _LT_RING0 = _LT_MISC0 + 16  # seq[RQ] u32 | cons[RQ] u32 | pre[RQ][4] f32
comptime _LT_SMEM = _LT_RING0 + _LT_RQ * 24
comptime _LT_TMEM_COLS = 256  # A stages 4 x 32 | accumulators 2 x 24
comptime _LT_ACC_COL0 = _LT_AS * 32
comptime _LT_CTR_MIX = 0  # [2 parities][256 m-blocks]
comptime _LT_CTR_NORM = 512
comptime _LT_CTR_EXIT = 1024


@inline(.always)
def _mbar_init(addr: UInt32, count: UInt32):
    inlined_assembly[
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        NoneType,
        constraints="r,r",
        has_side_effect=True,
    ](addr, count)


@inline(.always)
def _mbar_arrive(addr: UInt32):
    inlined_assembly[
        "mbarrier.arrive.shared::cta.b64 _, [$0];",
        NoneType,
        constraints="r,~{memory}",
        has_side_effect=True,
    ](addr)


@inline(.always)
def _mbar_wait(addr: UInt32, phase: UInt32):
    inlined_assembly[
        """{
        .reg .pred p;
        LAB_WAIT:
        mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;
        @!p bra LAB_WAIT;
        }""",
        NoneType,
        constraints="r,r,~{memory}",
        has_side_effect=True,
    ](addr, phase)


@inline(.always)
def _mbar_expect_tx(addr: UInt32, tx: UInt32):
    inlined_assembly[
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        NoneType,
        constraints="r,r,~{memory}",
        has_side_effect=True,
    ](addr, tx)


@inline(.always)
def _red_async_inc_release(p: UnsafePointer[UInt64, MutAnyOrigin]):
    """Arrives on a split counter the way DeepGEMM does: the release
    reduction retires asynchronously."""
    inlined_assembly[
        "red.async.release.gpu.global.add.u64 [$0], 1;",
        NoneType,
        constraints="l,~{memory}",
        has_side_effect=True,
    ](UInt64(Int(p)))


@inline(.always)
def _k_count[H: Int, S: Int](split: Int) -> Int:
    """Number of 64-hidden K blocks in `split`; the first `NKB % S` splits
    take one extra."""
    comptime NKB = H // _LT_BK
    return NKB // S + (1 if split < NKB % S else 0)


@inline(.always)
def _k_first[H: Int, S: Int](split: Int) -> Int:
    comptime NKB = H // _LT_BK
    return split * (NKB // S) + min(split, NKB % S)


@inline(.always)
def _quad_sum(v: Float32) -> Float32:
    """Sums `v` over the four lanes that share a row of an MMA fragment."""
    var r = v + warp.shuffle_xor(v, UInt32(1))
    return r + warp.shuffle_xor(r, UInt32(2))


@inline(.always)
def _load_coefs[
    SHIFTED: Bool
](
    post_map: OpaquePointer[mut=False, _],
    comb_map: OpaquePointer[mut=False, _],
    pre_map: OpaquePointer[mut=False, _],
    smem: UnsafePointer[UInt8, MutUntrackedOrigin, address_space=.SHARED],
    mbp: _Shared[DType.uint64],
    full_co: UInt32,
    stage: Int,
    m0: Int,
):
    """Starts the TMA loads of m-block row `m0`'s Post, Comb and (shifted)
    previous Pre gates into coefficient stage `stage`."""
    var cb = smem + _LT_CO0 + stage * _LT_CO_STAGE
    var bp = mbp + (_B_FULL_CO + stage)
    _mbar_expect_tx(
        full_co + UInt32(8 * stage),
        UInt32(_LT_BM * (20 + (4 if SHIFTED else 0)) * 4),
    )
    cp_async_bulk_tensor_shared_cluster_global(
        cb.bitcast[UInt8](), post_map, bp, Index(0, m0)
    )
    cp_async_bulk_tensor_shared_cluster_global(
        (cb + _LT_BM * 16).bitcast[UInt8](), comb_map, bp, Index(0, m0)
    )
    comptime if SHIFTED:
        cp_async_bulk_tensor_shared_cluster_global(
            (cb + _LT_BM * 80).bitcast[UInt8](), pre_map, bp, Index(0, m0)
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_LT_NT))
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
@__llvm_arg_metadata(tm_res, `nvvm.grid_constant`)
@__llvm_arg_metadata(tm_x, `nvvm.grid_constant`)
@__llvm_arg_metadata(tm_fn, `nvvm.grid_constant`)
@__llvm_arg_metadata(tm_post, `nvvm.grid_constant`)
@__llvm_arg_metadata(tm_comb, `nvvm.grid_constant`)
@__llvm_arg_metadata(tm_sp, `nvvm.grid_constant`)
@__llvm_arg_metadata(tm_nr, `nvvm.grid_constant`)
@__llvm_arg_metadata(tm_y, `nvvm.grid_constant`)
def _large_t_kernel[
    H: Int, SHIFTED: Bool, S: Int
](
    tm_res: TMADescriptor,
    tm_x: TMADescriptor,
    tm_fn: TMADescriptor,
    tm_post: TMADescriptor,
    tm_comb: TMADescriptor,
    tm_sp: TMADescriptor,
    tm_nr: TMADescriptor,
    tm_y: TMADescriptor,
    mix_scales: UnsafePointer[Float32, MutAnyOrigin],
    mix_bases: UnsafePointer[Float32, MutAnyOrigin],
    rms_w: UnsafePointer[BFloat16, MutAnyOrigin],
    new_res: UnsafePointer[BFloat16, MutAnyOrigin],
    new_prev: UnsafePointer[Float32, MutAnyOrigin],
    new_post: UnsafePointer[Float32, MutAnyOrigin],
    new_comb: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[BFloat16, MutAnyOrigin],
    part: UnsafePointer[Float32, MutAnyOrigin],
    ctr: UnsafePointer[UInt64, MutAnyOrigin],
    T_in: Int32,
    hc_norm_eps: Float32,
    hc_pre_eps: Float32,
    hc_post_scale: Float32,
    sinkhorn_eps: Float32,
    sinkhorn_iters_in: Int32,
    rms_eps: Float32,
    rms_scale: Float32,
):
    """One CTA per SM walks tasks (m-block, split) with stride `_NUM_SMS`.

    Warpgroup 0 loads the residual, x and projection tiles, issues the UMMAs
    and stores new_res / x1. Warpgroups 1-2 mix the streams into the A stages
    and 3 drains the accumulators into split partials. Warpgroup 4 derives
    the gates; once their phase-A roles end, the other warpgroups collapse
    and RMSNorm tokens (shifted: 1, 2 and 5).
    """
    # Ring stage and phase math uses shifts: the depths must stay powers of 2.
    comptime assert (
        _LT_IO == 4
        and _LT_FS == 2
        and _LT_AS == 4
        and _LT_ACC == 2
        and _LT_CS == 2
    )
    comptime assert H // _LT_BK >= S and H % (256 * 4) == 0
    var T = Int(T_in)
    var sinkhorn_iters = Int(sinkhorn_iters_in)
    var smem = external_memory[
        UInt8, address_space=.SHARED, alignment=1024, name="mhc_dg_smem"
    ]()
    var tid = thread_idx.x
    var lane = tid % 32
    # Warp index via a shuffle (like cutlass canonical_warp_idx_sync): ptxas then
    # knows the role branches are warp-uniform and keeps UMMA operands in uniform
    # registers instead of wrapping each tcgen05 instruction in an ELECT loop.
    var wid = Int(warp.shuffle_idx(UInt32(tid // 32), UInt32(0)))
    var wg = wid // 4
    var wi = wid % 4
    var bid = block_idx.x
    var num_mb = ceildiv(T, _LT_BM)
    var num_tasks = num_mb * S

    var mbase = UInt32(Int(smem + _LT_MB0))
    var full_io = mbase + UInt32(8 * _B_FULL_IO)
    var empty_io = mbase + UInt32(8 * _B_EMPTY_IO)
    var ready_io = mbase + UInt32(8 * _B_READY_IO)
    var full_fn = mbase + UInt32(8 * _B_FULL_FN)
    var empty_fn = mbase + UInt32(8 * _B_EMPTY_FN)
    var full_a = mbase + UInt32(8 * _B_FULL_A)
    var empty_a = mbase + UInt32(8 * _B_EMPTY_A)
    var full_acc = mbase + UInt32(8 * _B_FULL_ACC)
    var empty_acc = mbase + UInt32(8 * _B_EMPTY_ACC)
    var full_co = mbase + UInt32(8 * _B_FULL_CO)
    var empty_co = mbase + UInt32(8 * _B_EMPTY_CO)
    var full_stats = mbase + UInt32(8 * _B_FULL_STATS)
    var empty_stats = mbase + UInt32(8 * _B_EMPTY_STATS)
    var mix_arrival = mbase + UInt32(8 * _B_MIX)
    var misc = (smem + _LT_MISC0).bitcast[UInt32]()
    var mbp = (smem + _LT_MB0).bitcast[UInt64]()
    var post_map = Pointer(to=tm_post).bitcast[NoneType]()
    var comb_map = Pointer(to=tm_comb).bitcast[NoneType]()
    var pre_map = Pointer(to=tm_sp).bitcast[NoneType]()

    # The counters below may still be in use by the previous launch.
    wait_on_dependent_grids()
    if tid < 2 * _LT_RQ:
        (smem + _LT_RING0).bitcast[UInt32]()[tid] = 0
    if tid == 0:
        misc[1] = 0
        misc[2] = (
            UInt32(_ld_relaxed(ctr + _LT_CTR_EXIT) // UInt64(_NUM_SMS)) + 1
        )
        comptime for i in range(_LT_IO):
            _mbar_init(full_io + UInt32(8 * i), 1)
            _mbar_init(empty_io + UInt32(8 * i), 1)
            _mbar_init(ready_io + UInt32(8 * i), 8)
        comptime for i in range(_LT_FS):
            _mbar_init(full_fn + UInt32(8 * i), 1)
            _mbar_init(empty_fn + UInt32(8 * i), 1)
        comptime for i in range(_LT_AS):
            _mbar_init(full_a + UInt32(8 * i), 128)
            _mbar_init(empty_a + UInt32(8 * i), 1)
        comptime for i in range(_LT_ACC):
            _mbar_init(full_acc + UInt32(8 * i), 1)
            _mbar_init(empty_acc + UInt32(8 * i), 4)
        comptime for i in range(_LT_CS):
            _mbar_init(full_co + UInt32(8 * i), 1)
            _mbar_init(empty_co + UInt32(8 * i), 8)
        _mbar_init(full_stats, 8)
        _mbar_init(empty_stats, UInt32(4 + (1 if SHIFTED else 0)))
        _mbar_init(mix_arrival, 4)
    if wid == 2:
        tcgen05_alloc[1](misc, UInt32(_LT_TMEM_COLS))
        tcgen05_release_allocation_lock[1]()
    barrier()
    tcgen05_fence_after()
    var tmem = misc[0]
    # Launches alternate between two sets of split counters, by the parity of
    # the launch count (the exit counter / _NUM_SMS).
    var ep = Int(misc[2])
    var par = ep % 2
    var mix_ctr = ctr + _LT_CTR_MIX + par * 256
    var norm_ctr = ctr + _LT_CTR_NORM + par * 256
    # The other parity's words serve the next launch: zero them now (the
    # previous launch, their last user, has finished).
    if tid < 2:
        var mbi = bid + tid * _NUM_SMS
        if mbi < 256:
            ctr[_LT_CTR_MIX + (1 - par) * 256 + mbi] = 0
            ctr[_LT_CTR_NORM + (1 - par) * 256 + mbi] = 0

    if wg == 0:
        warpgroup_reg_dealloc[32]()
        if wid == 0:
            # residual + x loads
            if lane == 0:
                var it = 0
                var task = bid
                while task < num_tasks:
                    var sp = task % S
                    var nk = _k_count[H, S](sp)
                    var kb0 = _k_first[H, S](sp)
                    var m0 = (task // S) * _LT_BM
                    for kb in range(nk):
                        var st = it & (_LT_IO - 1)
                        _mbar_wait(
                            empty_io + UInt32(8 * st),
                            (UInt32(((it >> 2) & 1) ^ 1)),
                        )
                        _mbar_expect_tx(
                            full_io + UInt32(8 * st),
                            UInt32(_LT_RES_STAGE + _LT_X_STAGE),
                        )
                        var k = (kb0 + kb) * _LT_BK
                        cp_async_bulk_tensor_shared_cluster_global(
                            (smem + st * _LT_RES_STAGE).bitcast[UInt8](),
                            Pointer(to=tm_res).bitcast[NoneType](),
                            mbp + (_B_FULL_IO + st),
                            Index(k, m0, 0),
                        )
                        cp_async_bulk_tensor_shared_cluster_global(
                            (smem + _LT_X0 + st * _LT_X_STAGE).bitcast[UInt8](),
                            Pointer(to=tm_x).bitcast[NoneType](),
                            mbp + (_B_FULL_IO + st),
                            Index(k, m0),
                        )
                        it += 1
                    task += _NUM_SMS
        elif wid == 1:
            # projection loads: two 32-hidden swizzle atoms per K block
            if lane == 0:
                var it = 0
                var task = bid
                while task < num_tasks:
                    var sp = task % S
                    var nk = _k_count[H, S](sp)
                    var kb0 = _k_first[H, S](sp)
                    for kb in range(nk):
                        var st = it & (_LT_FS - 1)
                        _mbar_wait(
                            empty_fn + UInt32(8 * st),
                            (UInt32(((it >> 1) & 1) ^ 1)),
                        )
                        _mbar_expect_tx(
                            full_fn + UInt32(8 * st), UInt32(_LT_FN_STAGE)
                        )
                        var k = (kb0 + kb) * _LT_BK
                        comptime for at in range(2):
                            cp_async_bulk_tensor_shared_cluster_global(
                                (
                                    smem
                                    + _LT_FN0
                                    + st * _LT_FN_STAGE
                                    + at * _LT_FN_ATOM
                                ).bitcast[UInt8](),
                                Pointer(to=tm_fn).bitcast[NoneType](),
                                mbp + (_B_FULL_FN + st),
                                Index(k + 32 * at, 0, 0),
                            )
                        it += 1
                    task += _NUM_SMS
        elif wid == 2:
            # TF32 UMMA issue: all 32 lanes run the loop (uniform operands), so
            # ptxas issues bare UTCHMMA/UTCBAR instead of per-instruction ELECT loops.
            comptime idesc = UMMAInsDescriptor[UMMAKind.KIND_TF32].create[
                _F32,
                _F32,
                _F32,
                Index[dtype=DType.uint32](_LT_BM, _NMIX),
                transpose_b=True,
            ]()
            var fit = 0
            var cit = 0
            var task = bid
            while task < num_tasks:
                var nk = _k_count[H, S](task % S)
                var cst = cit & (_LT_ACC - 1)
                _mbar_wait(
                    empty_acc + UInt32(8 * cst), (UInt32(((cit >> 1) & 1) ^ 1))
                )
                tcgen05_fence_after()
                var acc = tmem + UInt32(_LT_ACC_COL0 + cst * _NMIX)
                for kb in range(nk):
                    var fst = fit & (_LT_FS - 1)
                    _mbar_wait(
                        full_fn + UInt32(8 * fst), UInt32((fit >> 1) & 1)
                    )
                    tcgen05_fence_after()
                    var fnb = smem + _LT_FN0 + fst * _LT_FN_STAGE
                    var bd0 = MMASmemDescriptor.create[
                        1024, 16, TensorMapSwizzle.SWIZZLE_128B
                    ](fnb)
                    # 8 A atoms per K block = two full turns of the
                    # 4-stage A ring, so its stage and phase are comptime.
                    comptime assert _LT_BK // 8 == 2 * _LT_AS
                    comptime for uk in range(_LT_BK // 8):
                        comptime ast = uk & (_LT_AS - 1)
                        _mbar_wait(
                            full_a + UInt32(8 * ast), UInt32((uk >> 2) & 1)
                        )
                        tcgen05_fence_after()
                        comptime for r in range(_HC):
                            var bd = bd0 + (
                                (uk * 8 // 32) * _LT_FN_ATOM
                                + r * _NMIX * 128
                                + ((uk * 8) % 32) * 4
                            )
                            var cs = UInt32(1)
                            comptime if uk == 0 and r == 0:
                                cs = UInt32(0) if kb == 0 else UInt32(1)
                            if elect_one_sync():
                                umma(
                                    tmem + UInt32(ast * 32 + r * 8),
                                    bd,
                                    acc,
                                    idesc,
                                    cs,
                                )
                        if elect_one_sync():
                            mma_arrive(mbp + (_B_EMPTY_A + ast))
                    if elect_one_sync():
                        mma_arrive(mbp + (_B_EMPTY_FN + fst))
                    fit += 1
                if elect_one_sync():
                    mma_arrive(mbp + (_B_FULL_ACC + cst))
                cit += 1
                task += _NUM_SMS
        else:
            # new_res / x1 stores, coefficient prefetch, Norm arrivals
            var iot = 0
            var cot = 0
            var tph = 0
            if bid < num_tasks:
                var cs0 = cot & (_LT_CS - 1)
                _mbar_wait(
                    empty_co + UInt32(8 * cs0), (UInt32(((cot >> 1) & 1) ^ 1))
                )
                cot += 1
                if lane == 0:
                    _load_coefs[SHIFTED](
                        post_map,
                        comb_map,
                        pre_map,
                        smem,
                        mbp,
                        full_co,
                        cs0,
                        (bid // S) * _LT_BM,
                    )
                syncwarp()
            var task = bid
            while task < num_tasks:
                var sp = task % S
                var nk = _k_count[H, S](sp)
                var kb0 = _k_first[H, S](sp)
                var mb = task // S
                for kb in range(nk):
                    var st = iot & (_LT_IO - 1)
                    _mbar_wait(
                        ready_io + UInt32(8 * st), UInt32((iot >> 2) & 1)
                    )
                    var pre_next = kb == nk - 1 and task + _NUM_SMS < num_tasks
                    var ncs = 0
                    if pre_next:
                        ncs = cot & (_LT_CS - 1)
                        _mbar_wait(
                            empty_co + UInt32(8 * ncs),
                            (UInt32(((cot >> 1) & 1) ^ 1)),
                        )
                        cot += 1
                    if lane == 0:
                        var k = (kb0 + kb) * _LT_BK
                        # Normal re-reads new_res from L2 in its RMSNorm: keep it.
                        comptime NR_EV = CacheEviction.EVICT_NORMAL if SHIFTED else CacheEviction.EVICT_LAST
                        cp_async_bulk_tensor_global_shared_cta[
                            eviction_policy=NR_EV
                        ](
                            (smem + st * _LT_RES_STAGE).bitcast[UInt8](),
                            Pointer(to=tm_nr).bitcast[NoneType](),
                            Index(k, mb * _LT_BM, 0),
                        )
                        comptime if SHIFTED:
                            cp_async_bulk_tensor_global_shared_cta[
                                eviction_policy=CacheEviction.EVICT_LAST
                            ](
                                (smem + _LT_X0 + st * _LT_X_STAGE).bitcast[
                                    UInt8
                                ](),
                                Pointer(to=tm_y).bitcast[NoneType](),
                                Index(k, mb * _LT_BM),
                            )
                        cp_async_bulk_commit_group()
                        if pre_next:
                            _load_coefs[SHIFTED](
                                post_map,
                                comb_map,
                                pre_map,
                                smem,
                                mbp,
                                full_co,
                                ncs,
                                ((task + _NUM_SMS) // S) * _LT_BM,
                            )
                        cp_async_bulk_wait_group[0]()
                        _mbar_arrive(empty_io + UInt32(8 * st))
                    iot += 1
                if lane == 0:
                    # new_res / x1 globally written before any arrival
                    cp_async_bulk_wait_group[0, read=False]()
                syncwarp()
                comptime if SHIFTED:
                    _mbar_wait(full_stats, UInt32(tph))
                    var st_sq = (smem + _LT_ST0).bitcast[Float32]()
                    var x1p = (
                        part + num_tasks * _LT_BM * (_NMIX + 1) + task * _LT_BM
                    )
                    x1p[lane] = (
                        st_sq[2 * _LT_BM + lane] + st_sq[3 * _LT_BM + lane]
                    )
                    x1p[lane + 32] = (
                        st_sq[2 * _LT_BM + lane + 32]
                        + st_sq[3 * _LT_BM + lane + 32]
                    )
                    syncwarp()
                    if lane == 0:
                        _mbar_arrive(empty_stats)
                        _red_async_inc_release(norm_ctr + mb)
                else:
                    if lane == 0:
                        _red_async_inc_release(mix_ctr + mb)
                tph ^= 1
                task += _NUM_SMS
    elif wg <= 2:
        comptime POST_REGS = 136 if SHIFTED else 112
        warpgroup_reg_alloc[POST_REGS]()
        var pw = wg - 1
        var row0 = wi * 16 + lane // 4
        var row1 = row0 + 8
        var cp = lane % 4
        var iot = 0
        var cot = 0
        var tph = 0
        var task = bid
        while task < num_tasks:
            var nk = _k_count[H, S](task % S)
            var cs = cot & (_LT_CS - 1)
            _mbar_wait(full_co + UInt32(8 * cs), UInt32((cot >> 1) & 1))
            cot += 1
            var cb = (smem + _LT_CO0 + cs * _LT_CO_STAGE).bitcast[Float32]()
            var po0 = cb.unsafe_load[width=4, alignment=16](row0 * 4)
            var po1 = cb.unsafe_load[width=4, alignment=16](row1 * 4)
            var cm0 = Array[SIMD[_F32, 4], _HC](fill=0)
            var cm1 = Array[SIMD[_F32, 4], _HC](fill=0)
            comptime for i in range(_HC):
                cm0[i] = cb.unsafe_load[width=4, alignment=16](
                    _LT_BM * 4 + row0 * 16 + 4 * i
                )
                cm1[i] = cb.unsafe_load[width=4, alignment=16](
                    _LT_BM * 4 + row1 * 16 + 4 * i
                )
            var pr0 = SIMD[_F32, 4](0)
            var pr1 = SIMD[_F32, 4](0)
            comptime if SHIFTED:
                pr0 = cb.unsafe_load[width=4, alignment=16](
                    _LT_BM * 20 + row0 * 4
                )
                pr1 = cb.unsafe_load[width=4, alignment=16](
                    _LT_BM * 20 + row1 * 4
                )
            syncwarp()
            if lane == 0:
                fence_async_view_proxy()
                _mbar_arrive(empty_co + UInt32(8 * cs))
            var hsq0 = SIMD[_F32, 2](0)
            var hsq1 = SIMD[_F32, 2](0)
            var xsq0 = SIMD[_F32, 2](0)
            var xsq1 = SIMD[_F32, 2](0)
            for _ in range(nk):
                var st = iot & (_LT_IO - 1)
                _mbar_wait(full_io + UInt32(8 * st), UInt32((iot >> 2) & 1))
                var rs = (smem + st * _LT_RES_STAGE).bitcast[UInt32]()
                var xs = (smem + _LT_X0 + st * _LT_X_STAGE).bitcast[UInt32]()
                comptime for q in range(4):
                    var uk = pw + 2 * q
                    var hp = uk * 4 + cp  # bf16x2 word within the 64-hidden row
                    var o0 = row0 * 32 + (((hp // 4) ^ (row0 % 8)) * 4) + hp % 4
                    var o1 = row1 * 32 + (((hp // 4) ^ (row1 % 8)) * 4) + hp % 4
                    var x0 = _unpack2(xs[o0])
                    var x1 = _unpack2(xs[o1])
                    var ra0 = Array[SIMD[_F32, 2], _HC](fill=0)
                    var ra1 = Array[SIMD[_F32, 2], _HC](fill=0)
                    comptime for r in range(_HC):
                        ra0[r] = _unpack2(rs[r * (_LT_BM * 32) + o0])
                        ra1[r] = _unpack2(rs[r * (_LT_BM * 32) + o1])
                    # ait = pw + 2q + 8 * (K blocks so far): the A ring turns
                    # twice per K block, so stage/phase follow from pw + 2q.
                    var ai = UInt32(pw + 2 * q)
                    var ast = Int(ai & 3)
                    _mbar_wait(empty_a + UInt32(8 * ast), ((ai >> 2) & 1) ^ 1)
                    tcgen05_fence_after()
                    var tv = Array[Float32, 16](fill=0)
                    var y0 = SIMD[_F32, 2](0)
                    var y1 = SIMD[_F32, 2](0)
                    comptime for j in range(_HC):
                        var v0 = x0 * po0[j]
                        var v1 = x1 * po1[j]
                        comptime for i in range(_HC):
                            v0 = ra0[i].fma(SIMD[_F32, 2](cm0[i][j]), v0)
                            v1 = ra1[i].fma(SIMD[_F32, 2](cm1[i][j]), v1)
                        var w0 = _pack2(v0)
                        var w1 = _pack2(v1)
                        v0 = _unpack2(w0)
                        v1 = _unpack2(w1)
                        rs[j * (_LT_BM * 32) + o0] = w0
                        rs[j * (_LT_BM * 32) + o1] = w1
                        hsq0 = v0.fma(v0, hsq0)
                        hsq1 = v1.fma(v1, hsq1)
                        comptime if SHIFTED:
                            y0 = v0.fma(SIMD[_F32, 2](pr0[j]), y0)
                            y1 = v1.fma(SIMD[_F32, 2](pr1[j]), y1)
                        tv[4 * j + 0] = v0[0]
                        tv[4 * j + 1] = v0[1]
                        tv[4 * j + 2] = v1[0]
                        tv[4 * j + 3] = v1[1]
                    tcgen05_st[datapaths=16, bits=256, repeat=4, pack=False](
                        tmem + UInt32(ast * 32) + (UInt32(32 * wi) << 16), tv
                    )
                    comptime if SHIFTED:
                        var yw0 = _pack2(y0)
                        var yw1 = _pack2(y1)
                        var yf0 = _unpack2(yw0)
                        var yf1 = _unpack2(yw1)
                        xsq0 = yf0.fma(yf0, xsq0)
                        xsq1 = yf1.fma(yf1, xsq1)
                        xs[o0] = yw0
                        xs[o1] = yw1
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                    _mbar_arrive(full_a + UInt32(8 * ast))
                fence_async_view_proxy()
                syncwarp()
                if lane == 0:
                    _mbar_arrive(ready_io + UInt32(8 * st))
                iot += 1
            _mbar_wait(empty_stats, UInt32(tph ^ 1))
            var h0 = hsq0.reduce_add()
            var h1 = hsq1.reduce_add()
            h0 = _quad_sum(h0)
            h1 = _quad_sum(h1)
            var st_sq = (smem + _LT_ST0).bitcast[Float32]()
            if cp == 0:
                st_sq[pw * _LT_BM + row0] = h0
                st_sq[pw * _LT_BM + row1] = h1
            comptime if SHIFTED:
                var q0 = xsq0.reduce_add()
                var q1 = xsq1.reduce_add()
                q0 = _quad_sum(q0)
                q1 = _quad_sum(q1)
                if cp == 0:
                    st_sq[(2 + pw) * _LT_BM + row0] = q0
                    st_sq[(2 + pw) * _LT_BM + row1] = q1
            syncwarp()
            if lane == 0:
                _mbar_arrive(full_stats)
            syncwarp()
            tph ^= 1
            task += _NUM_SMS
        warpgroup_reg_dealloc[40]()
    elif wg == 3:
        warpgroup_reg_dealloc[40]()
        var row0 = wi * 16 + lane // 4
        var row1 = row0 + 8
        var cp = lane % 4
        var cit = 0
        var tph = 0
        var task = bid
        while task < num_tasks:
            var mb = task // S
            var cst = cit & (_LT_ACC - 1)
            _mbar_wait(full_acc + UInt32(8 * cst), UInt32((cit >> 1) & 1))
            cit += 1
            tcgen05_fence_after()
            var tb = (
                tmem
                + UInt32(_LT_ACC_COL0 + cst * _NMIX)
                + (UInt32(32 * wi) << 16)
            )
            var va = tcgen05_ld[
                datapaths=16, bits=256, repeat=2, dtype=_F32, pack=False
            ](tb)
            var vb = tcgen05_ld[
                datapaths=16, bits=256, repeat=1, dtype=_F32, pack=False
            ](tb + 16)
            tcgen05_load_wait()
            var gp = part + task * (_LT_BM * _NMIX)
            var ok0 = mb * _LT_BM + row0 < T
            var ok1 = mb * _LT_BM + row1 < T
            comptime for g in range(3):
                var a0: SIMD[_F32, 2]
                var a1: SIMD[_F32, 2]
                comptime if g < 2:
                    a0 = SIMD[_F32, 2](va[4 * g], va[4 * g + 1])
                    a1 = SIMD[_F32, 2](va[4 * g + 2], va[4 * g + 3])
                else:
                    a0 = SIMD[_F32, 2](vb[0], vb[1])
                    a1 = SIMD[_F32, 2](vb[2], vb[3])
                gp.unsafe_store[alignment=8](
                    row0 * _NMIX + 8 * g + 2 * cp,
                    a0 if ok0 else SIMD[_F32, 2](0),
                )
                gp.unsafe_store[alignment=8](
                    row1 * _NMIX + 8 * g + 2 * cp,
                    a1 if ok1 else SIMD[_F32, 2](0),
                )
            tcgen05_fence_before()
            syncwarp()
            if lane == 0:
                _mbar_arrive(empty_acc + UInt32(8 * cst))
            _mbar_wait(full_stats, UInt32(tph))
            if lane < 16:
                var r = wi * 16 + lane
                var st_sq = (smem + _LT_ST0).bitcast[Float32]()
                part[num_tasks * _LT_BM * _NMIX + task * _LT_BM + r] = (
                    st_sq[r] + st_sq[_LT_BM + r]
                )
            syncwarp()
            if lane == 0:
                _mbar_arrive(mix_arrival)
                _mbar_arrive(empty_stats)
            if wi == 0:
                _mbar_wait(mix_arrival, UInt32(tph))
                if lane == 0:
                    _red_async_inc_release(mix_ctr + mb)
            syncwarp()
            tph ^= 1
            task += _NUM_SMS
    elif wg == 4:
        warpgroup_reg_dealloc[40]()
        comptime if SHIFTED:
            var token = bid * 4 + wi
            while token < T:
                _ = _mix_task[H, SHIFTED, S](
                    part,
                    num_tasks,
                    mix_ctr,
                    mix_scales,
                    mix_bases,
                    new_prev,
                    new_post,
                    new_comb,
                    token,
                    lane,
                    hc_norm_eps,
                    hc_pre_eps,
                    hc_post_scale,
                    sinkhorn_eps,
                    sinkhorn_iters,
                )
                token += _NUM_SMS * 4

    comptime if not SHIFTED:
        if wg == 5:
            warpgroup_reg_dealloc[40]()
        named_barrier[Int32(128)](Int32(1 + wg))
        warpgroup_reg_alloc[80]()
        var ring = UInt32(Int(smem + _LT_RING0))
        if wg == 4:
            _mix_producer[H, S](
                ring,
                part,
                num_tasks,
                mix_ctr,
                mix_scales,
                mix_bases,
                new_post,
                new_comb,
                T,
                lane,
                hc_norm_eps,
                hc_pre_eps,
                hc_post_scale,
                sinkhorn_eps,
                sinkhorn_iters,
            )
        else:
            _normal_consumer[H](
                (smem + _LT_NS0).bitcast[Float32](),
                UInt32(Int(smem + _LT_MISC0 + 4)),
                ring,
                rms_w,
                new_res,
                y,
                T,
                wg,
                wi,
                lane,
                rms_eps,
                rms_scale,
            )
    else:
        if (wg >= 1 and wg <= 2) or wg == 5:
            _shifted_norm_worker[H, S](
                UInt32(Int(smem + _LT_MISC0 + 4)),
                part,
                num_tasks,
                norm_ctr,
                rms_w,
                y,
                T,
                lane,
                rms_eps,
                rms_scale,
            )

    tcgen05_fence_before()
    barrier()
    if wid == 2:
        tcgen05_dealloc[1](tmem, UInt32(_LT_TMEM_COLS))
    if tid == 0:
        _red_add_release(ctr + _LT_CTR_EXIT, 1)
