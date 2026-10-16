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
"""Token epilogue of the large-T `mega_mhc` kernel: each token's gates from
its split partials, then the collapse and RMSNorm of its streams."""

from std.atomic import Atomic, Ordering, fence
from std.math import ceildiv, rsqrt
from std.memory import bitcast
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from max.gpu import block_idx, thread_idx
from max.gpu.intrinsics import warpgroup_reg_alloc
import max.gpu.primitives.warp as warp
from max.gpu.sync import named_barrier, syncwarp

from .common import (
    _F32,
    _HC,
    _NMIX,
    _NUM_SMS,
    _Shared,
    _ld_acquire,
    _pack2,
    _unpack2,
    _warp_gates,
)

comptime _LT_BM = 64
comptime _LT_RQ = 16  # Normal: Mix -> collapse ring slots (per CTA ticket)


@inline(.always)
def _smem_ticket(addr: UInt32) -> UInt32:
    var slot = UnsafePointer[UInt32, MutAnyOrigin, address_space=.SHARED](
        unsafe_from_address=Int(addr)
    )
    return Atomic[UInt32, scope="block"].fetch_add[ordering=Ordering.RELAXED](
        slot, 1
    )


@inline(.always)
def _lds_volatile(addr: UInt32) -> UInt32:
    # The stdlib exposes no volatile shared load; the Mix ring's spin waits
    # need the store-to-load forwarding bypass.
    return inlined_assembly[
        "ld.volatile.shared.u32 $0, [$1];",
        UInt32,
        constraints="=r,r,~{memory}",
        has_side_effect=True,
    ](addr)


@inline(.always)
def _sts_volatile(addr: UInt32, v: UInt32):
    # The stdlib exposes no volatile shared store; see `_lds_volatile`.
    inlined_assembly[
        "st.volatile.shared.u32 [$0], $1;",
        NoneType,
        constraints="r,r,~{memory}",
        has_side_effect=True,
    ](addr, v)


@inline(.always)
def _fence_cta():
    fence[ordering=Ordering.ACQUIRE_RELEASE, scope="block"]()


@inline(.always)
def _ld_ef(p: UnsafePointer[UInt32, MutAnyOrigin]) -> SIMD[DType.uint32, 4]:
    """Loads 16 bytes with no L1 allocation and DeepGEMM's constant L2
    evict-first policy."""
    var r = inlined_assembly[
        (
            "ld.global.L1::no_allocate.L2::cache_hint.v4.b32 {$0, $1, $2, $3},"
            " [$4], $5;"
        ),
        _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
        constraints="=r,=r,=r,=r,l,l",
        has_side_effect=False,
    ](UInt64(Int(p)), UInt64(0x12F0000000000000))
    return SIMD[DType.uint32, 4](r[0], r[1], r[2], r[3])


@inline(.always)
def _mix_task[
    H: Int, SHIFTED: Bool, S: Int
](
    part: UnsafePointer[Float32, MutAnyOrigin],
    num_tasks: Int,
    mix_ctr: UnsafePointer[UInt64, MutAnyOrigin],
    mix_scales: UnsafePointer[Float32, MutAnyOrigin],
    mix_bases: UnsafePointer[Float32, MutAnyOrigin],
    new_prev: UnsafePointer[Float32, MutAnyOrigin],
    new_post: UnsafePointer[Float32, MutAnyOrigin],
    new_comb: UnsafePointer[Float32, MutAnyOrigin],
    token: Int,
    lane: Int,
    hc_norm_eps: Float32,
    hc_pre_eps: Float32,
    hc_post_scale: Float32,
    sinkhorn_eps: Float32,
    sinkhorn_iters: Int,
) -> Float32:
    """Reduces one token's split partials and stores its gates (one warp).
    Returns the lane's gate value (lanes 0-3: Pre)."""
    var mb = token // _LT_BM
    var row = token % _LT_BM
    while _ld_acquire(mix_ctr + mb) < UInt64(S if SHIFTED else 2 * S):
        pass
    var hcp = part + num_tasks * _LT_BM * _NMIX
    var t0 = mb * S
    var hsq = Float32(0)
    comptime for k0 in range(0, S, 32):
        if k0 + lane < S:
            hsq += hcp[(t0 + k0 + lane) * _LT_BM + row]
    hsq = warp.sum(hsq)
    var is_out = lane < _NMIX
    var oi = min(lane, _NMIX - 1)
    var o = Float32(0)
    comptime for k in range(S):
        o += part[((t0 + k) * _LT_BM + row) * _NMIX + oi]
    o = o if is_out else Float32(0)
    return _warp_gates[H, SHIFTED](
        o,
        hsq,
        token,
        lane,
        mix_scales,
        mix_bases,
        new_prev,
        new_post,
        new_comb,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        sinkhorn_iters,
    )


@inline(.always)
def _mix_producer[
    H: Int, S: Int
](
    ring: UInt32,
    part: UnsafePointer[Float32, MutAnyOrigin],
    num_tasks: Int,
    mix_ctr: UnsafePointer[UInt64, MutAnyOrigin],
    mix_scales: UnsafePointer[Float32, MutAnyOrigin],
    mix_bases: UnsafePointer[Float32, MutAnyOrigin],
    new_post: UnsafePointer[Float32, MutAnyOrigin],
    new_comb: UnsafePointer[Float32, MutAnyOrigin],
    T: Int,
    lane: Int,
    hc_norm_eps: Float32,
    hc_pre_eps: Float32,
    hc_post_scale: Float32,
    sinkhorn_eps: Float32,
    sinkhorn_iters: Int,
):
    """Normal mode Mix warp: tickets k = wi, wi + 4, ... of this CTA (token
    bid + k * _NUM_SMS). Publishes the token's Pre coefficients in ring slot
    k % RQ once the collapse workers have released ticket k - RQ."""
    # Recompute the warp index here: a value kept live across warpgroup_reg_alloc
    # is spilled to local memory and reloaded every token.
    var k = Int(warp.shuffle_idx(UInt32(thread_idx.x // 32), UInt32(0))) % 4
    while True:
        var token = block_idx.x + k * _NUM_SMS
        if token >= T:
            return
        var slot = UInt32(k % _LT_RQ)
        if k >= _LT_RQ:
            var want = UInt32(k - _LT_RQ + 1)
            if lane == 0:
                while _lds_volatile(ring + 4 * (UInt32(_LT_RQ) + slot)) != want:
                    pass
            syncwarp()
        # Normal mode stores no Pre, so new_post stands in for new_prev.
        var mv = _mix_task[H, False, S](
            part,
            num_tasks,
            mix_ctr,
            mix_scales,
            mix_bases,
            new_post,
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
        var pre_base = ring + UInt32(8 * _LT_RQ) + 16 * slot
        if lane < _HC:
            _sts_volatile(
                pre_base + 4 * UInt32(lane), bitcast[DType.uint32, 1](mv)
            )
        syncwarp()
        if lane == 0:
            _fence_cta()
            _sts_volatile(ring + 4 * slot, UInt32(k + 1))
        k += 4


@inline(.always)
def _normal_consumer[
    H: Int
](
    sc: _Shared[_F32],
    ticket_addr: UInt32,
    ring: UInt32,
    rms_w: UnsafePointer[BFloat16, MutAnyOrigin],
    new_res: UnsafePointer[BFloat16, MutAnyOrigin],
    y: UnsafePointer[BFloat16, MutAnyOrigin],
    T: Int,
    wg: Int,
    wi: Int,
    lane: Int,
    rms_eps: Float32,
    rms_scale: Float32,
):
    """Normal mode collapse + RMSNorm worker (one warpgroup per token): takes
    the next CTA ticket and waits for its Pre coefficients in the Mix ring."""
    comptime NPK = H // (256 * 4)  # 256-hidden packs per warp
    var wsums = sc  # [24]
    var toks = (sc + 24).bitcast[UInt32]()  # [6]
    var nr = new_res.bitcast[UInt32]()
    if wi == 0 and lane == 0:
        toks[wg] = _smem_ticket(ticket_addr)
    while True:
        named_barrier[Int32(128)](Int32(1 + wg))
        var k = Int(toks[wg])
        var token = block_idx.x + k * _NUM_SMS
        if token >= T:
            return
        var slot = UInt32(k % _LT_RQ)
        while _lds_volatile(ring + 4 * slot) != UInt32(k + 1):
            pass
        _fence_cta()
        var pre = SIMD[_F32, 4](0)
        comptime for r in range(_HC):
            pre[r] = bitcast[_F32, 1](
                _lds_volatile(
                    ring + UInt32(8 * _LT_RQ) + 16 * slot + UInt32(4 * r)
                )
            )
        var keep = Array[SIMD[DType.uint32, 4], NPK](fill=0)
        var sq = SIMD[_F32, 2](0)
        comptime for p in range(NPK):
            var h8 = (wi * NPK + p) * 256 + lane * 8  # hidden index
            var acc = Array[SIMD[_F32, 2], 4](fill=0)
            comptime for r in range(_HC):
                var wv = _ld_ef(nr + (token * _HC * H + r * H + h8) // 2)
                comptime for q in range(4):
                    acc[q] = _unpack2(wv[q]).fma(SIMD[_F32, 2](pre[r]), acc[q])
            comptime for q in range(4):
                var w = _pack2(acc[q])
                keep[p][q] = w
                var f = _unpack2(w)
                sq = f.fma(f, sq)
        # RMSNorm weights load under the row-sum barrier.
        var wts = Array[SIMD[DType.uint32, 4], NPK](fill=0)
        comptime for p in range(NPK):
            wts[p] = rms_w.bitcast[UInt32]().unsafe_load[width=4, alignment=16](
                ((wi * NPK + p) * 256 + lane * 8) // 2
            )
        var ws = warp.sum(sq.reduce_add())
        if lane == 0:
            wsums[wg * 4 + wi] = ws
        named_barrier[Int32(128)](Int32(1 + wg))
        var tot = (
            wsums[wg * 4]
            + wsums[wg * 4 + 1]
            + wsums[wg * 4 + 2]
            + wsums[wg * 4 + 3]
        )
        if wi == 0 and lane == 0:
            # Free the ring slot and take the next ticket.
            _sts_volatile(ring + 4 * (UInt32(_LT_RQ) + slot), UInt32(k + 1))
            toks[wg] = _smem_ticket(ticket_addr)
        var scl = rsqrt(tot * (1.0 / Float32(H)) + rms_eps) * rms_scale
        comptime for p in range(NPK):
            var h8 = (wi * NPK + p) * 256 + lane * 8
            var wv = wts[p]
            var o = SIMD[DType.uint32, 4](0)
            comptime for q in range(4):
                o[q] = _pack2((_unpack2(keep[p][q]) * scl) * _unpack2(wv[q]))
            y.bitcast[UInt32]().unsafe_store[alignment=16](
                (token * H + h8) // 2, o
            )


@inline(.always)
def _shifted_norm_worker[
    H: Int, S: Int
](
    ticket_addr: UInt32,
    part: UnsafePointer[Float32, MutAnyOrigin],
    num_tasks: Int,
    norm_ctr: UnsafePointer[UInt64, MutAnyOrigin],
    rms_w: UnsafePointer[BFloat16, MutAnyOrigin],
    y: UnsafePointer[BFloat16, MutAnyOrigin],
    T: Int,
    lane: Int,
    rms_eps: Float32,
    rms_scale: Float32,
):
    """Shifted mode: warp tasks (token pair, hidden partition) RMSNorm the
    x1 rows that the Post workers stored into y. Like DeepGEMM it runs at 96
    registers; it keeps `NPF` 256-hidden packs in flight."""
    comptime NHP = H // 256
    comptime WAVE = _NUM_SMS * 4
    comptime NPF = 3
    var npairs = ceildiv(T, 2)
    var npart = max(ceildiv(NHP, max(1, npairs * NHP // WAVE)), 2)
    var ntask = npairs * npart
    var base = block_idx.x * 4
    if base >= ntask:
        return
    warpgroup_reg_alloc[96]()
    var x1p = part + num_tasks * _LT_BM * (_NMIX + 1)
    var yw = y.bitcast[UInt32]()
    var ww = rms_w.bitcast[UInt32]()
    while True:
        var tk = UInt32(0)
        if lane == 0:
            tk = _smem_ticket(ticket_addr)
        tk = warp.shuffle_idx(tk, UInt32(0))
        var task = base + Int(tk % 4) + Int(tk // 4) * WAVE
        if task >= ntask:
            return
        var pi = task % npart
        var hb = pi * NHP // npart
        var he = (pi + 1) * NHP // npart
        var m = (task // npart) * 2
        var mb = m // _LT_BM
        var two = m + 1 < T
        var o2 = H // 2 if two else 0  # second token's row offset (u32 words)
        while _ld_acquire(norm_ctr + mb) < UInt64(S):
            pass
        var row = (m * H + lane * 8) // 2
        var wl = (lane * 8) // 2
        # packs hp .. hp + NPF - 1 in registers: index 0 is normalized, the rest are in flight
        var pa = Array[SIMD[DType.uint32, 4], NPF](fill=0)
        var pb = Array[SIMD[DType.uint32, 4], NPF](fill=0)
        var pw = Array[SIMD[DType.uint32, 4], NPF](fill=0)
        comptime for j in range(NPF):
            if hb + j < he:
                pa[j] = _ld_ef(yw + row + (hb + j) * 128)
                pb[j] = _ld_ef(yw + row + (hb + j) * 128 + o2)
                pw[j] = ww.unsafe_load[width=4, alignment=16](
                    wl + (hb + j) * 128
                )
        var q = Float32(0)
        if lane < 2 and m + lane < T:
            for k in range(S):
                q += x1p[(mb * S + k) * _LT_BM + m % _LT_BM + lane]
        var rs = rsqrt(q * (1.0 / Float32(H)) + rms_eps) * rms_scale
        var rs0 = warp.shuffle_idx(rs, UInt32(0))
        var rs1 = warp.shuffle_idx(rs, UInt32(1))
        for hp in range(hb, he):
            var an = SIMD[DType.uint32, 4](0)
            var bn = SIMD[DType.uint32, 4](0)
            var wn = SIMD[DType.uint32, 4](0)
            if hp + NPF < he:
                an = _ld_ef(yw + row + (hp + NPF) * 128)
                bn = _ld_ef(yw + row + (hp + NPF) * 128 + o2)
                wn = ww.unsafe_load[width=4, alignment=16](
                    wl + (hp + NPF) * 128
                )
            var oa = SIMD[DType.uint32, 4](0)
            var ob = SIMD[DType.uint32, 4](0)
            comptime for i in range(4):
                var wf = _unpack2(pw[0][i])
                oa[i] = _pack2((_unpack2(pa[0][i]) * rs0) * wf)
                ob[i] = _pack2((_unpack2(pb[0][i]) * rs1) * wf)
            yw.unsafe_store[alignment=16](row + hp * 128, oa)
            if two:
                yw.unsafe_store[alignment=16](row + hp * 128 + o2, ob)
            comptime for j in range(NPF - 1):
                pa[j] = pa[j + 1]
                pb[j] = pb[j + 1]
                pw[j] = pw[j + 1]
            pa[NPF - 1] = an
            pb[NPF - 1] = bn
            pw[NPF - 1] = wn
