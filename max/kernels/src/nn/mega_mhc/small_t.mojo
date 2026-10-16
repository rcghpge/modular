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
"""Decode and mid-T `mega_mhc` kernels: split-K over hidden slices, one CTA
per (slice, m-block), with a cross-CTA reduction through arrival counters.
"""

from std.math import align_up, ceildiv, exp, recip, rsqrt
from std.memory import bitcast
from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.compute.mma import mma
from max.gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
    external_memory,
)
from max.gpu.primitives.grid_controls import wait_on_dependent_grids
import max.gpu.primitives.warp as warp
from max.gpu.sync import barrier, named_barrier

from .common import (
    _BF,
    _F32,
    _HC,
    _NMIX,
    _Shared,
    _ld_acquire,
    _ld_relaxed,
    _pack2,
    _red_add_release,
    _unpack2,
)

comptime _NT = 256
comptime _NW = _NT // 32
comptime _PW = 32  # floats per (token, split) partial row
comptime _SKW = 32  # floats per deferred-Sinkhorn queue entry
comptime _MID_BM = 64  # tokens per mid-T task


@inline(.always)
def _sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


@inline(.always)
def _sinkhorn_store(
    m: _Shared[_F32],
    tg: Int,
    new_post: UnsafePointer[Float32, MutAnyOrigin],
    new_comb: UnsafePointer[Float32, MutAnyOrigin],
    hc_post_scale: Float32,
    sinkhorn_eps: Float32,
    sinkhorn_iters: Int,
):
    """Stores token `tg`'s Post and Sinkhorn Comb gates from its scaled mix
    logits `m[4..23]`, on one thread."""
    var post = SIMD[_F32, 4]()
    comptime for j in range(_HC):
        post[j] = _sigmoid(m[_HC + j]) * hc_post_scale
    new_post.unsafe_store[alignment=16](tg * _HC, post)
    var rw = Array[SIMD[_F32, 4], _HC](fill=0)
    comptime for i in range(_HC):
        var lg = m.unsafe_load[width=4, alignment=16](2 * _HC + i * _HC)
        var e = exp(lg - lg.reduce_max())
        rw[i] = e / e.reduce_add() + sinkhorn_eps
    var r0 = rw[0]
    var r1 = rw[1]
    var r2 = rw[2]
    var r3 = rw[3]
    var inv = recip(r0 + r1 + r2 + r3 + sinkhorn_eps)
    r0 *= inv
    r1 *= inv
    r2 *= inv
    r3 *= inv
    for _ in range(sinkhorn_iters - 1):
        var rs4 = SIMD[_F32, 4](
            r0.reduce_add(), r1.reduce_add(), r2.reduce_add(), r3.reduce_add()
        )
        var ri = recip(rs4 + sinkhorn_eps)
        r0 *= ri[0]
        r1 *= ri[1]
        r2 *= ri[2]
        r3 *= ri[3]
        var ci = recip(r0 + r1 + r2 + r3 + sinkhorn_eps)
        r0 *= ci
        r1 *= ci
        r2 *= ci
        r3 *= ci
    new_comb.unsafe_store[alignment=16](tg * 16 + 0, r0)
    new_comb.unsafe_store[alignment=16](tg * 16 + 4, r1)
    new_comb.unsafe_store[alignment=16](tg * 16 + 8, r2)
    new_comb.unsafe_store[alignment=16](tg * 16 + 12, r3)


@inline(.always)
def _fn_values[
    H: Int, S: Int
](
    fnw: UnsafePointer[Float32, MutAnyOrigin],
    nt: Int,
    row: Int,
    kk: Int,
    s: Int,
) -> SIMD[_F32, 4]:
    """Loads projection row `nt * 8 + row` at split `s`'s K offsets `kk`,
    `kk + 1`, `kk + 8` and `kk + 9`: one MMA B fragment."""
    comptime W = H // S
    var base = (nt * 8 + row) * (_HC * H)
    return fnw.unsafe_load[width=2, alignment=8](
        base + (kk // W) * H + s * W + kk % W
    ).join(
        fnw.unsafe_load[width=2, alignment=8](
            base + ((kk + 8) // W) * H + s * W + (kk + 8) % W
        )
    )


@inline(.always)
def _hi_lo(v: SIMD[_F32, 4]) -> SIMD[_BF, 8]:
    """Splits FP32 values into BF16 high and low parts, so two BF16 MMAs give
    about FP32-exact products."""
    var vh = v.cast[_BF]()
    return vh.join((v - vh.cast[_F32]()).cast[_BF]())


@inline(.always)
def _mix_chunk[
    SHIFTED: Bool
](
    xw: SIMD[DType.uint32, 4],
    rw: Array[SIMD[DType.uint32, 4], _HC],
    cf: _Shared[_F32],
    mut ow: Array[SIMD[DType.uint32, 4], _HC],
    mut x1w: SIMD[DType.uint32, 4],
    mut sq2: SIMD[_F32, 2],
    mut x1sq2: SIMD[_F32, 2],
):
    """Mixes 8 hidden values of `x` (`xw`) into the four streams (`rw`) with
    the token's gates at `cf` (Post 4, Comb 16, previous Pre 4), in packed
    FP32x2 math on BF16x2 words.

    Writes the BF16 new streams to `ow` and, shifted, their collapse with the
    previous Pre to `x1w`, and adds both one's squares to `sq2` and `x1sq2`.
    """
    var cpost = cf.unsafe_load[width=4, alignment=16](0)
    var cc0 = cf.unsafe_load[width=4, alignment=16](4)
    var cc1 = cf.unsafe_load[width=4, alignment=16](8)
    var cc2 = cf.unsafe_load[width=4, alignment=16](12)
    var cc3 = cf.unsafe_load[width=4, alignment=16](16)
    comptime for q in range(4):
        var xf = _unpack2(xw[q])
        var r0 = _unpack2(rw[0][q])
        var r1 = _unpack2(rw[1][q])
        var r2 = _unpack2(rw[2][q])
        var r3 = _unpack2(rw[3][q])
        var x1 = SIMD[_F32, 2](0)
        comptime for j in range(_HC):
            var v = xf * cpost[j]
            v = r0.fma(SIMD[_F32, 2](cc0[j]), v)
            v = r1.fma(SIMD[_F32, 2](cc1[j]), v)
            v = r2.fma(SIMD[_F32, 2](cc2[j]), v)
            v = r3.fma(SIMD[_F32, 2](cc3[j]), v)
            var pw = _pack2(v)
            ow[j][q] = pw
            var nf = _unpack2(pw)
            sq2 = nf.fma(nf, sq2)
            comptime if SHIFTED:
                x1 = nf.fma(SIMD[_F32, 2](cf[20 + j]), x1)
        comptime if SHIFTED:
            var xw1 = _pack2(x1)
            x1w[q] = xw1
            var x1f = _unpack2(xw1)
            x1sq2 = x1f.fma(x1f, x1sq2)


@inline(.always)
def _mma_k16[
    MT: Int, STRIDE: Int
](
    a_words: _Shared[DType.uint32],
    kc: Int,
    g: Int,
    tig: Int,
    b: Array[SIMD[_BF, 8], 3],
    mut acc: Array[SIMD[_F32, 4], MT * 3],
):
    """Accumulates K chunk `kc` of the tile's 24 mix dots: A from the staged
    BF16 new streams (row stride `STRIDE`), B as hi and lo fragments."""
    comptime for mt in range(MT):
        var r0 = mt * 16 + g
        var k0 = kc * 16 + 2 * tig
        var a = bitcast[_BF, 8](
            SIMD[DType.uint32, 4](
                a_words[(r0 * STRIDE + k0) // 2],
                a_words[((r0 + 8) * STRIDE + k0) // 2],
                a_words[(r0 * STRIDE + k0 + 8) // 2],
                a_words[((r0 + 8) * STRIDE + k0 + 8) // 2],
            )
        )
        comptime for nt in range(3):
            var d = acc[mt * 3 + nt]
            mma(d, a, b[nt].slice[4, offset=0](), d)
            mma(d, a, b[nt].slice[4, offset=4](), d)
            acc[mt * 3 + nt] = d


@inline(.always)
def _store_acc[
    MT: Int, BM: Int
](
    red: _Shared[_F32],
    wid: Int,
    g: Int,
    tig: Int,
    acc: Array[SIMD[_F32, 4], MT * 3],
):
    """Stores this warp's MMA accumulators as `red[warp][row][24]`."""
    comptime for mt in range(MT):
        comptime for nt in range(3):
            var d = acc[mt * 3 + nt]
            var row = mt * 16 + g
            var n = nt * 8 + 2 * tig
            red[(wid * BM + row) * _NMIX + n] = d[0]
            red[(wid * BM + row) * _NMIX + n + 1] = d[1]
            red[(wid * BM + row + 8) * _NMIX + n] = d[2]
            red[(wid * BM + row + 8) * _NMIX + n + 1] = d[3]


@inline(.always)
def _sum_partials[
    S: Int
](
    part: UnsafePointer[Float32, MutAnyOrigin],
    tg: Int,
    tid: Int,
    c2: _Shared[_F32],
    c3: _Shared[_F32],
):
    """Sums token `tg`'s `S` split partial rows into `c3[0.._PW]`."""
    comptime assert S % _NW == 0
    var lane = tid % 32
    var wid = tid // 32
    comptime NSP = S // _NW
    var pv = Array[Float32, NSP](fill=0)
    comptime for q in range(NSP):
        pv[q] = part[(tg * S + wid + q * _NW) * _PW + lane]
    var v = Float32(0)
    comptime for q in range(NSP):
        v += pv[q]
    c2[wid * 32 + lane] = v
    barrier()
    if tid < 32:
        var tot = Float32(0)
        comptime for w in range(_NW):
            tot += c2[w * 32 + tid]
        c3[tid] = tot
    barrier()


@inline(.always)
def _load_rows[
    H: Int, SHIFTED: Bool, R: Int, STRIDE: Int
](
    active: Bool,
    tid: Int,
    tg: Int,
    rms_w: UnsafePointer[BFloat16, MutAnyOrigin],
    y: UnsafePointer[BFloat16, MutAnyOrigin],
    new_res: UnsafePointer[BFloat16, MutAnyOrigin],
    mut rows: Array[SIMD[_BF, 8], R * _HC],
    mut ws: Array[SIMD[_BF, 8], R],
):
    """Loads this thread's 8-wide chunks of token `tg`'s RMSNorm input (the
    shifted collapse in `y`, else the four new streams) and weights."""
    comptime for q in range(R):
        var c8 = tid + q * STRIDE
        if active and c8 < H // 8:
            ws[q] = rms_w.unsafe_load[width=8, alignment=16](8 * c8)
            comptime if SHIFTED:
                rows[q * _HC] = y.unsafe_load[width=8, alignment=16](
                    tg * H + 8 * c8
                )
            else:
                comptime for j in range(_HC):
                    rows[q * _HC + j] = new_res.unsafe_load[
                        width=8, alignment=16
                    ]((tg * _HC + j) * H + 8 * c8)


@inline(.always)
def _collapse[
    R: Int
](mut rows: Array[SIMD[_BF, 8], R * _HC], pre: SIMD[_F32, 4]) -> Float32:
    """Collapses each chunk's four streams with Pre into `rows[q * _HC]`,
    rounded to BF16. Returns this thread's sum of squares of the result."""
    var sumsq = Float32(0)
    comptime for q in range(R):
        var acc = rows[q * _HC + 0].cast[_F32]() * pre[0]
        acc = rows[q * _HC + 1].cast[_F32]().fma(SIMD[_F32, 8](pre[1]), acc)
        acc = rows[q * _HC + 2].cast[_F32]().fma(SIMD[_F32, 8](pre[2]), acc)
        acc = rows[q * _HC + 3].cast[_F32]().fma(SIMD[_F32, 8](pre[3]), acc)
        var nib = acc.cast[_BF]()
        rows[q * _HC] = nib
        var nif = nib.cast[_F32]()
        sumsq += (nif * nif).reduce_add()
    return sumsq


@inline(.always)
def _store_y[
    H: Int, R: Int, STRIDE: Int
](
    rows: Array[SIMD[_BF, 8], R * _HC],
    ws: Array[SIMD[_BF, 8], R],
    tid: Int,
    tg: Int,
    rstd: Float32,
    rms_scale: Float32,
    y: UnsafePointer[BFloat16, MutAnyOrigin],
):
    comptime for q in range(R):
        var c8 = tid + q * STRIDE
        if c8 < H // 8:
            var wf = ws[q].cast[_F32]() * rms_scale
            var yv = (rows[q * _HC].cast[_F32]() * rstd) * wf
            y.unsafe_store[alignment=16](tg * H + 8 * c8, yv.cast[_BF]())


@inline(.always)
def _bfrag_regs[H: Int, S: Int]() -> Bool:
    comptime KC = _HC * (H // S) // 16
    return ceildiv(KC, _NW) <= 2


@inline(.always)
def _decode_smem[H: Int, S: Int, BM: Int]() -> IndexList[3]:
    """Byte offsets of the B-fragment and scalar regions, and the total."""
    comptime W = H // S
    comptime K = _HC * W
    comptime CH = W // 8
    comptime A_BYTES = align_up(
        max(BM * (K + 8) * 2, _NW * BM * _NMIX * 4), 128
    )
    comptime B_BYTES = 0 if _bfrag_regs[H, S]() else (K // 16) * 3 * 32 * 16
    comptime SC_FLOATS = BM * 24 + 2 * BM * CH + _NW * 32 + 32 + _NW + 32 * _SKW
    return IndexList[3](
        A_BYTES, A_BYTES + B_BYTES, A_BYTES + B_BYTES + 4 * SC_FLOATS
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_NT))
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
def _decode_kernel[
    H: Int, SHIFTED: Bool, S: Int, BM: Int
](
    x: UnsafePointer[BFloat16, MutAnyOrigin],
    res: UnsafePointer[BFloat16, MutAnyOrigin],
    post_in: UnsafePointer[Float32, MutAnyOrigin],
    comb_in: UnsafePointer[Float32, MutAnyOrigin],
    sp_in: UnsafePointer[Float32, MutAnyOrigin],
    fnw: UnsafePointer[Float32, MutAnyOrigin],
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
    R_in: Int32,
    hc_norm_eps: Float32,
    hc_pre_eps: Float32,
    hc_post_scale: Float32,
    sinkhorn_eps: Float32,
    sinkhorn_iters_in: Int32,
    rms_eps: Float32,
    rms_scale: Float32,
):
    """Decode path: `R_in` groups of `S` CTAs walk the m-blocks. Phase 1 of
    m-block mb (mix, new_res, split-K mixes, partials) overlaps phase 2 of the
    group's previous m-block, where CTA s owns token s (gates, collapse,
    RMSNorm)."""
    comptime W = H // S
    comptime K = _HC * W
    comptime KC = K // 16
    comptime KB = ceildiv(KC, _NW)  # k16 chunks per warp
    comptime CH = W // 8
    comptime MT = BM // 16
    comptime NCHUNK = BM * CH
    comptime CPT = ceildiv(NCHUNK, _NT)
    comptime CE = ceildiv(BM * 24, _NT)
    comptime AST = K + 8  # A-tile row stride (bf16): conflict-free fragment loads
    comptime RPT = ceildiv(H // 8, _NT)  # phase-2 row chunks per thread
    comptime B_REGS = _bfrag_regs[H, S]()
    # With enough splits a second set of CTAs runs the Sinkhorn, overlapping
    # the collapse + RMSNorm. Otherwise each CTA queues its tokens' gates and
    # runs them lane-parallel when the queue fills and at the end.
    comptime SK_SPLIT = S >= 2 * BM
    comptime LAY = _decode_smem[H, S, BM]()
    comptime assert W % 8 == 0
    comptime assert BM % 16 == 0
    comptime assert BM <= S, "one phase-2 token per CTA per m-block"

    var T = Int(T_in)
    var R = Int(R_in)
    var sinkhorn_iters = Int(sinkhorn_iters_in)

    var smem = external_memory[
        UInt8, address_space=.SHARED, alignment=128, name="mhc_smem"
    ]()
    var a_tile = smem.bitcast[BFloat16]()
    var red = smem.bitcast[Float32]()  # aliases a_tile after the MMAs
    # [KC][3][32 lanes][hi 4, lo 4]
    var bfr = (smem + LAY[0]).bitcast[BFloat16]()
    # [BM][24]: post 4, comb 16, previous pre 4
    var coef = (smem + LAY[1]).bitcast[Float32]()
    var sqp = coef + BM * 24  # [BM][CH]
    var x1p = sqp + BM * CH  # [BM][CH]
    var c2 = x1p + BM * CH  # [_NW][32] phase-2 split sums
    var c3 = c2 + _NW * 32  # [32] reduced partials
    var c4 = c3 + 32  # [_NW] row sum of squares
    var skq = c4 + _NW  # [32][_SKW] deferred gates

    var tid = thread_idx.x
    var lane = tid % 32
    var wid = tid // 32
    var g = lane // 4
    var tig = lane % 4
    var s = block_idx.x % S
    var r = block_idx.x // S

    # The projection is a checkpoint weight, so it loads before the PDL wait.
    # Small slices stay in registers and split into hi + lo once the first
    # activations are in flight; larger ones go to shared memory.
    var fraw = Array[SIMD[_F32, 4], KB * 3 if B_REGS else 1](fill=0)
    var breg = Array[SIMD[_BF, 8], KB * 3 if B_REGS else 1](fill=0)
    comptime if B_REGS:
        comptime for i in range(KB):
            var kc = wid + i * _NW
            if kc < KC:
                comptime for nt in range(3):
                    fraw[i * 3 + nt] = _fn_values[H, S](
                        fnw, nt, g, kc * 16 + 2 * tig, s
                    )
    else:
        comptime NB = ceildiv(KC * 3 * 32, _NT)
        comptime for q in range(NB):
            var e = tid + q * _NT
            if e < KC * 3 * 32:
                var l = e % 32
                var v = _fn_values[H, S](
                    fnw, (e // 32) % 3, l // 4, (e // 96) * 16 + 2 * (l % 4), s
                )
                bfr.unsafe_store[alignment=16](e * 8, _hi_lo(v))
    var frags_ready = not B_REGS

    var sc0 = mix_scales[0]
    var sc1 = mix_scales[1]
    var sc2 = mix_scales[2]
    var bias = Array[Float32, _HC](fill=0)
    comptime for j in range(_HC):
        bias[j] = mix_bases[j]

    wait_on_dependent_grids()

    var num_mb = ceildiv(T, BM)
    var prev_mb = -1
    var prev_target = UInt64(0)
    var nsk = 0
    var mb = r
    while True:
        var cur_target = UInt64(0)
        if mb < num_mb:
            var t0 = mb * BM
            var v0 = UInt64(0)
            if tid == 0:
                v0 = _ld_relaxed(ctr + mb)  # before this CTA's arrival
            var xs = Array[SIMD[_BF, 8], CPT](fill=0)
            var rs = Array[SIMD[_BF, 8], CPT * _HC](fill=0)
            comptime for ci in range(CPT):
                var idx = tid + ci * _NT
                if idx < NCHUNK:
                    var t = idx // CH
                    var c = idx % CH
                    var tg = t0 + t
                    if tg < T:
                        var col = s * W + 8 * c
                        xs[ci] = x.unsafe_load[width=8, alignment=16](
                            tg * H + col
                        )
                        comptime for i in range(_HC):
                            rs[ci * _HC + i] = res.unsafe_load[
                                width=8, alignment=16
                            ]((tg * _HC + i) * H + col)
            var cv = Array[Float32, CE](fill=0)
            comptime for q in range(CE):
                var e = tid + q * _NT
                var t = e // 24
                var k = e % 24
                var tg = t0 + t
                if e < BM * 24 and tg < T:
                    if k < 4:
                        cv[q] = post_in[tg * 4 + k]
                    elif k < 20:
                        cv[q] = comb_in[tg * 16 + (k - 4)]
                    else:
                        comptime if SHIFTED:
                            cv[q] = sp_in[tg * 4 + (k - 20)]
            comptime if B_REGS:
                if not frags_ready:
                    comptime for i in range(KB * 3):
                        breg[i] = _hi_lo(fraw[i])
                    frags_ready = True
            comptime for q in range(CE):
                var e = tid + q * _NT
                if e < BM * 24:
                    coef[e] = cv[q]
            barrier()
            comptime for ci in range(CPT):
                var idx = tid + ci * _NT
                if idx < NCHUNK:
                    var t = idx // CH
                    var c = idx % CH
                    var tg = t0 + t
                    var sq = Float32(0)
                    var x1sq = Float32(0)
                    if tg < T:
                        var col = s * W + 8 * c
                        var rw = Array[SIMD[DType.uint32, 4], _HC](fill=0)
                        comptime for i in range(_HC):
                            rw[i] = bitcast[DType.uint32, 4](rs[ci * _HC + i])
                        var ow = Array[SIMD[DType.uint32, 4], _HC](fill=0)
                        var x1w = SIMD[DType.uint32, 4](0)
                        var sq2 = SIMD[_F32, 2](0)
                        var x1sq2 = SIMD[_F32, 2](0)
                        _mix_chunk[SHIFTED](
                            bitcast[DType.uint32, 4](xs[ci]),
                            rw,
                            coef + t * 24,
                            ow,
                            x1w,
                            sq2,
                            x1sq2,
                        )
                        comptime for j in range(_HC):
                            var nr = bitcast[_BF, 8](ow[j])
                            new_res.unsafe_store[alignment=16](
                                (tg * _HC + j) * H + col, nr
                            )
                            a_tile.unsafe_store[alignment=16](
                                t * AST + j * W + 8 * c, nr
                            )
                        sq = sq2.reduce_add()
                        comptime if SHIFTED:
                            y.unsafe_store[alignment=16](
                                tg * H + col, bitcast[_BF, 8](x1w)
                            )
                            x1sq = x1sq2.reduce_add()
                    else:
                        comptime for j in range(_HC):
                            a_tile.unsafe_store[alignment=16](
                                t * AST + j * W + 8 * c, SIMD[_BF, 8](0)
                            )
                    sqp[t * CH + c] = sq
                    x1p[t * CH + c] = x1sq
            barrier()
            var acc = Array[SIMD[_F32, 4], MT * 3](fill=0)
            comptime for i in range(KB):
                var kc = wid + i * _NW
                if kc < KC:
                    var b = Array[SIMD[_BF, 8], 3](fill=0)
                    comptime for nt in range(3):
                        comptime if B_REGS:
                            b[nt] = breg[i * 3 + nt]
                        else:
                            b[nt] = bfr.unsafe_load[width=8, alignment=16](
                                ((kc * 3 + nt) * 32 + lane) * 8
                            )
                    _mma_k16[MT, AST](
                        smem.bitcast[UInt32](), kc, g, tig, b, acc
                    )
            barrier()
            _store_acc[MT, BM](red, wid, g, tig, acc)
            barrier()
            comptime for q in range(ceildiv(BM * _PW, _NT)):
                var o = tid + q * _NT
                var t = o // _PW
                var n = o % _PW
                if o < BM * _PW:
                    var v = Float32(0)
                    if n < _NMIX:
                        comptime for w in range(_NW):
                            v += red[(w * BM + t) * _NMIX + n]
                    elif n == _NMIX:
                        comptime for c in range(CH):
                            v += sqp[t * CH + c]
                    elif n == _NMIX + 1:
                        comptime if SHIFTED:
                            comptime for c in range(CH):
                                v += x1p[t * CH + c]
                    part[((t0 + t) * S + s) * _PW + n] = v
            barrier()
            if tid == 0:
                # bar.sync orders the CTA's writes before the release
                _red_add_release(ctr + mb, 1)
                cur_target = v0 - v0 % UInt64(S) + UInt64(S)
        var p2_y = s < BM
        var p2_sk: Bool
        var s_tok: Int
        comptime if SK_SPLIT:
            p2_sk = s >= BM and s < 2 * BM
            s_tok = s if p2_y else s - BM
        else:
            p2_sk = p2_y
            s_tok = s
        if prev_mb >= 0 and (p2_y or p2_sk) and prev_mb * BM + s_tok < T:
            var tg = prev_mb * BM + s_tok
            if tid == 0:
                while _ld_acquire(ctr + prev_mb) < prev_target:
                    pass
            barrier()
            # whole-row loads first, so they overlap the partial reduction
            var rows = Array[SIMD[_BF, 8], RPT * _HC](fill=0)
            var ws = Array[SIMD[_BF, 8], RPT](fill=0)
            if p2_y:
                _load_rows[H, SHIFTED, RPT, _NT](
                    True, tid, tg, rms_w, y, new_res, rows, ws
                )
            _sum_partials[S](part, tg, tid, c2, c3)
            var rms_hc = rsqrt(c3[_NMIX] / Float32(_HC * H) + hc_norm_eps)
            var pre = SIMD[_F32, 4]()
            comptime for j in range(_HC):
                pre[j] = _sigmoid(c3[j] * rms_hc * sc0 + bias[j]) + hc_pre_eps
            if p2_y:
                var rstd: Float32
                comptime if SHIFTED:
                    rstd = rsqrt(c3[_NMIX + 1] / Float32(H) + rms_eps)
                else:
                    var sumsq = warp.sum(_collapse[RPT](rows, pre))
                    if lane == 0:
                        c4[wid] = sumsq
                    barrier()
                    var tot = Float32(0)
                    comptime for w in range(_NW):
                        tot += c4[w]
                    rstd = rsqrt(tot / Float32(H) + rms_eps)
                _store_y[H, RPT, _NT](rows, ws, tid, tg, rstd, rms_scale, y)
            if p2_sk:
                if tid < 2 * _HC + _HC * _HC:
                    var k = _HC + tid
                    var scl = sc1 if k < 2 * _HC else sc2
                    skq[nsk * _SKW + k] = c3[k] * rms_hc * scl + mix_bases[k]
                if tid == 0:
                    comptime if SHIFTED:
                        new_prev.unsafe_store[alignment=16](tg * _HC, pre)
                    skq[nsk * _SKW] = bitcast[_F32, 1](Int32(tg))
                nsk += 1
                comptime if SK_SPLIT:
                    barrier()
                    if tid == 0:
                        _sinkhorn_store(
                            skq,
                            tg,
                            new_post,
                            new_comb,
                            hc_post_scale,
                            sinkhorn_eps,
                            sinkhorn_iters,
                        )
                    nsk = 0
                else:
                    if nsk == 32:
                        barrier()
                        if wid == 0:
                            var e = skq + lane * _SKW
                            _sinkhorn_store(
                                e,
                                Int(bitcast[DType.int32, 1](e[0])),
                                new_post,
                                new_comb,
                                hc_post_scale,
                                sinkhorn_eps,
                                sinkhorn_iters,
                            )
                        barrier()
                        nsk = 0
        if mb >= num_mb:
            break
        prev_mb = mb
        prev_target = cur_target
        mb += R
    comptime if not SK_SPLIT:
        if nsk > 0:
            barrier()
            if wid == 0 and lane < nsk:
                var e = skq + lane * _SKW
                _sinkhorn_store(
                    e,
                    Int(bitcast[DType.int32, 1](e[0])),
                    new_post,
                    new_comb,
                    hc_post_scale,
                    sinkhorn_eps,
                    sinkhorn_iters,
                )


@inline(.always)
def _mid_smem[H: Int, S: Int]() -> IndexList[4]:
    """Byte offsets: coefficients | sum-of-squares partials | phase-2 scratch
    | total. The staging rows ([BM][res 0..3 | x], bf16, updated in place to
    new_res: the MMA A tile) start at 0 and are later reused for the
    cross-warp MMA reduction."""
    comptime W = H // S
    comptime CH = W // 8
    comptime SST = (_HC + 1) * W + 8
    comptime R0 = align_up(
        max(_MID_BM * SST * 2, _NW * _MID_BM * _NMIX * 4), 128
    )
    comptime C0 = R0 + _MID_BM * 24 * 4
    comptime Q0 = C0 + _MID_BM * CH * 2 * 4
    return IndexList[4](R0, C0, Q0, Q0 + (_NW * 32 + 32 + _NW + 32) * 4)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_NT))
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(2))
def _mid_kernel[
    H: Int, SHIFTED: Bool, S: Int
](
    x: UnsafePointer[BFloat16, MutAnyOrigin],
    res: UnsafePointer[BFloat16, MutAnyOrigin],
    post_in: UnsafePointer[Float32, MutAnyOrigin],
    comb_in: UnsafePointer[Float32, MutAnyOrigin],
    sp_in: UnsafePointer[Float32, MutAnyOrigin],
    fnw: UnsafePointer[Float32, MutAnyOrigin],
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
    """Mid-T path: one task per CTA, CTA (s, mb) = hidden slice s of m-block
    mb, two CTAs per SM. Phase 1 streams the task in with coalesced cp.async
    into shared memory, mixes it in place, stores new_res, runs the split-K
    mixes and writes the partial, then arrives on the m-block's counter.
    Phase 2: CTA s owns token s of its m-block (gates, collapse + RMSNorm,
    y), and its last warp runs the token's Sinkhorn meanwhile."""
    comptime W = H // S
    comptime BM = _MID_BM
    comptime K = _HC * W
    comptime KC = K // 16
    comptime KB = ceildiv(KC, _NW)
    comptime CH = W // 8
    comptime MT = BM // 16
    comptime SST = (_HC + 1) * W + 8
    comptime NLD = BM * (_HC + 1) * CH  # 16-byte chunks per task
    comptime CPT = ceildiv(BM * CH, _NT)
    comptime LAY = _mid_smem[H, S]()
    comptime assert BM <= S and S % _NW == 0 and W % 8 == 0

    var T = Int(T_in)
    var sinkhorn_iters = Int(sinkhorn_iters_in)
    var smem = external_memory[
        UInt8, address_space=.SHARED, alignment=128, name="mhc_mid_smem"
    ]()
    var stg = smem.bitcast[BFloat16]()
    var red = smem.bitcast[Float32]()
    var coef = (smem + LAY[0]).bitcast[Float32]()  # [BM][24]
    var sqp = (smem + LAY[1]).bitcast[Float32]()  # [BM][CH][2]
    var c2 = (smem + LAY[2]).bitcast[Float32]()  # [_NW][32]
    var c3 = c2 + _NW * 32  # [32]
    var c4 = c3 + 32  # [_NW]
    var msk = c4 + _NW  # [32] scaled mixes for the Sinkhorn

    var tid = thread_idx.x
    var lane = tid % 32
    var wid = tid // 32
    var g = lane // 4
    var tig = lane % 4
    var s = block_idx.x % S
    var mb = block_idx.x // S
    var t0 = mb * BM

    # The projection is a checkpoint weight, so it loads before the PDL wait.
    var fraw = Array[SIMD[_F32, 4], KB * 3](fill=0)
    comptime for i in range(KB):
        var kc = wid + i * _NW
        if kc < KC:
            comptime for nt in range(3):
                fraw[i * 3 + nt] = _fn_values[H, S](
                    fnw, nt, g, kc * 16 + 2 * tig, s
                )
    wait_on_dependent_grids()
    var v0 = UInt64(0)
    if tid == 0:
        v0 = _ld_relaxed(ctr + mb)  # before this CTA's arrival

    comptime for i in range(ceildiv(BM * 6, _NT)):
        var e = tid + i * _NT
        var tc = e // 6
        var pc = e % 6
        if e < BM * 6 and t0 + tc < T:
            var tgc = t0 + tc
            var src = post_in + tgc * 4
            if pc >= 1 and pc <= 4:
                src = comb_in + (tgc * 16 + (pc - 1) * 4)
            elif pc == 5:
                src = sp_in + tgc * 4
            async_copy[16](src.address_space_cast[.GLOBAL](), coef + 4 * e)
    comptime NLI = ceildiv(NLD, _NT)
    # Two halves in flight: mix the first while the second lands.
    comptime PIPE = CPT == 2 and NLI == 2 * (_HC + 1)
    comptime for i in range(NLI):
        var e = tid + i * _NT
        var c = e % CH
        var r = (e // CH) % (_HC + 1)
        var t = e // (CH * (_HC + 1))
        if e < NLD and t0 + t < T:
            var src = x + (
                (t0 + t) * H + s * W + 8 * c
            ) if r == _HC else res + (((t0 + t) * _HC + r) * H + s * W + 8 * c)
            async_copy[16](
                src.address_space_cast[.GLOBAL](),
                stg + (t * SST + r * W + 8 * c),
            )
        comptime if PIPE and i == NLI // 2 - 1:
            async_copy_commit_group()
    async_copy_commit_group()
    var breg = Array[SIMD[_BF, 8], KB * 3](fill=0)
    comptime for i in range(KB * 3):
        breg[i] = _hi_lo(fraw[i])
    comptime if PIPE:
        async_copy_wait_group(1)
    else:
        async_copy_wait_all()
    barrier()
    for i in range(CPT):
        comptime if PIPE:
            if i == 1:
                async_copy_wait_all()
                barrier()
        var e = tid + i * _NT
        if e < BM * CH:
            var c = e % CH
            var t = e // CH
            var tg = t0 + t
            var roww = (stg + (t * SST + 8 * c)).bitcast[UInt32]()
            var rw = Array[SIMD[DType.uint32, 4], _HC](fill=0)
            comptime for j in range(_HC):
                rw[j] = roww.unsafe_load[width=4, alignment=16](j * W // 2)
            var ow = Array[SIMD[DType.uint32, 4], _HC](fill=0)
            var x1w = SIMD[DType.uint32, 4](0)
            var sq2 = SIMD[_F32, 2](0)
            var x1sq2 = SIMD[_F32, 2](0)
            _mix_chunk[SHIFTED](
                roww.unsafe_load[width=4, alignment=16](_HC * W // 2),
                rw,
                coef + t * 24,
                ow,
                x1w,
                sq2,
                x1sq2,
            )
            comptime for j in range(_HC):
                roww.unsafe_store[alignment=16](j * W // 2, ow[j])
            if tg < T:
                var nrb = (new_res + (tg * _HC * H + s * W + 8 * c)).bitcast[
                    UInt32
                ]()
                comptime for j in range(_HC):
                    nrb.unsafe_store[alignment=16](UInt32(j * H // 2), ow[j])
                comptime if SHIFTED:
                    (y + (tg * H + s * W + 8 * c)).bitcast[
                        UInt32
                    ]().unsafe_store[alignment=16](0, x1w)
            sqp.unsafe_store[alignment=8](
                2 * e, SIMD[_F32, 2](sq2.reduce_add(), x1sq2.reduce_add())
            )
    barrier()
    var acc = Array[SIMD[_F32, 4], MT * 3](fill=0)
    comptime for i in range(KB):
        var kc = wid + i * _NW
        if kc < KC:
            var b = Array[SIMD[_BF, 8], 3](fill=0)
            comptime for nt in range(3):
                b[nt] = breg[i * 3 + nt]
            _mma_k16[MT, SST](smem.bitcast[UInt32](), kc, g, tig, b, acc)
    barrier()
    _store_acc[MT, BM](red, wid, g, tig, acc)
    barrier()
    var pb = part + (t0 * S + s) * _PW
    comptime for q in range(BM * _PW // _NT):
        var t = wid + _NW * q
        if t0 + t < T:
            var v = Float32(0)
            if lane < _NMIX:
                comptime for w in range(_NW):
                    v += red[(w * BM + t) * _NMIX + lane]
            elif lane < _NMIX + 2:
                comptime for c in range(CH):
                    v += sqp[(t * CH + c) * 2 + (lane - _NMIX)]
            pb[UInt32(t * S * _PW + lane)] = v
    barrier()
    var target = UInt64(0)
    if tid == 0:
        _red_add_release(ctr + mb, 1)
        target = v0 - v0 % UInt64(S) + UInt64(S)

    var tg = t0 + s
    if s >= BM or tg >= T:
        return
    var sc0 = mix_scales[0]
    var sc1 = mix_scales[1]
    var sc2 = mix_scales[2]
    if tid < _NMIX:
        msk[tid] = mix_bases[tid]
    if tid == 0:
        while _ld_acquire(ctr + mb) < target:
            pass
    barrier()
    _sum_partials[S](part, tg, tid, c2, c3)
    comptime YT = _NT - 32  # the last warp runs the Sinkhorn meanwhile
    comptime RPY = ceildiv(H // 8, YT)
    var rows = Array[SIMD[_BF, 8], RPY * _HC](fill=0)
    var ws = Array[SIMD[_BF, 8], RPY](fill=0)
    _load_rows[H, SHIFTED, RPY, YT](
        tid < YT, tid, tg, rms_w, y, new_res, rows, ws
    )
    var rms_hc = rsqrt(c3[_NMIX] / Float32(_HC * H) + hc_norm_eps)
    var pre = SIMD[_F32, 4]()
    comptime for j in range(_HC):
        pre[j] = _sigmoid(c3[j] * rms_hc * sc0 + msk[j]) + hc_pre_eps
    if wid == _NW - 1:
        if lane == 0:
            comptime if SHIFTED:
                new_prev.unsafe_store[alignment=16](tg * _HC, pre)
            comptime for kk in range(_HC, 2 * _HC + _HC * _HC):
                var scl = sc1 if kk < 2 * _HC else sc2
                msk[kk] = c3[kk] * rms_hc * scl + msk[kk]
            _sinkhorn_store(
                msk,
                tg,
                new_post,
                new_comb,
                hc_post_scale,
                sinkhorn_eps,
                sinkhorn_iters,
            )
        return
    var rstd: Float32
    comptime if SHIFTED:
        rstd = rsqrt(c3[_NMIX + 1] / Float32(H) + rms_eps)
    else:
        var sumsq = warp.sum(_collapse[RPY](rows, pre))
        if lane == 0:
            c4[wid] = sumsq
        named_barrier[Int32(YT)](1)
        var tot = Float32(0)
        comptime for w in range(_NW - 1):
            tot += c4[w]
        rstd = rsqrt(tot / Float32(H) + rms_eps)
    _store_y[H, RPY, YT](rows, ws, tid, tg, rstd, rms_scale, y)
