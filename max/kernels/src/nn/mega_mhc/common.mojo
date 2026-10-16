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
"""Constants, PTX wrappers and gate math shared by the `mega_mhc` kernels."""

from std.atomic import Atomic, Ordering
from std.math import exp, recip, rsqrt
from std.memory import bitcast
from std.sys._assembly import inlined_assembly
from max.gpu.memory import AddressSpace
import max.gpu.primitives.warp as warp

comptime _HC = 4
comptime _NMIX = 24
comptime _NUM_SMS = 148
comptime _BF = DType.bfloat16
comptime _F32 = DType.float32


@fieldwise_init
struct _Args(ImplicitlyCopyable):
    """One `mega_mhc` call's buffers and constants, as raw device pointers."""

    var x: UnsafePointer[BFloat16, MutUntrackedOrigin]
    var residual: UnsafePointer[BFloat16, MutUntrackedOrigin]
    var post: UnsafePointer[Float32, MutUntrackedOrigin]
    var comb: UnsafePointer[Float32, MutUntrackedOrigin]
    var previous_pre: UnsafePointer[Float32, MutUntrackedOrigin]
    var projection: UnsafePointer[Float32, MutUntrackedOrigin]
    var mix_scales: UnsafePointer[Float32, MutUntrackedOrigin]
    var mix_bias: UnsafePointer[Float32, MutUntrackedOrigin]
    var norm_weight: UnsafePointer[BFloat16, MutUntrackedOrigin]
    var new_residual: UnsafePointer[BFloat16, MutUntrackedOrigin]
    var new_pre: UnsafePointer[Float32, MutUntrackedOrigin]
    var new_post: UnsafePointer[Float32, MutUntrackedOrigin]
    var new_comb: UnsafePointer[Float32, MutUntrackedOrigin]
    var output: UnsafePointer[BFloat16, MutUntrackedOrigin]
    var scratch: UnsafePointer[Float32, MutUntrackedOrigin]
    var counters: UnsafePointer[UInt64, MutUntrackedOrigin]
    var count: Int
    var hc_norm_eps: Float32
    var hc_pre_eps: Float32
    var hc_post_scale: Float32
    var sinkhorn_eps: Float32
    var sinkhorn_iters: Int32
    var rms_norm_eps: Float32
    var rms_scale: Float32


comptime _Shared[dtype: DType] = UnsafePointer[
    Scalar[dtype], MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@inline(.always)
def _red_add_release(p: UnsafePointer[UInt64, MutAnyOrigin], v: UInt64):
    # `Atomic.fetch_add` lowers to the read-modify-write `atom` form; the
    # stdlib has no non-returning `red`, which retires asynchronously.
    inlined_assembly[
        "red.release.gpu.global.add.u64 [$0], $1;",
        NoneType,
        constraints="l,l,~{memory}",
        has_side_effect=True,
    ](UInt64(Int(p)), v)


@inline(.always)
def _ld_relaxed(p: UnsafePointer[UInt64, MutAnyOrigin]) -> UInt64:
    return Atomic[UInt64].load[ordering=Ordering.RELAXED](p.as_imm())


@inline(.always)
def _ld_acquire(p: UnsafePointer[UInt64, MutAnyOrigin]) -> UInt64:
    return Atomic[UInt64].load[ordering=Ordering.ACQUIRE](p.as_imm())


@inline(.always)
def _unpack2(w: UInt32) -> SIMD[_F32, 2]:
    """Widens a BF16x2 word to (low, high) FP32 values."""
    return SIMD[_F32, 2](
        bitcast[_F32, 1](w << 16), bitcast[_F32, 1](w & 0xFFFF0000)
    )


@inline(.always)
def _pack2(v: SIMD[_F32, 2]) -> UInt32:
    """Rounds (low, high) FP32 values to a BF16x2 word, to nearest even.

    The stdlib has no fused two-way BF16 cvt; per-value casts would double
    the instruction count in this innermost rounding path.
    """
    return inlined_assembly[
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        UInt32,
        constraints="=r,f,f",
        has_side_effect=False,
    ](v[1], v[0])


@inline(.always)
def _warp_gates[
    H: Int, STORE_PRE: Bool
](
    o: Float32,
    sumsq: Float32,
    token: Int,
    lane: Int,
    mix_scales: UnsafePointer[Float32, MutAnyOrigin],
    mix_bases: UnsafePointer[Float32, MutAnyOrigin],
    new_pre: UnsafePointer[Float32, MutAnyOrigin],
    new_post: UnsafePointer[Float32, MutAnyOrigin],
    new_comb: UnsafePointer[Float32, MutAnyOrigin],
    hc_norm_eps: Float32,
    hc_pre_eps: Float32,
    hc_post_scale: Float32,
    sinkhorn_eps: Float32,
    sinkhorn_iters: Int,
) -> Float32:
    """Derives one token's gates on one warp, lane k from mix logit k: Pre
    (lanes 0-3), Post (4-7) and the Sinkhorn-normalized 4x4 Comb (8-23).

    `o` is the lane's projection dot (zero past lane 23) and `sumsq` the
    token's sum of squares over all four streams. Stores Post, Comb and, with
    `STORE_PRE`, Pre. Returns the lane's gate value.
    """
    var is_out = lane < _NMIX
    var rms = rsqrt(sumsq * (1.0 / Float32(_HC * H)) + hc_norm_eps)
    var is_pre = lane < _HC
    var is_pp = lane < 2 * _HC
    var is_comb = is_out and not is_pp
    var mv = Float32(0)
    if is_out:
        var si = 0 if is_pre else (1 if is_pp else 2)
        var aff = o * rms * mix_scales[si] + mix_bases[lane]
        if is_pp:
            var sg = recip(1.0 + exp(-aff))
            mv = sg + hc_pre_eps if is_pre else sg * hc_post_scale
        else:
            mv = aff
    # Comb lane 8 + 4i + j: xor 1 and 2 reach its row, xor 4 and 24 its column.
    var v = mv if is_comb else Float32(0)
    var rmax = max(v, warp.shuffle_xor(v, UInt32(1)))
    rmax = max(rmax, warp.shuffle_xor(rmax, UInt32(2)))
    v = exp(v - rmax)
    var rs = v + warp.shuffle_xor(v, UInt32(1))
    rs += warp.shuffle_xor(rs, UInt32(2))
    v = v * recip(rs) + sinkhorn_eps
    var cs = v + warp.shuffle_xor(v, UInt32(4))
    cs += warp.shuffle_xor(cs, UInt32(24))
    v *= recip(cs + sinkhorn_eps)
    for _ in range(1, sinkhorn_iters):
        rs = v + warp.shuffle_xor(v, UInt32(1))
        rs += warp.shuffle_xor(rs, UInt32(2))
        v *= recip(rs + sinkhorn_eps)
        cs = v + warp.shuffle_xor(v, UInt32(4))
        cs += warp.shuffle_xor(cs, UInt32(24))
        v *= recip(cs + sinkhorn_eps)
    if is_comb:
        new_comb[token * 16 + lane - 2 * _HC] = v
    elif is_pp and not is_pre:
        new_post[token * _HC + lane - _HC] = mv
    elif is_pre:
        comptime if STORE_PRE:
            new_pre[token * _HC + lane] = mv
    return mv
