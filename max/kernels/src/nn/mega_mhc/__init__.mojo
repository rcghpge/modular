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
"""Fused mHC step for four residual streams (DeepSeek V4 hyper-connections).

On B200, `mega_mhc` picks one of three fused kernels by token count: decode
and mid-T (`small_t`) or the persistent large-T kernel (`large_t`, with its
token epilogue in `large_t_epilogue`). Other GPUs and shapes run the simple
kernels in `fallback`.
"""

from std.math import align_up, ceildiv
from layout import TileTensor
from max.gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    FuncAttribute,
)
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.host.nvidia.tma import TensorMapSwizzle, create_tma_descriptor
from std.utils.index import Index

from .common import _Args, _BF, _F32, _HC, _NMIX, _NUM_SMS
from .fallback import _fallback_scratch_size, _launch_fallback
from .large_t import _LT_BK, _LT_NT, _LT_SMEM, _large_t_kernel
from .large_t_epilogue import _LT_BM
from .small_t import (
    _MID_BM,
    _NT,
    _PW,
    _decode_kernel,
    _decode_smem,
    _mid_kernel,
    _mid_smem,
)

comptime _SMALL_CTR = 1088  # decode / mid-T counters: one 16-slot region per split count

comptime MEGA_MHC_COUNTERS = _SMALL_CTR + 3 * 16
"""Number of `UInt64` arrival counters `mega_mhc` needs. Zero them once, then
keep them across calls; give each stream its own."""

comptime _LARGE_T_MIN = 129  # first token count for the large-T kernel
comptime _MAX_T = 256 * _LT_BM  # large-T arrival counters cover 256 m-blocks

comptime _DECODE = 0
comptime _MID = 1
comptime _LARGE = 2


@fieldwise_init
struct _Plan(ImplicitlyCopyable):
    var kernel: Int  # _DECODE, _MID or _LARGE
    var splits: Int
    var block_m: Int


def _small_ctr[splits: Int]() -> Int:
    # Each region only ever advances by its own split count, so a slot is a
    # multiple of it whenever a launch starts.
    comptime assert splits == 64 or splits == 80 or splits == 128
    return _SMALL_CTR + 16 * (0 if splits == 64 else (1 if splits == 80 else 2))


def _mid_splits[hidden: Int]() -> Int:
    # 80 splits keep H=5120's per-split rows at 64 hidden (128 B).
    return 80 if hidden == 5120 else 128


def _large_t_splits(count: Int, hidden: Int) -> Int:
    """DeepGEMM's split heuristic: the fewest equal-ish splits that still
    reach the shortest longest task."""
    var k_blocks = hidden // _LT_BK
    var most = min(64, max(16, _NUM_SMS // ceildiv(count, _LT_BM)))
    return ceildiv(k_blocks, ceildiv(k_blocks, most))


def _busiest_k_blocks(count: Int, hidden: Int, splits: Int) -> Int:
    return ceildiv(ceildiv(count, _LT_BM) * splits, _NUM_SMS) * ceildiv(
        hidden // _LT_BK, splits
    )


def _plan[hidden: Int, shifted: Bool](count: Int) -> _Plan:
    if count >= _LARGE_T_MIN:
        var splits = _large_t_splits(count, hidden)
        # One even wave beats 1.7 ragged ones for shifted H=5120 (T=1024: 9
        # splits of 9 K blocks instead of 16 of 5). Normal keeps the second
        # wave: its heavier epilogue needs the overlap.
        comptime if hidden == 5120 and shifted:
            if _busiest_k_blocks(count, hidden, 9) < _busiest_k_blocks(
                count, hidden, splits
            ):
                splits = 9
        return _Plan(_LARGE, splits, _LT_BM)
    # The decode kernel wins through T=128 for H=4096 Normal.
    if count >= (17 if hidden == 5120 else 65) and (hidden == 5120 or shifted):
        return _Plan(_MID, _mid_splits[hidden](), _MID_BM)
    if count <= 16:
        return _Plan(_DECODE, 128, 16)
    if count <= 64:
        return _Plan(_DECODE, 128, 64)
    return _Plan(_DECODE, 64, 64)


def mega_mhc_scratch_size[hidden: Int, shifted: Bool](count: Int) -> Int:
    """Returns the number of `Float32` scratch values `mega_mhc` needs for
    `count` tokens.

    Parameters:
        hidden: Hidden size.
        shifted: Whether the call is the shifted variant.

    Args:
        count: Number of tokens.

    Returns:
        The scratch size in `Float32` elements.
    """
    # The device picks fused or fallback kernels, so cover both.
    var plan = _plan[hidden, shifted](count)
    var fused: Int
    if plan.kernel == _LARGE:
        # Per task: 24 mix dots and, per row, the stream and x1 sums of squares.
        fused = ceildiv(count, _LT_BM) * plan.splits * _LT_BM * (_NMIX + 2)
    else:
        fused = align_up(count, plan.block_m) * plan.splits * _PW
    return max(_fallback_scratch_size(count), fused)


def _launch_decode[
    hidden: Int, shifted: Bool, splits: Int, block_m: Int
](ctx: DeviceContext, a: _Args) raises:
    comptime kernel = _decode_kernel[hidden, shifted, splits, block_m]
    comptime smem = _decode_smem[hidden, splits, block_m]()[2]
    var f = ctx.compile_function[kernel](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem))
    )
    var resident = f.occupancy_max_active_blocks_per_multiprocessor(_NT, smem)
    # Spin-waits need every CTA of a split group resident.
    var groups = min(ceildiv(a.count, block_m), resident * _NUM_SMS // splits)
    if groups < 1:
        raise Error("mega_mhc: decode split group does not fit")
    ctx.enqueue_function[kernel](
        a.x,
        a.residual,
        a.post,
        a.comb,
        a.previous_pre,
        a.projection,
        a.mix_scales,
        a.mix_bias,
        a.norm_weight,
        a.new_residual,
        a.new_pre,
        a.new_post,
        a.new_comb,
        a.output,
        a.scratch,
        a.counters + _small_ctr[splits](),
        Int32(a.count),
        Int32(groups),
        a.hc_norm_eps,
        a.hc_pre_eps,
        a.hc_post_scale,
        a.sinkhorn_eps,
        a.sinkhorn_iters,
        a.rms_norm_eps,
        a.rms_scale,
        grid_dim=splits * groups,
        block_dim=_NT,
        shared_mem_bytes=smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem)
        ),
    )


def _launch_mid[
    hidden: Int, shifted: Bool
](ctx: DeviceContext, a: _Args) raises:
    comptime splits = _mid_splits[hidden]()
    comptime kernel = _mid_kernel[hidden, shifted, splits]
    comptime smem = _mid_smem[hidden, splits]()[3]
    var f = ctx.compile_function[kernel](
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem))
    )
    var resident = f.occupancy_max_active_blocks_per_multiprocessor(_NT, smem)
    var blocks = ceildiv(a.count, _MID_BM) * splits
    # Spin-waits need the whole grid resident.
    if blocks > resident * _NUM_SMS:
        raise Error("mega_mhc: mid-T grid is not co-resident")
    ctx.enqueue_function[kernel](
        a.x,
        a.residual,
        a.post,
        a.comb,
        a.previous_pre,
        a.projection,
        a.mix_scales,
        a.mix_bias,
        a.norm_weight,
        a.new_residual,
        a.new_pre,
        a.new_post,
        a.new_comb,
        a.output,
        a.scratch,
        a.counters + _small_ctr[splits](),
        Int32(a.count),
        a.hc_norm_eps,
        a.hc_pre_eps,
        a.hc_post_scale,
        a.sinkhorn_eps,
        a.sinkhorn_iters,
        a.rms_norm_eps,
        a.rms_scale,
        grid_dim=blocks,
        block_dim=_NT,
        shared_mem_bytes=smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem)
        ),
    )


def _launch_large_t[
    hidden: Int, shifted: Bool, splits: Int
](ctx: DeviceContext, a: _Args) raises:
    comptime kernel = _large_t_kernel[hidden, shifted, splits]
    comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
    var t = a.count
    var tm_res = create_tma_descriptor[_BF, 3, swizzle_mode=swizzle](
        DeviceBuffer(ctx, a.residual, t * _HC * hidden, owning=False),
        Index(_HC, t, hidden),
        Index(hidden, _HC * hidden, 1),
        Index(_HC, _LT_BM, _LT_BK),
    )
    var tm_x = create_tma_descriptor[_BF, 2, swizzle_mode=swizzle](
        DeviceBuffer(ctx, a.x, t * hidden, owning=False),
        Index(t, hidden),
        Index(hidden, 1),
        Index(_LT_BM, _LT_BK),
    )
    var tm_fn = create_tma_descriptor[_F32, 3, swizzle_mode=swizzle](
        DeviceBuffer(ctx, a.projection, _NMIX * _HC * hidden, owning=False),
        Index(_HC, _NMIX, hidden),
        Index(hidden, _HC * hidden, 1),
        Index(_HC, _NMIX, 32),
    )
    var tm_post = create_tma_descriptor[_F32, 2](
        DeviceBuffer(ctx, a.post, t * _HC, owning=False),
        Index(t, _HC),
        Index(_HC, 1),
        Index(_LT_BM, _HC),
    )
    var tm_comb = create_tma_descriptor[_F32, 2](
        DeviceBuffer(ctx, a.comb, t * 16, owning=False),
        Index(t, 16),
        Index(16, 1),
        Index(_LT_BM, 16),
    )
    # Normal never reads it; any valid map will do.
    var tm_prev = create_tma_descriptor[_F32, 2](
        DeviceBuffer(
            ctx, a.previous_pre if shifted else a.post, t * _HC, owning=False
        ),
        Index(t, _HC),
        Index(_HC, 1),
        Index(_LT_BM, _HC),
    )
    var tm_new_res = create_tma_descriptor[_BF, 3, swizzle_mode=swizzle](
        DeviceBuffer(ctx, a.new_residual, t * _HC * hidden, owning=False),
        Index(_HC, t, hidden),
        Index(hidden, _HC * hidden, 1),
        Index(_HC, _LT_BM, _LT_BK),
    )
    var tm_out = create_tma_descriptor[_BF, 2, swizzle_mode=swizzle](
        DeviceBuffer(ctx, a.output, t * hidden, owning=False),
        Index(t, hidden),
        Index(hidden, 1),
        Index(_LT_BM, _LT_BK),
    )
    ctx.enqueue_function[kernel](
        tm_res,
        tm_x,
        tm_fn,
        tm_post,
        tm_comb,
        tm_prev,
        tm_new_res,
        tm_out,
        a.mix_scales,
        a.mix_bias,
        a.norm_weight,
        a.new_residual,
        a.new_pre,
        a.new_post,
        a.new_comb,
        a.output,
        a.scratch,
        a.counters,
        Int32(t),
        a.hc_norm_eps,
        a.hc_pre_eps,
        a.hc_post_scale,
        a.sinkhorn_eps,
        a.sinkhorn_iters,
        a.rms_norm_eps,
        a.rms_scale,
        grid_dim=_NUM_SMS,
        block_dim=_LT_NT,
        shared_mem_bytes=_LT_SMEM,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(_LT_SMEM)
        ),
    )


def _launch_fused[
    hidden: Int, shifted: Bool
](ctx: DeviceContext, a: _Args) raises:
    var plan = _plan[hidden, shifted](a.count)
    if plan.kernel == _LARGE:
        # Every split count `_plan` can return for this hidden size.
        comptime if hidden == 4096:
            if plan.splits == 32:
                _launch_large_t[hidden, shifted, 32](ctx, a)
            elif plan.splits == 22:
                _launch_large_t[hidden, shifted, 22](ctx, a)
            else:
                _launch_large_t[hidden, shifted, 16](ctx, a)
        else:
            if plan.splits == 40:
                _launch_large_t[hidden, shifted, 40](ctx, a)
            elif plan.splits == 27:
                _launch_large_t[hidden, shifted, 27](ctx, a)
            elif plan.splits == 20:
                _launch_large_t[hidden, shifted, 20](ctx, a)
            elif plan.splits == 9:
                _launch_large_t[hidden, shifted, 9](ctx, a)
            else:
                _launch_large_t[hidden, shifted, 16](ctx, a)
    elif plan.kernel == _MID:
        _launch_mid[hidden, shifted](ctx, a)
    elif plan.block_m == 16:
        _launch_decode[hidden, shifted, 128, 16](ctx, a)
    elif plan.splits == 128:
        _launch_decode[hidden, shifted, 128, 64](ctx, a)
    else:
        _launch_decode[hidden, shifted, 64, 64](ctx, a)


def _raw[
    dtype: DType
](t: TileTensor[dtype, ...]) -> UnsafePointer[
    Scalar[dtype], MutUntrackedOrigin
]:
    # Every kernel indexes the pointer below as dense row-major, so a strided
    # layout would read and write the wrong addresses.
    comptime assert t.is_row_major, "mega_mhc tensors must be row-major"
    return (
        t.ptr.unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .address_space_cast[.GENERIC]()
    )


def mega_mhc[
    hidden: Int, shifted: Bool
](
    new_residual: TileTensor[mut=True, _BF, ...],
    new_pre: TileTensor[mut=True, _F32, ...],
    new_post: TileTensor[mut=True, _F32, ...],
    new_comb: TileTensor[mut=True, _F32, ...],
    output: TileTensor[mut=True, _BF, ...],
    x: TileTensor[mut=False, _BF, ...],
    residual: TileTensor[mut=False, _BF, ...],
    post: TileTensor[mut=False, _F32, ...],
    comb: TileTensor[mut=False, _F32, ...],
    previous_pre: TileTensor[mut=False, _F32, ...],
    projection: TileTensor[mut=False, _F32, ...],
    mix_scales: TileTensor[mut=False, _F32, ...],
    mix_bias: TileTensor[mut=False, _F32, ...],
    norm_weight: TileTensor[mut=False, _BF, ...],
    counters: TileTensor[mut=True, DType.uint64, ...],
    scratch: TileTensor[mut=True, _F32, ...],
    hc_norm_eps: Float32,
    hc_pre_eps: Float32,
    hc_post_scale: Float32,
    sinkhorn_eps: Float32,
    sinkhorn_iters: Int,
    rms_norm_eps: Float32,
    rms_scale: Float32,
    ctx: DeviceContext,
) raises:
    """Runs one fused mHC step for four residual streams.

    Mixes the sublayer output `x` back into the streams (`new_residual =
    x * post + comb^T residual`, rounded to BF16), projects the new streams
    onto the 24 mix logits, derives the next Pre, Post and Sinkhorn-normalized
    Comb gates, and writes the RMSNorm of the Pre-weighted stream collapse.
    The shifted variant collapses with `previous_pre` instead and also writes
    `new_pre`. On B200 at hidden 4096 or 5120 and up to 16384 tokens, this
    runs one of three fused kernels chosen by the token count; otherwise it
    runs four simple kernels.

    Parameters:
        hidden: Hidden size; a multiple of 1024.
        shifted: Collapse with `previous_pre` and write `new_pre`.

    Args:
        new_residual: Mixed streams. Shape: `[count, 4, hidden]`.
        new_pre: Pre gates, written only when shifted. Shape: `[count, 4]`.
        new_post: Post gates. Shape: `[count, 4]`.
        new_comb: Row-major Comb gates. Shape: `[count, 16]`.
        output: Normalized collapse. Shape: `[count, hidden]`.
        x: Sublayer output. Shape: `[count, hidden]`.
        residual: Input streams. Shape: `[count, 4, hidden]`.
        post: Post gates of `x`. Shape: `[count, 4]`.
        comb: Row-major Comb gates of `residual`. Shape: `[count, 16]`.
        previous_pre: Pre gates for the shifted collapse. Shape: `[count, 4]`.
        projection: Mix projection. Shape: `[24, 4 * hidden]`.
        mix_scales: Pre, Post and Comb logit scales. Shape: `[3]`.
        mix_bias: Mix logit bias. Shape: `[24]`.
        norm_weight: RMSNorm weight. Shape: `[hidden]`.
        counters: `MEGA_MHC_COUNTERS` arrival counters, zeroed once before
            the first call and reused afterwards.
        scratch: `mega_mhc_scratch_size[hidden, shifted](count)` values.
        hc_norm_eps: Epsilon of the projection's RMS scale.
        hc_pre_eps: Added to the Pre gates.
        hc_post_scale: Multiplies the Post gates.
        sinkhorn_eps: Epsilon of the Sinkhorn divisions.
        sinkhorn_iters: Sinkhorn iterations, at least 1.
        rms_norm_eps: Epsilon of the output RMSNorm.
        rms_scale: Multiplies the output RMSNorm.
        ctx: The device context.
    """
    comptime assert hidden % 1024 == 0, "hidden must be a multiple of 1024"
    # All three implementations run the first column-normalization pass
    # unconditionally, so a zero would silently behave as one iteration.
    if sinkhorn_iters < 1:
        raise Error("mega_mhc: sinkhorn_iters must be >= 1")
    var count = x.num_elements() // hidden
    if count == 0:
        return
    var a = _Args(
        _raw(x),
        _raw(residual),
        _raw(post),
        _raw(comb),
        _raw(previous_pre),
        _raw(projection),
        _raw(mix_scales),
        _raw(mix_bias),
        _raw(norm_weight),
        _raw(new_residual),
        _raw(new_pre),
        _raw(new_post),
        _raw(new_comb),
        _raw(output),
        _raw(scratch),
        _raw(counters),
        count,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        Int32(sinkhorn_iters),
        rms_norm_eps,
        rms_scale,
    )
    comptime if (hidden == 4096 or hidden == 5120) and _is_sm10x_gpu(
        ctx.default_device_info
    ):
        if (
            count <= _MAX_T
            and ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
            == _NUM_SMS
        ):
            _launch_fused[hidden, shifted](ctx, a)
            return
    _launch_fallback[hidden, shifted](ctx, a)
