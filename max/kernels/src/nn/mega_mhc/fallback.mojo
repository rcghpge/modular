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
"""Four simple `mega_mhc` kernels for shapes and GPUs the fused kernels don't
cover: the residual mix, the projection, the gates, and the collapse with
RMSNorm."""

from std.math import rsqrt
from std.utils.static_tuple import StaticTuple
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
import max.gpu.primitives.block as block
import max.gpu.primitives.warp as warp

from .common import _Args, _BF, _F32, _HC, _NMIX, _warp_gates

comptime _SPLITS = 4  # projection splits per output


def _fallback_scratch_size(count: Int) -> Int:
    """Scratch for `count` tokens: the projection partials
    `[count][_NMIX + 1][_SPLITS]` (24 dots, then the sum of squares), then
    Normal mode's Pre `[count][4]`."""
    return count * ((_NMIX + 1) * _SPLITS + _HC)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
def _fallback_post[
    hidden: Int
](
    x: UnsafePointer[BFloat16, MutAnyOrigin],
    residual: UnsafePointer[BFloat16, MutAnyOrigin],
    post: UnsafePointer[Float32, MutAnyOrigin],
    comb: UnsafePointer[Float32, MutAnyOrigin],
    new_residual: UnsafePointer[BFloat16, MutAnyOrigin],
):
    var row = block_idx.x
    var channel = block_idx.y * 512 + thread_idx.x * 4
    var input = x.unsafe_load[width=4](row * hidden + channel).cast[_F32]()
    var streams = Array[SIMD[_F32, 4], _HC](fill=0)
    comptime for source in range(_HC):
        streams[source] = residual.unsafe_load[width=4](
            (row * _HC + source) * hidden + channel
        ).cast[_F32]()
    comptime for dest in range(_HC):
        var mixed = input * post[row * _HC + dest]
        comptime for source in range(_HC):
            mixed += streams[source] * comb[row * 16 + source * _HC + dest]
        new_residual.unsafe_store(
            (row * _HC + dest) * hidden + channel, mixed.cast[_BF]()
        )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
def _fallback_project[
    hidden: Int
](
    new_residual: UnsafePointer[BFloat16, MutAnyOrigin],
    projection: UnsafePointer[Float32, MutAnyOrigin],
    partials: UnsafePointer[Float32, MutAnyOrigin],
):
    comptime width = _HC * hidden
    comptime part = width // _SPLITS
    var row = block_idx.x
    var output = block_idx.y // _SPLITS
    var split = block_idx.y % _SPLITS
    var total = SIMD[_F32, 4](0)
    var squares = SIMD[_F32, 4](0)
    comptime for step in range(part // 512):
        var channel = split * part + step * 512 + thread_idx.x * 4
        var value = new_residual.unsafe_load[width=4](
            row * width + channel
        ).cast[_F32]()
        total += value * projection.unsafe_load[width=4](
            output * width + channel
        )
        squares += value * value
    var dot = block.sum[block_size=128, broadcast=False](total.reduce_add())
    if thread_idx.x == 0:
        partials[(row * (_NMIX + 1) + output) * _SPLITS + split] = dot
    if output == 0:
        var norm = block.sum[block_size=128, broadcast=False](
            squares.reduce_add()
        )
        if thread_idx.x == 0:
            partials[(row * (_NMIX + 1) + _NMIX) * _SPLITS + split] = norm


def _fallback_gates[
    hidden: Int
](
    partials: UnsafePointer[Float32, MutAnyOrigin],
    mix_scales: UnsafePointer[Float32, MutAnyOrigin],
    mix_bias: UnsafePointer[Float32, MutAnyOrigin],
    pre: UnsafePointer[Float32, MutAnyOrigin],
    new_post: UnsafePointer[Float32, MutAnyOrigin],
    new_comb: UnsafePointer[Float32, MutAnyOrigin],
    hc_norm_eps: Float32,
    hc_pre_eps: Float32,
    hc_post_scale: Float32,
    sinkhorn_eps: Float32,
    sinkhorn_iters: Int32,
):
    """One warp per token."""
    var row = block_idx.x
    var lane = thread_idx.x
    var squares = Float32(0)
    if lane < _SPLITS:
        squares = partials[(row * (_NMIX + 1) + _NMIX) * _SPLITS + lane]
    var mix = Float32(0)
    if lane < _NMIX:
        comptime for split in range(_SPLITS):
            mix += partials[(row * (_NMIX + 1) + lane) * _SPLITS + split]
    _ = _warp_gates[hidden, True](
        mix,
        warp.sum(squares),
        row,
        lane,
        mix_scales,
        mix_bias,
        pre,
        new_post,
        new_comb,
        hc_norm_eps,
        hc_pre_eps,
        hc_post_scale,
        sinkhorn_eps,
        Int(sinkhorn_iters),
    )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](256))
def _fallback_collapse_norm[
    hidden: Int
](
    new_residual: UnsafePointer[BFloat16, MutAnyOrigin],
    pre: UnsafePointer[Float32, MutAnyOrigin],
    norm_weight: UnsafePointer[BFloat16, MutAnyOrigin],
    output: UnsafePointer[BFloat16, MutAnyOrigin],
    rms_norm_eps: Float32,
    rms_scale: Float32,
):
    comptime tiles = hidden // 1024
    var row = block_idx.x
    var values = Array[SIMD[_F32, 4], tiles](fill=0)
    var squares = SIMD[_F32, 4](0)
    comptime for tile in range(tiles):
        var channel = tile * 1024 + thread_idx.x * 4
        var value = SIMD[_F32, 4](0)
        comptime for source in range(_HC):
            value += (
                new_residual.unsafe_load[width=4](
                    (row * _HC + source) * hidden + channel
                ).cast[_F32]()
                * pre[row * _HC + source]
            )
        values[tile] = value.cast[_BF]().cast[_F32]()
        squares += values[tile] * values[tile]
    var norm = (
        rsqrt(
            block.sum[block_size=256](squares.reduce_add()) / Float32(hidden)
            + rms_norm_eps
        )
        * rms_scale
    )
    comptime for tile in range(tiles):
        var channel = tile * 1024 + thread_idx.x * 4
        var weight = norm_weight.unsafe_load[width=4](channel).cast[_F32]()
        output.unsafe_store(
            row * hidden + channel, (values[tile] * norm * weight).cast[_BF]()
        )


def _launch_fallback[
    hidden: Int, shifted: Bool
](ctx: DeviceContext, a: _Args) raises:
    var partials = a.scratch
    # The shifted collapse uses the previous Pre; Normal uses this step's.
    var pre = (
        a.new_pre if shifted else a.scratch + a.count * (_NMIX + 1) * _SPLITS
    )
    ctx.enqueue_function[_fallback_post[hidden]](
        a.x,
        a.residual,
        a.post,
        a.comb,
        a.new_residual,
        grid_dim=(a.count, hidden // 512),
        block_dim=128,
    )
    ctx.enqueue_function[_fallback_project[hidden]](
        a.new_residual,
        a.projection,
        partials,
        grid_dim=(a.count, _NMIX * _SPLITS),
        block_dim=128,
    )
    ctx.enqueue_function[_fallback_gates[hidden]](
        partials,
        a.mix_scales,
        a.mix_bias,
        pre,
        a.new_post,
        a.new_comb,
        a.hc_norm_eps,
        a.hc_pre_eps,
        a.hc_post_scale,
        a.sinkhorn_eps,
        a.sinkhorn_iters,
        grid_dim=a.count,
        block_dim=WARP_SIZE,
    )
    ctx.enqueue_function[_fallback_collapse_norm[hidden]](
        a.new_residual,
        a.previous_pre if shifted else pre,
        a.norm_weight,
        a.output,
        a.rms_norm_eps,
        a.rms_scale,
        grid_dim=a.count,
        block_dim=256,
    )
