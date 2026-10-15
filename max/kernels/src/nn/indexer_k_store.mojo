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
"""Normalize, rotate and store projected index keys in a paged quantized cache."""

from std.math import ceildiv, clamp, fma, rsqrt
from std.memory import bitcast
from std.utils.index import StaticTuple
from layout import TileTensor
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.host import DeviceContext
import max.gpu.primitives.warp as warp
from linalg.fp4_utils import cast_fp32_to_fp4e2m1


@inline(.always)
def _raw[
    dtype: DType
](t: TileTensor[dtype, ...]) -> UnsafePointer[
    Scalar[dtype], MutUntrackedOrigin
]:
    # The kernel below indexes the pointer as dense row-major, so a strided
    # layout would read and write the wrong addresses.
    comptime assert t.is_row_major, "indexer_k_store tensors must be row-major"
    return (
        t.ptr.unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .address_space_cast[.GENERIC]()
    )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
def _indexer_k_store[
    fp4: Bool, ratio: Int
](
    keys: ImmPointer[SIMD[.bfloat16, 1], ImmutAnyOrigin],
    positions: ImmPointer[Int64, ImmutAnyOrigin],
    frequencies: ImmPointer[Float32, ImmutAnyOrigin],
    weight: ImmPointer[SIMD[.bfloat16, 1], ImmutAnyOrigin],
    slots: ImmPointer[Int64, ImmutAnyOrigin],
    cache: MutPointer[UInt8, MutAnyOrigin],
    count_arg: Int32,
    key_stride_arg: Int32,
    frequency_stride_arg: Int32,
    page_size_arg: Int32,
    page_stride_arg: Int32,
    epsilon: Float32,
):
    var count = Int(count_arg)
    var key_stride = Int(key_stride_arg)
    var frequency_stride = Int(frequency_stride_arg)
    var page_size = Int(page_size_arg)
    var page_stride = Int(page_stride_arg)
    var row = block_idx.x * 8 + thread_idx.x // 16
    if row >= count:
        return
    var slot = Int(slots[unsafe_offset=row])
    var position = Int(positions[unsafe_offset=row])
    if slot < 0 or (position + 1) % ratio != 0:
        return
    var lane = thread_idx.x % 16
    var channel = lane * 8
    var values = keys.unsafe_load[width=8](row * key_stride + channel).cast[
        .float32
    ]()
    var norm = rsqrt(
        warp.lane_group_sum[num_lanes=16]((values * values).reduce_add())
        / 128.0
        + epsilon
    )
    values = (
        (values * norm * weight.unsafe_load[width=8](channel).cast[.float32]())
        .cast[.bfloat16]()
        .cast[.float32]()
    )
    if channel >= 64:
        var base = (position // ratio * ratio) * frequency_stride
        comptime for pair in range(4):
            var local_pair = (channel - 64) // 2 + pair
            var cosine = frequencies[unsafe_offset=base + local_pair]
            var sine = frequencies[unsafe_offset=base + 32 + local_pair]
            var even = values[2 * pair]
            var odd = values[2 * pair + 1]
            # Preserve the reference's contraction before rounding to BF16.
            values[2 * pair] = fma(even, cosine, -(odd * sine))
            values[2 * pair + 1] = fma(odd, cosine, even * sine)
    # Both rounding points are part of the caller's BF16 contract.
    values = values.cast[.bfloat16]().cast[.float32]()
    comptime value_bytes = 64 if fp4 else 128
    var page = cache.unsafe_offset((slot // page_size) * page_stride)
    var offset = slot % page_size
    var scales = page.unsafe_offset(page_size * value_bytes + offset * 4)
    comptime if fp4:
        var peak = warp.lane_group_max[num_lanes=4](abs(values).reduce_max())
        # 6 * 2^-126, represented exactly before upward scale rounding.
        peak = max(peak, Float32(7.052966104933725e-38))
        var scale = (peak * (1.0 / 6.0)).cast[.float8_e8m0fnu]()
        var packed = cast_fp32_to_fp4e2m1(
            values * (1.0 / scale.cast[.float32]())
        )
        page.unsafe_offset(offset * 64 + lane * 4).unsafe_bitcast[UInt32]()[
            unsafe_offset=0
        ] = packed
        if lane % 4 == 0:
            scales[unsafe_offset=lane // 4] = bitcast[.uint8](scale)
    else:
        var peak = max(
            warp.lane_group_max[num_lanes=16](abs(values).reduce_max()),
            Float32(1e-4),
        )
        var scale = (
            (peak * (1.0 / 448.0)).cast[.float8_e8m0fnu]().cast[.float32]()
        )
        var quantized = clamp(values * (1.0 / scale), -448.0, 448.0).cast[
            .float8_e4m3fn
        ]()
        page.unsafe_offset(offset * 128).unsafe_store[width=8](
            lane * 8, bitcast[.uint8, 8](quantized)
        )
        if lane == 0:
            scales.unsafe_bitcast[Float32]()[unsafe_offset=0] = scale


def indexer_k_norm_rope_store[
    fp4: Bool, ratio: Int
](
    keys: TileTensor[mut=False, .bfloat16, ...],
    positions: TileTensor[mut=False, .int64, ...],
    frequencies: TileTensor[mut=False, .float32, ...],
    weight: TileTensor[mut=False, .bfloat16, ...],
    slots: TileTensor[mut=False, .int64, ...],
    cache: TileTensor[mut=True, .uint8, ...],
    page_size: Int,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """Store 128-channel index keys after RMSNorm and 64-channel GPT-J RoPE.

    Parameters:
        fp4: Select MXFP4 E2M1/UE8M0; otherwise use E4M3/FP32.
        ratio: Compression group size, either 1 or 2.

    Args:
        keys: BF16 projected keys, with 128 contiguous channels per row.
            Shape: `[count, key_stride]`, row-major.
        positions: Nonnegative token positions. Shape: `[count]`.
        frequencies: FP32 cosine then sine values, 32 each per position.
            Shape: `[rows, frequency_stride]` with 64 used per row,
            row-major.
        weight: 128 BF16 RMSNorm weights. Shape: `[128]`.
        slots: Unique destination slots; negative entries skip the row.
            Shape: `[count]`.
        cache: Byte cache pages, each page's values preceding its scales.
            Shape: `[pages, page_stride]`, row-major.
        page_size: Number of slots in a cache page.
        epsilon: RMSNorm epsilon.
        ctx: The device context.

    Constraints:
        The FP4 path requires SM100 or newer. Each page contains at least
        `page_size * (64 + 4)` bytes for FP4, or `page_size * (128 + 4)`
        for FP8. Only group-boundary rows with nonnegative slots write.
    """
    comptime assert ratio == 1 or ratio == 2
    comptime assert keys.rank == 2, "keys must be [count, key_stride]"
    comptime assert (
        frequencies.rank == 2
    ), "frequencies must be [rows, frequency_stride]"
    comptime assert cache.rank == 2, "cache must be [pages, page_stride]"
    var count = Int(keys.dim[0]())
    var key_stride = Int(keys.dim[1]())
    var frequency_stride = Int(frequencies.dim[1]())
    var page_stride = Int(cache.dim[1]())
    debug_assert(count >= 0 and key_stride >= 128 and frequency_stride >= 64)
    debug_assert(positions.num_elements() >= count)
    debug_assert(slots.num_elements() >= count)
    debug_assert(weight.num_elements() >= 128)
    debug_assert(page_size > 0 and page_stride % 4 == 0)
    debug_assert(page_stride >= page_size * (68 if fp4 else 132))
    if count == 0:
        return
    ctx.enqueue_function[_indexer_k_store[fp4, ratio]](
        _raw(keys).as_imm().as_unsafe_any_origin(),
        _raw(positions).as_imm().as_unsafe_any_origin(),
        _raw(frequencies).as_imm().as_unsafe_any_origin(),
        _raw(weight).as_imm().as_unsafe_any_origin(),
        _raw(slots).as_imm().as_unsafe_any_origin(),
        _raw(cache).as_unsafe_any_origin(),
        Int32(count),
        Int32(key_stride),
        Int32(frequency_stride),
        Int32(page_size),
        Int32(page_stride),
        epsilon,
        grid_dim=ceildiv(count, 8),
        block_dim=128,
    )
