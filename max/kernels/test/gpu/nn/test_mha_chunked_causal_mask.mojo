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

from math import isclose
from random import rand

from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from nn.mha import flash_attention
from nn.mha_mask import (
    MASK_VALUE,
    ChunkedCausalMask,
    MaterializedMask,
    TileMaskStatus,
)
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal, assert_equal

from utils.index import Index


def build_ChunkedCausalMask[
    local_window_size: Int, mask_type: DType
](
    batch_size: Int,
    num_heads: Int,
    seq_len: Int,
    num_keys: Int,
    mask: LayoutTensor[mut=True, mask_type, **_],
):
    # Initialize causal mask.
    for b in range(batch_size):
        for h in range(num_heads):
            for q_idx in range(seq_len):
                for k_idx in range(num_keys):
                    start_pos = num_keys - seq_len
                    var q_chunk_idx = (q_idx + start_pos) // local_window_size
                    var k_chunk_idx = k_idx // local_window_size
                    var chunk_masked = q_chunk_idx != k_chunk_idx
                    var causal_masked = q_idx + start_pos < k_idx
                    var masked = chunk_masked or causal_masked
                    mask.store(
                        Index(b, h, q_idx, k_idx),
                        Scalar[mask.dtype](0 if not masked else MASK_VALUE),
                    )


fn test_attention[
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    local_window_size: Int = 256,
    group: Int = 1,
](seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    print("test_mha_chunked_causal_mask")
    print(
        "qkv_type:",
        qkv_type,
        "mask_type:",
        mask_type,
        "depth:",
        depth,
        "num_heads:",
        num_heads,
        "group:",
        group,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
    )
    # Query, key, value dimensions.
    alias batch_size = 1
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    alias kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = batch_size * num_heads * seq_len * num_keys

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(mask_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Construct buffers.
    alias layout_4d = Layout.row_major[4]()
    var q = LayoutTensor[qkv_type, layout_4d](
        q_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    var k = LayoutTensor[qkv_type, layout_4d](
        k_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    var v = LayoutTensor[qkv_type, layout_4d](
        v_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    var mask = LayoutTensor[mask_type, layout_4d, MutableAnyOrigin](
        mask_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_heads, seq_len, num_keys)
        ),
    )
    var output = LayoutTensor[qkv_type, layout_4d](
        output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    # Q, K, V are randomly initialized.
    rand[qkv_type](q_ptr, q_size, min=-1.0, max=1.0)
    rand[qkv_type](k_ptr, k_size, min=-1.0, max=1.0)
    rand[qkv_type](v_ptr, v_size, min=-1.0, max=1.0)

    # Initialize causal mask.
    build_ChunkedCausalMask[local_window_size](
        batch_size, num_heads, seq_len, num_keys, mask
    )

    # Device pointers
    var q_device_ptr = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.enqueue_create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.enqueue_create_buffer[qkv_type](v_size)
    var mask_device_ptr = ctx.enqueue_create_buffer[mask_type](mask_size)
    var output_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy(q_device_ptr, q_ptr)
    ctx.enqueue_copy(k_device_ptr, k_ptr)
    ctx.enqueue_copy(v_device_ptr, v_ptr)
    ctx.enqueue_copy(mask_device_ptr, mask_ptr)

    # Construct device buffers.
    alias q_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )
    var q_device = LayoutTensor[qkv_type, q_layout](
        q_device_ptr.unsafe_ptr(),
        RuntimeLayout[q_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    alias k_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth
    )
    var k_device = LayoutTensor[qkv_type, k_layout](
        k_device_ptr.unsafe_ptr(),
        RuntimeLayout[k_layout].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    alias v_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, kv_num_heads, depth
    )
    var v_device = LayoutTensor[qkv_type, v_layout](
        v_device_ptr.unsafe_ptr(),
        RuntimeLayout[v_layout].row_major(
            Index(batch_size, num_keys, kv_num_heads, depth)
        ),
    )
    var mask4d = LayoutTensor[mask_type, Layout.row_major[4]()](
        mask_device_ptr.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[4]()].row_major(
            Index(batch_size, num_heads, seq_len, num_keys)
        ),
    )
    alias output_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )
    var output_device = LayoutTensor[qkv_type, output_layout](
        output_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    flash_attention(
        output_device,
        q_device,
        k_device,
        v_device,
        ChunkedCausalMask[local_window_size](),
        IdentityScoreMod(),
        scale,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
    ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

    alias output_ref_layout = Layout.row_major(
        UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
    )
    var output_device_ref = LayoutTensor[qkv_type, output_ref_layout](
        output_ref_device_ptr.unsafe_ptr(),
        RuntimeLayout[output_ref_layout].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )
    flash_attention(
        output_device_ref,
        q_device,
        k_device,
        v_device,
        MaterializedMask(mask4d),
        IdentityScoreMod(),
        scale,
        ctx,
    )

    ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
    ctx.synchronize()
    _ = output_ref_device_ptr

    var rtol = 1e-2
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                var expect = output_ptr.load(
                    d + depth * (h + s * num_heads)
                ).cast[DType.float64]()
                var actual = flash_output_ptr.load(
                    d + depth * (h + s * num_heads)
                ).cast[DType.float64]()
                if not isclose(actual, expect, atol=1e-5, rtol=rtol):
                    var rerr = abs((actual - expect) / expect)
                    print(h, s, d, actual, expect, rerr)
                assert_almost_equal(actual, expect, atol=1e-5, rtol=rtol)

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = mask_device_ptr
    _ = output_device_ptr

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()


def test_attention_suite(ctx: DeviceContext):
    alias types = (DType.bfloat16, DType.float32)

    @parameter
    for type_idx in range(len(types)):
        alias type = types[type_idx]
        # context encoding
        test_attention[
            type,
            type,
            depth=128,
            num_heads=1,
        ](14, 14, ctx)

        test_attention[
            type,
            type,
            depth=128,
            num_heads=1,
        ](128, 128, ctx)

        test_attention[
            type,
            DType.float32,
            depth=128,
            num_heads=1,
        ](384, 384, ctx)

        # token gen
        test_attention[
            type,
            DType.float32,
            128,
            32,
        ](1, 512, ctx)

        test_attention[
            type,
            DType.float32,
            128,
            11,
        ](1, 256, ctx)

        test_attention[
            type,
            DType.float32,
            128,
            1,
        ](1, 11, ctx)


def test_mask_status():
    var mask = ChunkedCausalMask[local_window_size=4]()

    assert_equal(
        mask.status(Index(0, 0), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(4, 4), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(2, 2), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(0, 2), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(2, 0), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )

    assert_equal(
        mask.status(Index(0, 4), Index(4, 4)), TileMaskStatus.FULL_MASK
    )
    assert_equal(
        mask.status(Index(4, 0), Index(4, 4)), TileMaskStatus.FULL_MASK
    )

    # cases where tile_size >> local_window_size
    assert_equal(
        mask.status(Index(100, 0), Index(128, 128)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(0, 0), Index(100, 100)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(50, 0), Index(100, 100)), TileMaskStatus.PARTIAL_MASK
    )

    var bigger_mask = ChunkedCausalMask[local_window_size=256]()
    assert_equal(
        bigger_mask.status(Index(256, 256), Index(128, 128)),
        TileMaskStatus.PARTIAL_MASK,
    )
    assert_equal(
        bigger_mask.status(Index(128, 0), Index(128, 128)),
        TileMaskStatus.NO_MASK,
    )
    assert_equal(
        bigger_mask.status(Index(256, 0), Index(128, 128)),
        TileMaskStatus.FULL_MASK,
    )


def test_mask_apply():
    alias local_window_size = 4
    var mask = ChunkedCausalMask[local_window_size]()

    var score_vec = SIMD[DType.float32, 4](0.0)
    score_vec[0] = 1.0
    score_vec[1] = 2.0
    score_vec[2] = 3.0
    score_vec[3] = 4.0

    alias simd_width = 4
    alias SIMD_T = SIMD[DType.float32, simd_width]
    alias UNMASKED_INPUT = SIMD_T(0.0)
    var inf_vec = SIMD_T(MASK_VALUE)

    # first two dims should be arbitrary, we pass in junk just to help confirm.
    assert_equal(
        mask.mask(Index(0, 0, 0, 0), score_vec),
        SIMD_T(1.0, MASK_VALUE, MASK_VALUE, MASK_VALUE),
    )
    assert_equal(mask.mask(Index(10, 0, 4, 8), score_vec), inf_vec)
    assert_equal(
        mask.mask(Index(2, 10, 8, 8), score_vec),
        SIMD_T(1.0, MASK_VALUE, MASK_VALUE, MASK_VALUE),
    )

    assert_equal(
        mask.mask(Index(0, 4, 8, 10), score_vec),
        inf_vec,
    )
    assert_equal(
        mask.mask(Index(4, 0, 13, 10), score_vec),
        SIMD_T(MASK_VALUE, MASK_VALUE, 3.0, 4.0),
    )
    assert_equal(
        mask.mask(Index(4, 0, 12, 10), score_vec),
        SIMD_T(MASK_VALUE, MASK_VALUE, 3.0, MASK_VALUE),
    )
    assert_equal(
        mask.mask(Index(1000, 1000, 14, 12), score_vec),
        SIMD_T(1.0, 2.0, 3.0, MASK_VALUE),
    )


def main():
    test_mask_status()
    test_mask_apply()
    with DeviceContext() as ctx:
        test_attention_suite(ctx)
