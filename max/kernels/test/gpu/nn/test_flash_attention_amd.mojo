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
from math import isclose
from random import rand
from sys import argv

from gpu import *
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from memory import memset_zero
from nn.mha import (
    _naive_attention_with_transpose,
    flash_attention,
    mha_gpu_naive,
)
from nn.mha_mask import MaterializedMask
from nn.mha_score_mod import IdentityScoreMod
from testing import assert_almost_equal

from utils.index import Index


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


fn test[
    mask_rank: Int,
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    against_gpu_naive: Bool = False,
    batch_size: Int = 1,
    num_partitions: OptionalReg[Int] = None,
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    is_benchmark: Bool = False,
    use_index_input: Bool = False,
) raises:
    print(
        "test_flash_attention",
        "batch_size:",
        batch_size,
        "num_partitions:",
        num_partitions.value() if num_partitions else -1,
        "seq_len:",
        seq_len,
        "num_keys:",
        num_keys,
        "qkv_type:",
        qkv_type,
        "mask_type:",
        mask_type,
        "mask_rank:",
        mask_rank,
        "depth:",
        depth,
    )

    constrained[mask_rank in (3, 4), "mha only support rank 3 or 4."]()
    constrained[
        against_gpu_naive or mask_rank == 3,
        "Testing against cpu requires mask of rank 3.",
    ]()

    # Query, key, value dimensions.
    alias scale = Float32(0.25)  # rsqrt[type, 1](Float32(depth))
    alias kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = (
        (num_heads if mask_rank == 4 else 1) * seq_len * num_keys * batch_size
    )

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(mask_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Q, K, V are randomly initialized.
    if use_index_input:
        debug_assert(batch_size == 1)
        for i in range(seq_len):
            for h in range(num_heads):
                for j in range(depth):
                    q_ptr[(i * num_heads + h) * depth + j] = i * depth + j
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    k_ptr[(i * kv_num_heads + h) * depth + j] = i * depth + j
        for i in range(num_keys):
            for h in range(kv_num_heads):
                for j in range(depth):
                    v_ptr[(i * kv_num_heads + h) * depth + j] = i * depth + j

    else:
        rand[qkv_type](q_ptr, q_size)
        rand[qkv_type](k_ptr, k_size)
        rand[qkv_type](v_ptr, v_size)

    memset_zero(mask_ptr, mask_size)
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
    var mask = LayoutTensor[mask_type, Layout.row_major[2]()](
        mask_ptr,
        RuntimeLayout[Layout.row_major[2]()].row_major(
            Index(seq_len, num_keys)
        ),
    )
    var output = LayoutTensor[qkv_type, layout_4d](
        output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    var flash_output = LayoutTensor[qkv_type, layout_4d](
        flash_output_ptr,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, seq_len, num_heads, depth)
        ),
    )

    @parameter
    if not against_gpu_naive:
        constrained[
            qkv_type == mask_type, "expect qkv and mask have same type for CPU."
        ]()
        _naive_attention_with_transpose[qkv_type](
            output,
            q,
            k,
            v,
            mask.bitcast[qkv_type](),
            scale,
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
    var mask3d = LayoutTensor[mask_type, Layout.row_major[3]()](
        mask_device_ptr.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[3]()].row_major(
            Index(batch_size, seq_len, num_keys)
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

    alias q_tile_num_rows = 32
    alias k_tile_num_rows = 128

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, v_device, mask3d, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        @parameter
        if mask_rank == 3:
            flash_attention(
                output_device,
                q_device,
                k_device,
                v_device,
                MaterializedMask(mask3d),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )
        else:
            flash_attention(
                output_device,
                q_device,
                k_device,
                v_device,
                MaterializedMask(mask4d),
                IdentityScoreMod(),
                scale,
                ctx,
                num_partitions,
            )

    if is_benchmark:
        alias nrun = 50

        # Warmup
        kernel_launch(ctx)

        var nstime = ctx.execution_time[kernel_launch](nrun) / nrun
        var sectime = nstime / 1000000
        print(nrun, "runs avg", sectime, "ms")

    else:
        kernel_launch(ctx)

    ctx.synchronize()

    ctx.enqueue_copy(flash_output_ptr, output_device_ptr)

    @parameter
    if against_gpu_naive:
        var output_ref_device_ptr = ctx.enqueue_create_buffer[qkv_type](o_size)
        alias output_ref_layout = Layout.row_major(
            UNKNOWN_VALUE, UNKNOWN_VALUE, num_heads, depth
        )
        var output_ref_device = LayoutTensor[qkv_type, output_ref_layout](
            output_ref_device_ptr.unsafe_ptr(),
            RuntimeLayout[output_ref_layout].row_major(
                Index(batch_size, seq_len, num_heads, depth)
            ),
        )
        ctx.enqueue_copy(output_ref_device_ptr, output_ptr)

        @parameter
        if mask_rank == 3:
            mha_gpu_naive(
                q_device,
                k_device,
                v_device,
                mask3d,
                output_ref_device,
                scale,
                batch_size,
                seq_len,
                num_keys,
                num_heads,
                depth,
                group,
                ctx,
            )
        elif mask_rank == 4:
            mha_gpu_naive(
                q_device,
                k_device,
                v_device,
                mask4d,
                output_ref_device,
                scale,
                batch_size,
                seq_len,
                num_keys,
                num_heads,
                depth,
                group,
                ctx,
            )

        ctx.synchronize()
        ctx.enqueue_copy(output_ptr, output_ref_device_ptr)
        _ = output_ref_device_ptr

    # This is useful for debugging.

    var rtol = 2e-2
    for bs in range(batch_size):
        for h in range(num_heads):
            for s in range(seq_len):
                for d in range(depth):
                    var actual = flash_output[bs, s, Int(h), Int(d)][0]
                    var expect = output[bs, s, Int(h), Int(d)][0]
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


fn test_context_encoding[
    batch_size: Int, depth: Int
](ctx: DeviceContext) raises:
    # # fp32 arbitrary depth and num_heads, baseline impl.
    # test[3, DType.float32, DType.float32, 127, 2](111, 121, ctx)
    # # fp32 depth == 128, tf32-fp32 mma, llama2 shape.
    # test[
    #     4,
    #     DType.float32,
    #     DType.float32,
    #     128,
    #     32,
    #     against_gpu_naive=True,
    # ](1024, 1024, ctx, is_benchmark())
    # test[
    #     3,
    #     DType.float32,
    #     DType.float32,
    #     128,
    #     3,
    #     against_gpu_naive=True,
    # ](14, 14, ctx, is_benchmark())
    # test[
    #     3,
    #     DType.float32,
    #     DType.float32,
    #     128,
    #     1,
    #     against_gpu_naive=True,
    # ](178, 178, ctx, is_benchmark())
    # bf16 depth == 128, bf16-fp32 mma
    test[
        4,
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=1,
        against_gpu_naive=True,
    ](128, 128, ctx, use_index_input=False)
    test[
        4,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=1,
        against_gpu_naive=True,
    ](384, 384, ctx)
    test[
        3,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=3,
        against_gpu_naive=True,
    ](256, 256, ctx)
    test[
        4,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=32,
        against_gpu_naive=True,
    ](1024, 1024, ctx, is_benchmark())
    test[
        4,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=24,
        group=3,
        against_gpu_naive=True,
    ](1024, 1024, ctx)
    # BF16 with sequence length not multiple of 128
    test[
        4,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=3,
        group=3,
        against_gpu_naive=True,
    ](64, 64, ctx)
    test[
        4,
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=3,
        group=3,
        against_gpu_naive=True,
    ](102, 102, ctx)
    test[
        3,
        DType.bfloat16,
        DType.float32,
        depth=depth,
        num_heads=1,
        against_gpu_naive=True,
    ](14, 14, ctx)
    test[
        3,
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=1,
        against_gpu_naive=True,
    ](528, 528, ctx)

    test[
        4,
        DType.bfloat16,
        DType.bfloat16,
        depth=depth,
        num_heads=32,
        group=4,
        against_gpu_naive=True,
    ](1, 1, ctx)


fn test_decoding[
    batch_size: Int,
    depth: Int,
    num_partitions: OptionalReg[Int] = None,
    qkv_type: DType = DType.bfloat16,
](ctx: DeviceContext, use_index_input: Bool) raises:
    # fp32 arbitrary depth and num_heads, baseline impl.
    # BF16 token gen
    test[
        3,
        qkv_type,
        DType.float32,
        depth=depth,
        num_heads=2,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 512, ctx, use_index_input=use_index_input)
    test[
        4,
        qkv_type,
        DType.bfloat16,
        depth=depth,
        num_heads=2,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 512, ctx, use_index_input=use_index_input)
    test[
        3,
        qkv_type,
        DType.float32,
        depth=depth,
        num_heads=1,
        group=1,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 128, ctx, use_index_input=use_index_input)
    test[
        4,
        qkv_type,
        DType.bfloat16,
        depth=depth,
        num_heads=3,
        group=3,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 156, ctx, use_index_input=use_index_input)
    test[
        4,
        qkv_type,
        DType.bfloat16,
        depth=depth,
        num_heads=3,
        group=3,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 208, ctx, use_index_input=use_index_input)

    test[
        4,
        qkv_type,
        DType.bfloat16,
        depth=depth,
        num_heads=32,
        group=4,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 1, ctx, use_index_input=use_index_input)
    test[
        3,
        qkv_type,
        DType.float32,
        depth=depth,
        num_heads=2,
        against_gpu_naive=True,
        batch_size=batch_size,
        num_partitions=num_partitions,
    ](1, 5120, ctx, use_index_input=use_index_input)


def main():
    with DeviceContext() as ctx:
        alias depths = (64, 128, 256)

        @parameter
        for i in range(len(depths)):
            alias depth = depths[i]
            test_context_encoding[1, depth](ctx)

            @parameter
            for batch_size in range(1, 5, 3):
                test_decoding[batch_size, depth, 1](ctx, False)
                test_decoding[batch_size, depth, 1, DType.float32](ctx, False)
            test_decoding[1, depth, None](ctx, False)
