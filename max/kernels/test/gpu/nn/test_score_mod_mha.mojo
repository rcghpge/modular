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

from collections import Set
from math import exp2, iota, isqrt
from random import random_ui64, seed

from bit import prev_power_of_two
from buffer import Dim, DimList, NDBuffer
from gpu.host import DeviceContext
from internal_utils import DeviceNDBuffer, HostNDBuffer, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from nn.mha import flash_attention
from nn.mha_mask import CausalMask, MaterializedMask
from nn.mha_score_mod import AlibiScoreMod, IdentityScoreMod
from tensor_internal import IOUnknown, ManagedTensorSlice
from tensor_internal.managed_tensor_slice import StaticTensorSpec
from testing import assert_almost_equal

from utils import Index, IndexList
from utils.numerics import min_or_neg_inf

alias kv_params_replit = KVCacheStaticParams(num_heads=8, head_size=128)
alias replit_num_q_heads = 24

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


fn generate_alibi_bias[
    dtype: DType,
    width: Int,
    num_heads: Int,
](
    head_idx: SIMD[DType.int, width],
    q_idx: SIMD[DType.int, width],
    k_idx: SIMD[DType.int, width],
    max_prompt_len: Int = 0,
) -> SIMD[dtype, width]:
    var scale: SIMD[dtype, width]

    @parameter
    if num_heads.is_power_of_two():
        scale = exp2(-((head_idx + 1).cast[dtype]() * 8.0 / num_heads))
    else:
        alias floor_power_of_2 = prev_power_of_two(num_heads)
        if head_idx < floor_power_of_2:
            scale = exp2(
                -((head_idx + 1).cast[dtype]() * 8.0 / floor_power_of_2)
            )
        else:
            scale = exp2(
                -(
                    ((head_idx - floor_power_of_2) * 2 + 1).cast[dtype]()
                    * 8.0
                    / (floor_power_of_2 * 2)
                )
            )
    var bias = (
        -(max_prompt_len - 1 - k_idx - iota[DType.int, width]()).cast[dtype]()
        * scale
    )
    return bias


def execute_flash_attention[
    num_q_heads: Int,
    dtype: DType,
    kv_params: KVCacheStaticParams,
](
    batch_size: Int,
    valid_length: NDBuffer[DType.uint32, 1],
    max_seq_len: Int,
    cache_valid_length: NDBuffer[DType.uint32, 1],
    ctx: DeviceContext,
):
    alias num_blocks = 32
    alias CollectionType = ContinuousBatchingKVCacheCollection[dtype, kv_params]

    debug_assert(
        batch_size < num_blocks,
        "batch_size passed to unit test (",
        batch_size,
        ") is larger than configured num_blocks (",
        num_blocks,
        ")",
    )

    # initialize our KVCache
    max_prompt_len = 0
    max_context_len = 0

    for i in range(batch_size):
        max_prompt_len = max(max_prompt_len, Int(valid_length[i]))
        max_context_len = max(
            max_context_len, Int(cache_valid_length[i] + valid_length[i])
        )

    var cache_lengths_dev = ctx.enqueue_create_buffer[DType.uint32](batch_size)

    ctx.enqueue_copy(cache_lengths_dev, cache_valid_length.data)
    var cache_lengths = NDBuffer[DType.uint32, 1](
        cache_lengths_dev.unsafe_ptr(), Index(batch_size)
    )

    # initialize q tensor
    # TODO parameterize to layout
    q_host = HostNDBuffer[
        dtype, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        )
    )

    random(q_host.tensor)

    valid_length_device = DeviceNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size),
        ctx=ctx,
    )
    ctx.enqueue_copy(valid_length_device.buffer, valid_length.data)

    q_device = DeviceNDBuffer[
        dtype, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
        ctx=ctx,
    )
    ctx.enqueue_copy(q_device.buffer, q_host.tensor.data)

    # initialize mask tensor
    mask_host = HostNDBuffer[
        DType.float32, 4, DimList(Dim(), num_q_heads, Dim(), Dim())
    ](IndexList[4](batch_size, num_q_heads, max_prompt_len, max_context_len))

    # Initialize causal mask.
    for b in range(batch_size):
        for h in range(num_q_heads):
            for q_idx in range(max_prompt_len):
                for k_idx in range(max_context_len):
                    mask_host.tensor.store(
                        Index(b, h, q_idx, k_idx),
                        0 if q_idx + cache_valid_length[b]
                        >= k_idx else min_or_neg_inf[DType.float32](),
                    )

    # initialize mask tensor
    mask_host_mod = HostNDBuffer[
        DType.float32, 4, DimList(Dim(), num_q_heads, Dim(), Dim())
    ](IndexList[4](batch_size, num_q_heads, max_prompt_len, max_context_len))

    # Initialize causal mask with a bias for when q_idx >= k_idx.
    # In this case this is the alibi as added bias.
    # This is used to compare against the score_mod implementation.
    for b in range(batch_size):
        for h in range(num_q_heads):
            for q_idx in range(max_prompt_len):
                for k_idx in range(max_context_len):
                    mask_host_mod.tensor.store(
                        Index(b, h, q_idx, k_idx),
                        generate_alibi_bias[DType.float32, 1, num_q_heads](
                            h,
                            q_idx,
                            k_idx,
                            max_context_len,
                        ) if q_idx
                        + cache_valid_length[b]
                        >= k_idx else min_or_neg_inf[DType.float32](),
                    )

    mask_device = DeviceNDBuffer[
        DType.float32, 4, DimList(Dim(), num_q_heads, Dim(), Dim())
    ](
        IndexList[4](batch_size, num_q_heads, max_prompt_len, max_context_len),
        ctx=ctx,
    )
    ctx.enqueue_copy(mask_device.buffer, mask_host.tensor.data)

    mask_device_mod = DeviceNDBuffer[
        DType.float32, 4, DimList(Dim(), num_q_heads, Dim(), Dim())
    ](
        IndexList[4](batch_size, num_q_heads, max_prompt_len, max_context_len),
        ctx=ctx,
    )
    ctx.enqueue_copy(mask_device_mod.buffer, mask_host_mod.tensor.data)

    # initialize reference output
    ref_output_host = HostNDBuffer[
        dtype, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
    )
    ref_output_device = DeviceNDBuffer[
        dtype, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
        ctx=ctx,
    )

    # initialize test output
    test_output_host = HostNDBuffer[
        dtype, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
    )
    test_output_device = DeviceNDBuffer[
        dtype, 4, DimList(Dim(), Dim(), num_q_heads, kv_params.head_size)
    ](
        IndexList[4](
            batch_size, max_prompt_len, num_q_heads, kv_params.head_size
        ),
        ctx=ctx,
    )

    kv_block_host = HostNDBuffer[dtype, 6](
        IndexList[6](
            num_blocks,
            2,
            1,
            max_seq_len,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )
    random(kv_block_host.tensor)
    kv_block_device = kv_block_host.copy_to_device(ctx)

    var lookup_table_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size),
    )

    # hacky way to select random blocks.
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = Int(random_ui64(0, num_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table_host.tensor[idx] = UInt32(randval)
        idx += 1

    var lookup_table_device = lookup_table_host.copy_to_device(ctx)

    kv_collection_device = CollectionType(
        kv_block_device.tensor,
        cache_lengths,
        lookup_table_device.tensor,
        max_prompt_len,
        max_context_len,
    )

    k_cache_device = kv_collection_device.get_key_cache(0)
    v_cache_device = kv_collection_device.get_value_cache(0)

    flash_attention[use_score_mod=True](
        test_output_device.tensor,
        q_device.tensor,
        k_cache_device,
        v_cache_device,
        CausalMask(),
        AlibiScoreMod[num_q_heads](),
        ManagedTensorSlice[
            io_spec=IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ](valid_length_device.tensor),
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )

    # Here pass mask that includes bias in q_idx >= k_idx (to compare).
    flash_attention(
        ref_output_device.tensor,
        q_device.tensor,
        k_cache_device,
        v_cache_device,
        MaterializedMask(mask_device_mod.tensor, start_pos=cache_lengths),
        IdentityScoreMod(),
        ManagedTensorSlice[
            io_spec=IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ](valid_length_device.tensor),
        isqrt(Float32(kv_params.head_size)),
        ctx,
    )

    ctx.enqueue_copy(test_output_host.tensor.data, test_output_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    var ref_out = ref_output_host.tensor
    var test_out = test_output_host.tensor
    for bs in range(batch_size):
        for s in range(valid_length[bs]):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    var expect = ref_out[Index(bs, s, Int(h), Int(hd))]
                    var actual = test_out[Index(bs, s, Int(h), Int(hd))]
                    assert_almost_equal(
                        expect,
                        actual,
                        atol=1e-5,
                        rtol=8e-3,
                    )

    _ = q_device^
    _ = q_host^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = test_output_device^
    _ = test_output_host^
    _ = kv_block_device^
    _ = kv_block_host^
    _ = lookup_table_host^
    _ = lookup_table_device^
    _ = mask_host^
    _ = mask_host_mod^
    _ = mask_device^
    _ = mask_device_mod^
    _ = cache_lengths_dev^
    _ = valid_length_device^
    _ = valid_length


def execute_flash_attention_suite(ctx: DeviceContext):
    var bs = 2
    var valid_length_ptr = UnsafePointer[UInt32].alloc(bs)
    var valid_length = NDBuffer[DType.uint32, 1](valid_length_ptr, Index(1))

    var cache_valid_length_ptr = UnsafePointer[UInt32].alloc(bs)
    var cache_valid_length = NDBuffer[DType.uint32, 1](
        cache_valid_length_ptr, Index(1)
    )

    alias dtype = DType.bfloat16

    # Replit & Llama3 context encoding [testing even query valid lengths].
    valid_length[0] = 128
    valid_length[1] = 64
    cache_valid_length[0] = 0
    cache_valid_length[1] = 0

    execute_flash_attention[
        replit_num_q_heads,
        dtype,
        kv_params_replit,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    execute_flash_attention[
        llama_num_q_heads,
        dtype,
        kv_params_llama3,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    # Replit & Llama3 context encoding [testing odd query valid length].
    valid_length[0] = 128
    valid_length[1] = 65
    cache_valid_length[0] = 0
    cache_valid_length[1] = 0

    execute_flash_attention[
        replit_num_q_heads,
        dtype,
        kv_params_replit,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    execute_flash_attention[
        llama_num_q_heads,
        dtype,
        kv_params_llama3,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    # Replit & Llama3 token gen [testing even cache valid lengths].
    valid_length[0] = 1
    valid_length[1] = 1
    cache_valid_length[0] = 200
    cache_valid_length[1] = 256

    execute_flash_attention[
        replit_num_q_heads,
        dtype,
        kv_params_replit,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    execute_flash_attention[
        llama_num_q_heads,
        dtype,
        kv_params_llama3,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    # Replit & Llama3 token gen [testing even cache valid lengths].
    valid_length[0] = 1
    valid_length[1] = 1
    cache_valid_length[0] = 200
    cache_valid_length[1] = 255

    execute_flash_attention[
        replit_num_q_heads,
        dtype,
        kv_params_replit,
    ](bs, valid_length, 1024, cache_valid_length, ctx)

    execute_flash_attention[
        llama_num_q_heads,
        dtype,
        kv_params_llama3,
    ](bs, valid_length, 1024, cache_valid_length, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)

    print("Success!")
