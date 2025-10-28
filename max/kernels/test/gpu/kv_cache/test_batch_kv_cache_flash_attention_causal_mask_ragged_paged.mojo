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
from math import ceildiv, rsqrt
from random import random_ui64, seed

from buffer import Dim, DimList
from gpu.host import DeviceContext
from internal_utils import HostNDBuffer, random
from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
    PagedKVCacheCollection,
)
from memory import memcpy, memset_zero
from nn.mha import flash_attention
from nn.mha_mask import CausalMask
from nn.mha_score_mod import IdentityScoreMod
from tensor import IOUnknown, ManagedTensorSlice
from tensor.managed_tensor_slice import StaticTensorSpec
from testing import assert_almost_equal

from utils import IndexList

alias kv_params_llama3 = KVCacheStaticParams(num_heads=8, head_size=128)
alias llama_num_q_heads = 32


def execute_ragged_flash_attention[
    num_q_heads: Int, dtype: DType, kv_params: KVCacheStaticParams
](
    valid_lengths: List[Int],
    cache_lengths: List[Int],
    num_layers: Int,
    layer_idx: Int,
    ctx: DeviceContext,
):
    alias page_size = 512

    var batch_size = len(valid_lengths)
    debug_assert(
        len(valid_lengths) == len(cache_lengths),
        "expected valid_lengths and cache_lengths size to be equal",
    )

    var input_row_offsets_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var cache_lengths_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var total_length = 0
    var max_full_context_length = 0
    var max_prompt_length = 0
    for i in range(batch_size):
        input_row_offsets_host.tensor[i] = total_length
        cache_lengths_host.tensor[i] = cache_lengths[i]
        max_full_context_length = max(
            max_full_context_length, cache_lengths[i] + valid_lengths[i]
        )
        max_prompt_length = max(max_prompt_length, valid_lengths[i])
        total_length += valid_lengths[i]
    input_row_offsets_host.tensor[batch_size] = total_length

    input_row_offsets_device = input_row_offsets_host.copy_to_device(ctx)
    cache_lengths_device = cache_lengths_host.copy_to_device(ctx)

    q_ragged_host = HostNDBuffer[
        dtype, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    random(q_ragged_host.tensor)
    q_ragged_device = q_ragged_host.copy_to_device(ctx)

    # initialize reference output
    test_output_host = HostNDBuffer[
        dtype, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    test_output_device = test_output_host.copy_to_device(ctx)
    ref_output_host = HostNDBuffer[
        dtype, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](total_length, num_q_heads, kv_params.head_size))
    ref_output_device = ref_output_host.copy_to_device(ctx)

    var num_continuous_blocks = batch_size + 2
    var num_paged_blocks = (
        ceildiv(max_full_context_length, page_size) * batch_size
    )

    # initialize our KVCache
    kv_block_continuous_host = HostNDBuffer[dtype, 6](
        IndexList[6](
            num_continuous_blocks,
            2,
            num_layers,
            max_full_context_length,
            kv_params.num_heads,
            kv_params.head_size,
        ),
    )

    random(kv_block_continuous_host.tensor)
    kv_block_continuous_device = kv_block_continuous_host.copy_to_device(ctx)
    var lookup_table_continuous_host = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    # hacky way to select random blocks for continuous batching
    var block_idx_set = Set[Int]()
    var idx = 0
    while idx < batch_size:
        var randval = Int(random_ui64(0, num_continuous_blocks - 1))
        if randval in block_idx_set:
            continue

        block_idx_set.add(randval)
        lookup_table_continuous_host.tensor[idx] = UInt32(randval)
        idx += 1
    var lookup_table_device = lookup_table_continuous_host.copy_to_device(ctx)

    kv_collection_continuous_device = ContinuousBatchingKVCacheCollection[
        dtype, kv_params
    ](
        kv_block_continuous_device.tensor,
        cache_lengths_device.tensor,
        lookup_table_device.tensor,
        max_prompt_length,
        max_full_context_length,
    )

    # num_paged_blocks = 1,
    # num_layers = 2
    # page_size = 512
    # (total_blocks - 1) * self._stride() + Self.page_size
    # Self.page_size
    kv_block_paged_host = HostNDBuffer[dtype, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )

    paged_lut_host = HostNDBuffer[DType.uint32, 2](
        IndexList[2](batch_size, ceildiv(max_full_context_length, page_size))
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = cache_lengths[bs] + valid_lengths[bs]
        continuous_idx = Int(lookup_table_continuous_host.tensor[bs])

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut_host.tensor[bs, block_idx] = randval
            block_sz = min(page_size, seq_len - block_idx * page_size)

            for kv_idx in range(2):
                paged_ptr = kv_block_paged_host.tensor._offset(
                    IndexList[6](randval, kv_idx, layer_idx, 0, 0, 0)
                )
                n_cpy = block_sz * kv_params.num_heads * kv_params.head_size
                memcpy(
                    dest=paged_ptr,
                    src=kv_block_continuous_host.tensor._offset(
                        IndexList[6](
                            continuous_idx,
                            kv_idx,
                            layer_idx,
                            block_idx * page_size,
                            0,
                            0,
                        )
                    ),
                    count=n_cpy,
                )
                if block_sz < page_size:
                    memset_zero(
                        paged_ptr + n_cpy,
                        (page_size - block_sz)
                        * kv_params.num_heads
                        * kv_params.head_size,
                    )

    paged_lut_device = paged_lut_host.copy_to_device(ctx)
    kv_block_paged_device = kv_block_paged_host.copy_to_device(ctx)

    kv_collection_paged_device = PagedKVCacheCollection[
        dtype, kv_params, page_size
    ](
        kv_block_paged_device.tensor,
        cache_lengths_device.tensor,
        paged_lut_device.tensor,
        max_prompt_length,
        max_full_context_length,
    )

    # continuous execution
    flash_attention[ragged=True](
        ref_output_device.to_layout_tensor(),
        q_ragged_device.to_layout_tensor(),
        kv_collection_continuous_device.get_key_cache(layer_idx),
        kv_collection_continuous_device.get_value_cache(layer_idx),
        CausalMask(),
        IdentityScoreMod(),
        ManagedTensorSlice[
            io_spec=IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ](input_row_offsets_device.tensor),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )

    # paged execution
    flash_attention[ragged=True](
        test_output_device.to_layout_tensor(),
        q_ragged_device.to_layout_tensor(),
        kv_collection_paged_device.get_key_cache(layer_idx),
        kv_collection_paged_device.get_value_cache(layer_idx),
        CausalMask(),
        IdentityScoreMod(),
        ManagedTensorSlice[
            io_spec=IOUnknown,
            static_spec = StaticTensorSpec[DType.uint32, 1].create_unknown(),
        ](input_row_offsets_device.tensor),
        rsqrt(Float32(kv_params.head_size)),
        ctx,
    )
    ctx.enqueue_copy(test_output_host.tensor.data, test_output_device.buffer)
    ctx.enqueue_copy(ref_output_host.tensor.data, ref_output_device.buffer)
    ctx.synchronize()

    ref_out = ref_output_host.tensor
    test_out = test_output_host.tensor

    for bs in range(batch_size):
        prompt_len = valid_lengths[bs]
        ragged_offset = Int(input_row_offsets_host.tensor[bs])
        for s in range(prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            ref_out[ragged_offset + s, h, hd],
                            test_out[ragged_offset + s, h, hd],
                            atol=1e-2,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            ref_out[ragged_offset + s, h, hd],
                            test_out[ragged_offset + s, h, hd],
                        )
                        raise e

    _ = q_ragged_host^
    _ = q_ragged_device^
    _ = kv_block_continuous_host^
    _ = kv_block_continuous_device^
    _ = kv_block_paged_host^
    _ = kv_block_paged_device^
    _ = lookup_table_continuous_host^
    _ = lookup_table_device^
    _ = ref_output_device^
    _ = ref_output_host^
    _ = test_output_device^
    _ = test_output_host^
    _ = cache_lengths_host^
    _ = cache_lengths_device^
    _ = paged_lut_host^
    _ = paged_lut_device^


def execute_flash_attention_suite(ctx: DeviceContext):
    alias types = (DType.float32, DType.bfloat16)

    for bs in [1, 4]:

        @parameter
        for type_idx in range(len(types)):
            alias type = types[type_idx]
            ce_cache_sizes = List[Int]()
            ce_seq_lens = List[Int]()
            tg_cache_sizes = List[Int]()
            tg_seq_lens = List[Int]()
            for _ in range(bs):
                tg_seq_lens.append(1)
                tg_cache_sizes.append(Int(random_ui64(512, 1024)))
                ce_seq_lens.append(Int(random_ui64(512, 1024)))
                ce_cache_sizes.append(0)

            print("CE", bs, type)
            execute_ragged_flash_attention[
                llama_num_q_heads, type, kv_params_llama3
            ](ce_seq_lens, ce_cache_sizes, 2, 1, ctx)

            print("TG", bs, type)
            execute_ragged_flash_attention[
                llama_num_q_heads, type, kv_params_llama3
            ](tg_seq_lens, tg_cache_sizes, 2, 0, ctx)

    # edge cases
    print("CE", 1, DType.bfloat16)
    var short_ce_seq_len = [2]
    var short_ce_cache_size = [0]
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3
    ](short_ce_seq_len, short_ce_cache_size, 2, 1, ctx)

    print("TG", 2, DType.bfloat16)
    tg_seq_lens = [1, 1]
    tg_variable_cache_lens = [1024, 11]
    execute_ragged_flash_attention[
        llama_num_q_heads, DType.bfloat16, kv_params_llama3
    ](tg_seq_lens, tg_variable_cache_lens, 2, 0, ctx)


def main():
    seed(42)
    with DeviceContext() as ctx:
        execute_flash_attention_suite(ctx)
