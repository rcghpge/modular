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
from math import ceildiv, isqrt
from random import random_ui64

from buffer import Dim, DimList
from internal_utils import HostNDBuffer, random
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from memory import memcpy
from nn.flash_attention import flash_attention_kv_cache
from nn.mha_mask import CausalMask
from testing import assert_almost_equal

from utils import IndexList


def execute_ragged_flash_attention():
    alias num_q_heads = 32
    alias kv_params = KVCacheStaticParams(num_heads=8, head_size=128)
    alias type = DType.float32
    alias num_paged_blocks = 32
    alias page_size = 512
    alias PagedCollectionType = PagedKVCacheCollection[
        type, kv_params, page_size
    ]
    var num_layers = 1
    var layer_idx = 0

    var true_ce_prompt_lens = [100, 200, 300, 400]
    var mixed_ce_prompt_lens = [50, 100, 150, 100]

    var true_ce_cache_lens = [0, 0, 0, 0]
    var mixed_ce_cache_lens = [50, 100, 150, 300]

    var batch_size = len(true_ce_prompt_lens)

    var true_ce_row_offsets = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var true_ce_cache_lengths = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )
    var mixed_ce_row_offsets = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size + 1)
    )
    var mixed_ce_cache_lengths = HostNDBuffer[DType.uint32, 1](
        IndexList[1](batch_size)
    )

    var true_ce_total_length = 0
    var mixed_ce_total_length = 0
    var true_ce_max_full_context_length = 0
    var mixed_ce_max_full_context_length = 0
    var true_ce_max_prompt_length = 0
    var mixed_ce_max_prompt_length = 0
    for i in range(batch_size):
        true_ce_row_offsets.tensor[i] = true_ce_total_length
        mixed_ce_row_offsets.tensor[i] = mixed_ce_total_length
        true_ce_cache_lengths.tensor[i] = true_ce_cache_lens[i]
        mixed_ce_cache_lengths.tensor[i] = mixed_ce_cache_lens[i]

        true_ce_max_full_context_length = max(
            true_ce_max_full_context_length,
            true_ce_cache_lens[i] + true_ce_prompt_lens[i],
        )
        mixed_ce_max_full_context_length = max(
            mixed_ce_max_full_context_length,
            mixed_ce_cache_lens[i] + mixed_ce_prompt_lens[i],
        )

        true_ce_max_prompt_length = max(
            true_ce_max_prompt_length, true_ce_prompt_lens[i]
        )
        mixed_ce_max_prompt_length = max(
            mixed_ce_max_prompt_length, mixed_ce_prompt_lens[i]
        )

        true_ce_total_length += true_ce_prompt_lens[i]
        mixed_ce_total_length += mixed_ce_prompt_lens[i]

    true_ce_row_offsets.tensor[batch_size] = true_ce_total_length
    mixed_ce_row_offsets.tensor[batch_size] = mixed_ce_total_length
    true_ce_q_ragged = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size))
    random(true_ce_q_ragged.tensor)

    mixed_ce_q_ragged = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](mixed_ce_total_length, num_q_heads, kv_params.head_size))
    for bs_idx in range(batch_size):
        true_ce_prompt_len = true_ce_prompt_lens[bs_idx]
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs_idx]

        true_ce_row_offset = true_ce_row_offsets.tensor[bs_idx]
        mixed_ce_row_offset = mixed_ce_row_offsets.tensor[bs_idx]

        mixed_ce_cache_len = mixed_ce_cache_lens[bs_idx]

        true_ce_offset = true_ce_q_ragged.tensor._offset(
            IndexList[3](Int(true_ce_row_offset + mixed_ce_cache_len), 0, 0)
        )
        mixed_ce_offset = mixed_ce_q_ragged.tensor._offset(
            IndexList[3](Int(mixed_ce_row_offset), 0, 0)
        )

        memcpy(
            mixed_ce_offset,
            true_ce_offset,
            mixed_ce_prompt_len * num_q_heads * kv_params.head_size,
        )

    # initialize reference output
    mixed_ce_output = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](mixed_ce_total_length, num_q_heads, kv_params.head_size))
    true_ce_output = HostNDBuffer[
        type, 3, DimList(Dim(), num_q_heads, kv_params.head_size)
    ](IndexList[3](true_ce_total_length, num_q_heads, kv_params.head_size))

    # initialize our KVCache
    kv_block_paged = HostNDBuffer[type, 6](
        IndexList[6](
            num_paged_blocks,
            2,
            num_layers,
            page_size,
            kv_params.num_heads,
            kv_params.head_size,
        )
    )
    random(kv_block_paged.tensor)

    paged_lut = HostNDBuffer[DType.uint32, 2](
        IndexList[2](
            batch_size,
            ceildiv(true_ce_max_full_context_length, page_size),
        )
    )
    paged_lut_set = Set[Int]()
    for bs in range(batch_size):
        seq_len = true_ce_cache_lens[bs] + true_ce_prompt_lens[bs]

        for block_idx in range(0, ceildiv(seq_len, page_size)):
            var randval = Int(random_ui64(0, num_paged_blocks - 1))
            while randval in paged_lut_set:
                randval = Int(random_ui64(0, num_paged_blocks - 1))

            paged_lut_set.add(randval)
            paged_lut.tensor[bs, block_idx] = randval

    true_ce_kv_collection = PagedCollectionType(
        kv_block_paged.tensor,
        true_ce_cache_lengths.tensor,
        paged_lut.tensor,
        true_ce_max_prompt_length,
        true_ce_max_full_context_length,
    )

    mixed_ce_kv_collection = PagedCollectionType(
        kv_block_paged.tensor,
        mixed_ce_cache_lengths.tensor,
        paged_lut.tensor,
        mixed_ce_max_prompt_length,
        mixed_ce_max_full_context_length,
    )

    # "true CE" execution
    print("true")
    flash_attention_kv_cache(
        true_ce_q_ragged.to_layout_tensor(),
        true_ce_row_offsets.to_layout_tensor(),
        true_ce_row_offsets.to_layout_tensor(),
        true_ce_kv_collection.get_key_cache(layer_idx),
        true_ce_kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        isqrt(Float32(kv_params.head_size)),
        true_ce_output.to_layout_tensor(),
    )

    # "mixed CE" execution
    print("mixed")
    flash_attention_kv_cache(
        mixed_ce_q_ragged.to_layout_tensor(),
        mixed_ce_row_offsets.to_layout_tensor(),
        mixed_ce_row_offsets.to_layout_tensor(),
        mixed_ce_kv_collection.get_key_cache(layer_idx),
        mixed_ce_kv_collection.get_value_cache(layer_idx),
        CausalMask(),
        isqrt(Float32(kv_params.head_size)),
        mixed_ce_output.to_layout_tensor(),
    )

    true_ce_out = true_ce_output.tensor
    mixed_ce_out = mixed_ce_output.tensor
    for bs in range(batch_size):
        mixed_ce_prompt_len = mixed_ce_prompt_lens[bs]
        mixed_ce_row_offset = mixed_ce_row_offsets.tensor[bs]
        true_ce_row_offset = true_ce_row_offsets.tensor[bs]
        mixed_ce_cache_len = mixed_ce_cache_lens[bs]

        true_ce_ragged_offset = Int(true_ce_row_offset + mixed_ce_cache_len)
        mixed_ce_ragged_offset = Int(mixed_ce_row_offset)
        for s in range(mixed_ce_prompt_len):
            for h in range(num_q_heads):
                for hd in range(kv_params.head_size):
                    try:
                        assert_almost_equal(
                            true_ce_out[true_ce_ragged_offset + s, h, hd],
                            mixed_ce_out[mixed_ce_ragged_offset + s, h, hd],
                            atol=1e-3,
                        )
                    except e:
                        print(
                            "MISMATCH:",
                            bs,
                            s,
                            h,
                            hd,
                            true_ce_out[true_ce_ragged_offset + s, h, hd],
                            mixed_ce_out[mixed_ce_ragged_offset + s, h, hd],
                        )
                        raise e

    _ = true_ce_q_ragged^
    _ = mixed_ce_q_ragged^
    _ = true_ce_row_offsets^
    _ = mixed_ce_row_offsets^
    _ = kv_block_paged^
    _ = paged_lut^
    _ = true_ce_output^
    _ = mixed_ce_output^
    _ = true_ce_cache_lengths^
    _ = mixed_ce_cache_lengths^


def main():
    execute_ragged_flash_attention()
