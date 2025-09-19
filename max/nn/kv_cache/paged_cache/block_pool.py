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

"""Device specific block pool for PagedAttention KVCache.

Supports allocating/freeing blocks and maps block hashes to committed blocks.

This logic is largely borrowed from vLLM v1:
- https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_manager.py#L1
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_utils.py#L1

TODO E2EOPT-116: Port block_pool.py and block_utils.py to Mojo
"""

from __future__ import annotations

import logging
import multiprocessing

from max.profiler import traced
from max.serve.kvcache_agent.kvcache_agent import KVCacheChangeMessage
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
    UpdateType,
)

from .block_utils import FreeKVCacheBlockQueue, KVCacheBlock

logger = logging.getLogger("max.pipelines")


class BlockPool:
    @traced
    def __init__(
        self,
        memory_tier: MemoryTier,
        total_num_blocks: int,
        enable_prefix_caching: bool,
        enable_runtime_checks: bool = False,
    ) -> None:
        self.memory_tier = memory_tier
        self.total_num_blocks = total_num_blocks
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_runtime_checks = enable_runtime_checks

        # A Block pool of all kv-cache blocks.
        self.pool: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(self.total_num_blocks)
        ]

        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.pool)

        # Mapping from block hash to a committed block.
        # A committed block is a full block with a block hash that can be shared
        # between requests for prefix caching. The cached block may be used by
        # running requests or in the free_block_queue that could potentially
        # be evicted.
        self.hash_to_committed_block: dict[int, KVCacheBlock] = {}

        # Queue for the KV Cache Agent updates
        self.kv_cache_agent_queue: (
            multiprocessing.Queue[KVCacheChangeMessage] | None
        ) = None

    @traced
    def commit_into_prefix_cache(
        self,
        block_hash: int,
        block: KVCacheBlock,
    ) -> None:
        """Commit a block into the prefix cache."""
        assert block.block_hash is None
        block.block_hash = block_hash

        # Commit the block into the prefix cache.
        hash_value = block_hash
        self.hash_to_committed_block[hash_value] = block

        if self.kv_cache_agent_queue is None:
            return

        logger.debug(
            f"Updating KV Cache Agent with block {hash_value}, memory tier {self.memory_tier}, update type {UpdateType.UPDATE_TYPE_ADDED}"
        )
        self.kv_cache_agent_queue.put_nowait(
            KVCacheChangeMessage(
                cache_id=str(hash_value),
                memory_tier=self.memory_tier,
                update_type=UpdateType.UPDATE_TYPE_ADDED,
            )
        )

    def get_or_commit_into_prefix_cache(
        self,
        block_hash: int,
        block: KVCacheBlock,
    ) -> None | KVCacheBlock:
        """Get or commit a block into the prefix cache.

        If there already exists a committed block with the same hash, we return
        the already committed block. Otherwise, we commit the provided block
        into the prefix cache and return None.
        """
        hash_value = block_hash
        if hash_value in self.hash_to_committed_block:
            # Check if a block with the same hash is already committed.
            # If so, we reuse the already committed block.
            prefix_cache_block = self.hash_to_committed_block[hash_value]
            if block.bid == prefix_cache_block.bid:
                return None

            self.touch(prefix_cache_block)

            # Free the block we currently have.
            assert block.block_hash is None
            self.free_block(block)

            return prefix_cache_block

        self.commit_into_prefix_cache(block_hash, block)
        return None

    @traced
    def uncommit_block(self, block: KVCacheBlock) -> None:
        """Evict a block from the prefix cache."""
        assert block.block_hash is not None
        hash_value = block.block_hash

        # Nothing to do if it is not committed.
        if hash_value not in self.hash_to_committed_block:
            return

        del self.hash_to_committed_block[hash_value]
        block.block_hash = None

        if self.kv_cache_agent_queue is None:
            return

        # Notify KV Cache Agent of update
        logger.debug(
            f"Updating KV Cache Agent with block {hash_value}, memory tier {self.memory_tier}, update type {UpdateType.UPDATE_TYPE_ADDED}"
        )
        self.kv_cache_agent_queue.put_nowait(
            KVCacheChangeMessage(
                cache_id=str(hash_value),
                memory_tier=self.memory_tier,
                update_type=UpdateType.UPDATE_TYPE_REMOVED,
            )
        )

    @traced
    def alloc_block(self) -> tuple[KVCacheBlock, int | None]:
        """Allocate a block from the free block queue."""

        # First allocate block
        curr_block = self.free_block_queue.popleft()
        assert curr_block.ref_cnt == 0

        # If the block is committed into prefix cache, evict it.
        block_hash = curr_block.block_hash
        assert self.enable_prefix_caching or block_hash is None
        if block_hash is not None:
            self.uncommit_block(curr_block)

        curr_block.ref_cnt += 1
        assert curr_block.block_hash is None
        return curr_block, block_hash

    @traced
    def free_block(self, block: KVCacheBlock) -> None:
        """Free a block by decreasing its reference count.

        If the reference count is 0, the block is added to the free block queue.

        Note that a block can be in both the prefix cache and the free block
        queue at the same time.
        """
        block.ref_cnt -= 1
        assert block.ref_cnt >= 0
        if block.ref_cnt == 0:
            self.free_block_queue.append(block)

    @traced
    def touch(self, block: KVCacheBlock) -> None:
        """Touching a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.
        """
        # ref_cnt=0 means this block is in the free list (i.e. eviction
        # candidate), so remove it.
        if block.ref_cnt == 0:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1

    @property
    def free_blocks(self) -> set[int]:
        """Get the number of free blocks."""
        return self.free_block_queue.free_blocks

    @traced
    def assert_runtime_invariants(self, active_bids: list[int]) -> None:
        """If runtime checks are enabled, assert that the runtime checks are
        correct.
        """
        if not self.enable_runtime_checks:
            return

        # Check that all blocks in the prefix cache are committed.
        for (
            block_hash,
            block,
        ) in self.hash_to_committed_block.items():
            assert block.block_hash is not None
            assert block.block_hash == block_hash

        # Check that the total number of blocks is correct.
        free_blocks = len(self.free_block_queue)
        assert free_blocks + len(set(active_bids)) == self.total_num_blocks
