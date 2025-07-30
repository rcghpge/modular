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

"""Block manager for PagedAttention KVCache.

Handles allocating new blocks for requests as well as prefix caching/reuse.
This is done very efficiently and largely avoids Python memory allocations.

This logic is largely borrowed from vLLM v1:
- https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_manager.py#L1
- https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/v1/core/kv_cache_utils.py#L1
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from typing import Generic, TypeVar

import numpy as np
from max.interfaces.request import RequestID
from max.profiler import traced
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
)
from max.support.math import ceildiv

from ..context import KVCacheAwareContext
from .block_copy_engine import BlockCopyEngine
from .block_pool import BlockPool
from .block_utils import (
    ROOT_BLOCK_HASH,
    BlockHashType,
    KVCacheBlock,
    hash_block_tokens,
    hash_request_tokens,
)

logger = logging.getLogger("max.pipelines")


class SwappingStrategy(Enum):
    """Strategy for offboarding blocks from the device when experimental paging to
    CPU feature is enabled."""

    # Copy device blocks to host as soon as they are committed.
    # This results in more copies but is efficient if copies can be hidden in a
    # separate stream.
    EAGER = "EAGER"
    # Copy device blocks to host only when they are evicted.
    LAZY = "LAZY"


T = TypeVar("T", bound=KVCacheAwareContext)


class BlockManager(Generic[T]):
    @traced
    def __init__(
        self,
        device_memory_tier: MemoryTier,
        total_num_blocks: int,
        total_num_host_blocks: int,
        block_size: int,
        block_copy_engine: BlockCopyEngine | None,
        enable_prefix_caching: bool,
        enable_runtime_checks: bool = False,
    ) -> None:
        # The number of tokens in a single page.
        self.block_size = block_size

        # Whether to enable prefix caching.
        self.enable_prefix_caching = enable_prefix_caching

        # The block copy engine.
        if enable_prefix_caching and block_copy_engine is None:
            raise ValueError(
                "Block copy engine must be provided if prefix caching is enabled"
            )
        self.block_copy_engine = block_copy_engine

        # A pool of device blocks.
        self.device_block_pool = BlockPool(
            device_memory_tier,
            total_num_blocks,
            enable_prefix_caching,
            enable_parent_to_child_mapping=True,
            enable_runtime_checks=enable_runtime_checks,
        )

        # A pool of host blocks.
        self.host_block_pool: BlockPool | None = None
        self.swapping_strategy: SwappingStrategy | None = None
        if total_num_host_blocks > 0:
            self.host_block_pool = BlockPool(
                MemoryTier.MEMORY_TIER_CPU,
                total_num_host_blocks,
                enable_prefix_caching,
                enable_parent_to_child_mapping=False,
                enable_runtime_checks=enable_runtime_checks,
            )

            if self.block_copy_engine is None:
                raise ValueError(
                    "Block copy engine must be provided if host block pool is enabled"
                )

            # Determine the swapping strategy based on whether the block copy engine
            # supports multistream.
            if self.block_copy_engine.supports_multistream():
                self.swapping_strategy = SwappingStrategy.EAGER
            else:
                self.swapping_strategy = SwappingStrategy.LAZY
            logger.info(
                f"Host KVCache swapping strategy: {self.swapping_strategy.value}"
            )

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.current_blocks_per_request: dict[RequestID, list[KVCacheBlock]] = (
            defaultdict(list)
        )

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_hashes: dict[RequestID, list[BlockHashType]] = defaultdict(
            list
        )

        # Cache hit rate metrics.
        self.prompt_tokens = 0
        self.cached_prompt_tokens = 0

        # Tracks recently committed device blocks to offload to host.
        self.recently_committed_device_blocks: list[KVCacheBlock] | None = None
        if self.swapping_strategy == SwappingStrategy.EAGER:
            self.recently_committed_device_blocks = []

        # Whether to enable runtime checks.
        self.enable_runtime_checks = enable_runtime_checks

    @traced
    def step(self, ctx: T) -> None:
        """Step the block manager by committing blocks into prefix cache."""
        self.assert_runtime_invariants(ctx)

        if not self.enable_prefix_caching:
            return

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_hashes_for_request(ctx)

        # Now that we generated new tokens, we can possibly commit additional
        # blocks into prefix cache.
        self.commit_to_prefix_cache(ctx)

        self.assert_runtime_invariants(ctx)

    @traced
    def compute_hashes_for_request(
        self,
        ctx: T,
    ) -> None:
        """Compute the block hashes for the request."""

        hashes = self.req_to_hashes[ctx.request_id]

        num_unhashed_tokens = ctx.current_length - (
            len(hashes) * self.block_size
        )
        if num_unhashed_tokens < self.block_size:
            return

        parent_hash_value = None
        if len(hashes) > 0:
            parent_hash_value = hashes[-1].value

        unhashed_tokens = ctx.tokens[
            len(hashes) * self.block_size : ctx.current_length
        ]
        new_hashes = hash_request_tokens(
            unhashed_tokens, self.block_size, parent_hash_value
        )
        hashes.extend(new_hashes)

    @traced
    def reuse_blocks_from_prefix_cache(self, ctx: T) -> None:
        """Reuse blocks from prefix cache.

        Full blocks are directly reused and appended to the request's blocks.
        Partial blocks can be reused via COW. The blocks/tokens to copy to and
        from are returned as a tuple.

        This also updates the cache hit rate metrics.
        """
        self.assert_runtime_invariants(ctx)

        if not self.enable_prefix_caching or ctx.active_length == 1:
            return

        req_blocks = self.current_blocks_per_request[ctx.request_id]

        # Update cache hit rate metrics.
        orig_prompt_len = ctx.active_length
        self.prompt_tokens += orig_prompt_len - 1

        # Compute block hashes. These hashes are used by the subsequent methods.
        self.compute_hashes_for_request(ctx)

        # Query prefix cache for full blocks.
        prefix_cache_blocks = self.get_full_blocks_from_prefix_cache(ctx)
        orig_start_idx = ctx.start_idx

        if len(prefix_cache_blocks) > 0:
            # Since we got cache hits, clear out existing uncommitted blocks
            self.release_uncommitted_blocks(ctx)

            # Append them to the request's blocks.
            req_blocks.extend(prefix_cache_blocks)
            new_committed_idx = (
                ctx.committed_idx + len(prefix_cache_blocks) * self.block_size
            )
            ctx.set_token_indices(
                committed_idx=new_committed_idx, start_idx=new_committed_idx
            )
            assert ctx.committed_idx == ctx.start_idx

            # Check that the cached_idx has increased.
            assert ctx.start_idx > orig_start_idx
            orig_start_idx = ctx.start_idx

        # Query prefix cache for partial blocks
        partial_block, tokens_matched = (
            self.get_partial_block_from_prefix_cache(ctx)
        )

        if partial_block is not None:
            # Since we got cache hits, clear out existing uncommitted blocks
            self.release_uncommitted_blocks(ctx)

            # Touch and free block to move it to end of the free list.
            self.device_block_pool.touch(partial_block)
            self.device_block_pool.free_block(partial_block)

            # We can only perform COW if we can allocate a new block to copy into
            if self.device_block_pool.free_block_queue:
                # Append them to the request's blocks.
                block_hash = partial_block.block_hash
                assert block_hash is not None

                fresh_block = self.allocate_device_block()
                req_blocks.append(fresh_block)
                ctx.bump_token_indices(start_idx=tokens_matched)

                # Enqueue a D2D block copy operation.
                assert self.block_copy_engine is not None
                self.block_copy_engine.memcpy_d2d(
                    fresh_block.bid, partial_block.bid, tokens_matched
                )

                # Check that the cached_idx has increased.
                assert ctx.start_idx > orig_start_idx
                orig_start_idx = ctx.start_idx

        # Update cache hit rate metrics.
        new_prompt_len = ctx.active_length
        self.cached_prompt_tokens += orig_prompt_len - new_prompt_len

    @traced
    def _get_full_blocks_from_device_prefix_cache(
        self,
        desired_hashes: list[BlockHashType],
    ) -> list[KVCacheBlock]:
        """Returns a list of device blocks with the desired hashes."""

        device_prefix_cache = self.device_block_pool.hash_to_committed_block

        blocks = []
        for block_hash in desired_hashes:
            hash_value = block_hash.value
            if hash_value not in device_prefix_cache:
                break
            block = device_prefix_cache[hash_value]
            blocks.append(block)
            self.device_block_pool.touch(block)

        return blocks

    @traced
    def _get_full_blocks_from_host_prefix_cache(
        self,
        desired_hashes: list[BlockHashType],
    ) -> list[KVCacheBlock]:
        """Returns a list of device blocks with the desired hashes.

        These device blocks are newly allocated and initialized with the
        contents of the host blocks.
        """

        assert self.host_block_pool is not None
        host_prefix_cache = self.host_block_pool.hash_to_committed_block

        blocks = []
        for block_hash in desired_hashes:
            hash_value = block_hash.value
            if (
                hash_value not in host_prefix_cache
                or len(self.device_block_pool.free_block_queue) == 0
            ):
                break

            host_block = host_prefix_cache[hash_value]
            assert host_block.block_hash is not None
            self.host_block_pool.touch(host_block)

            # Allocate a new device block.
            device_block = self.allocate_device_block()
            blocks.append(device_block)

            # Enqueue a H2D block copy operation.
            assert self.block_copy_engine is not None
            self.block_copy_engine.memcpy_h2d(device_block.bid, host_block.bid)

            # Commit the device block into the prefix cache.
            # We should use the hash from the host block.
            self.device_block_pool.commit_into_prefix_cache(
                host_block.block_hash, device_block
            )

        return blocks

    @traced
    def get_full_blocks_from_prefix_cache(
        self,
        ctx: T,
    ) -> list[KVCacheBlock]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.
        """

        assert self.enable_prefix_caching

        req_hashes = self.req_to_hashes[ctx.request_id]
        num_committed_blocks = ctx.committed_idx // self.block_size
        # we exclude the last inflight token to ensure that there is at least
        # one prompt token to be encoded.
        num_inflight_blocks = (ctx.current_length - 1) // self.block_size
        uncommitted_hashes = req_hashes[
            num_committed_blocks:num_inflight_blocks
        ]

        # query the device prefix cache for full blocks
        device_blocks = self._get_full_blocks_from_device_prefix_cache(
            uncommitted_hashes
        )

        if self.host_block_pool is None:
            return device_blocks

        # remove the hashes that were found in the device prefix cache
        if len(device_blocks) > 0:
            uncommitted_hashes = uncommitted_hashes[len(device_blocks) :]

        # query the host prefix cache for full blocks
        host_blocks = self._get_full_blocks_from_host_prefix_cache(
            uncommitted_hashes
        )

        return device_blocks + host_blocks

    @traced
    def get_partial_block_from_prefix_cache(
        self,
        ctx: T,
    ) -> tuple[KVCacheBlock | None, int]:
        """Get the computed (cached) blocks for the request."""
        assert self.enable_prefix_caching

        if self.block_size == 1:
            return None, 0

        req_hashes = self.req_to_hashes[ctx.request_id]
        num_committed_blocks = ctx.committed_idx // self.block_size

        parent_hash = ROOT_BLOCK_HASH
        if num_committed_blocks > 0:
            parent_hash = req_hashes[num_committed_blocks - 1]
        parent_tokens = ctx.tokens[
            num_committed_blocks * self.block_size : ctx.current_length - 1
        ]
        if len(parent_tokens) == 0:
            return None, 0

        # Find the longest prefix match in the prefix cache.
        children = self.device_block_pool.parent_hash_to_child_token_ids[
            parent_hash.value
        ]

        parent_tokens = parent_tokens[: self.block_size]
        res = children.find_string_with_largest_common_prefix(parent_tokens)
        if res is None:
            return None, 0
        best_child_tokens, best_tokens_matched = res
        assert best_tokens_matched < self.block_size

        # It is not profitable to do COW if this request's partial block has
        # at least as many tokens as the best match in the prefix cache.
        current_tokens_in_partial_block = ctx.start_idx % self.block_size
        if current_tokens_in_partial_block >= best_tokens_matched:
            return None, 0

        child_hash = hash_block_tokens(
            np.array(best_child_tokens), parent_hash.value
        )
        child_block = self.device_block_pool.hash_to_committed_block[
            child_hash.value
        ]
        return child_block, best_tokens_matched

    @traced
    def commit_to_prefix_cache(
        self,
        ctx: T,
    ) -> None:
        """Commits all blocks whose hashes are known for prefix caching.

        This increments the committed_idx.

        Args:
            ctx: Request InputContext.
        """

        req_blocks = self.current_blocks_per_request[ctx.request_id]
        req_hashes = self.req_to_hashes[ctx.request_id]
        num_committed_blocks = ctx.committed_idx // self.block_size

        # Count the number of tokens for which we know the values of and align
        # to the block size.
        num_computed_blocks = ctx.start_idx // self.block_size

        # Commit these blocks into the prefix cache.
        for block_idx in range(num_committed_blocks, num_computed_blocks):
            block = req_blocks[block_idx]

            # Get the block hash.
            block_hash = req_hashes[block_idx]

            # Get the parent block hash.
            new_block = self.device_block_pool.get_or_commit_into_prefix_cache(
                block_hash, block
            )
            if new_block is not None:
                req_blocks[block_idx] = new_block

            if (
                new_block is None
                and self.recently_committed_device_blocks is not None
            ):
                self.recently_committed_device_blocks.append(block)

        ctx.set_token_indices(
            committed_idx=num_computed_blocks * self.block_size
        )

    def release(self, request_id: RequestID) -> None:
        """Release the blocks for the request."""

        blocks = self.current_blocks_per_request[request_id]
        ordered_blocks: Iterable[KVCacheBlock] = blocks
        if self.enable_prefix_caching:
            # Free blocks in reverse order so that the tail blocks are
            # freed first.
            ordered_blocks = reversed(blocks)

        for block in ordered_blocks:
            self.device_block_pool.free_block(block)

        self.current_blocks_per_request[request_id] = []
        self.req_to_hashes[request_id] = []

    @traced
    def allocate_new_blocks(self, ctx: T, num_steps: int = 1) -> None:
        """Allocate new blocks for a request to accommodate additional tokens.

        Calculates the number of additional blocks needed based on the current sequence
        length and number of steps, then allocates them from the device block pool.
        Validates that there are sufficient free blocks available and that the current
        blocks can accommodate the completed tokens.

        Args:
            ctx: The request context containing sequence information and token indices.
            num_steps: Number of additional steps to allocate blocks for. Defaults to 1.

        Raises:
            RuntimeError: If there are insufficient free blocks to satisfy the allocation.
            AssertionError: If the current blocks cannot accommodate the completed tokens.
        """
        # Determine number of new blocks to allocate.
        current_blocks = self.current_blocks_per_request[ctx.request_id]
        num_current_blocks = len(current_blocks)
        current_seq_len = ctx.current_length + num_steps - 1
        num_required_blocks = ceildiv(current_seq_len, self.block_size)
        num_new_blocks = num_required_blocks - num_current_blocks
        num_new_blocks = max(num_new_blocks, 0)

        # Check that the number of completed tokens is less than or equal to the number of tokens we can
        # currently store in the reserved blocks.
        num_completed_tokens = ctx.current_length - ctx.active_length
        assert num_completed_tokens <= (num_current_blocks * self.block_size), (
            f"the current blocks reserved, have space for {num_current_blocks * self.block_size} tokens, but {num_completed_tokens} are already completed. This should never happen."
        )

        # Check that we have enough free blocks to allocate the new blocks.
        if num_new_blocks > len(self.device_block_pool.free_block_queue):
            raise RuntimeError(
                f"Cannot get {num_new_blocks} free blocks from the free block queue (only {len(self.device_block_pool.free_block_queue)} available)"
            )

        # Allocate new blocks.
        for _ in range(num_new_blocks):
            new_block = self.allocate_device_block()
            current_blocks.append(new_block)

    @traced
    def eagerly_offload_recently_committed_blocks(self) -> None:
        """Offload recently committed blocks to host memory."""
        assert self.recently_committed_device_blocks is not None
        assert self.swapping_strategy == SwappingStrategy.EAGER

        for block in self.recently_committed_device_blocks:
            self.maybe_offload_gpu_block_to_host(block, block.block_hash)
        self.recently_committed_device_blocks.clear()

    @traced
    def maybe_offload_gpu_block_to_host(
        self, gpu_block: KVCacheBlock, old_hash: BlockHashType | None
    ) -> None:
        # Can't swap if there is no host block pool.
        if self.host_block_pool is None:
            return

        # Can't swap if the block was not previously committed.
        if old_hash is None:
            return

        # Should not swap if another block with the same hash is present.
        if old_hash.value in self.host_block_pool.hash_to_committed_block:
            return

        # Allocate a host block
        host_block, _ = self.host_block_pool.alloc_block()

        # Copy the block from the GPU to the host.
        assert self.block_copy_engine is not None
        self.block_copy_engine.memcpy_d2h(host_block.bid, gpu_block.bid)

        # Commit the host block into the host prefix cache.
        self.host_block_pool.commit_into_prefix_cache(old_hash, host_block)

    @traced
    def allocate_device_block(self) -> KVCacheBlock:
        new_block, block_hash = self.device_block_pool.alloc_block()
        if self.swapping_strategy == SwappingStrategy.LAZY:
            self.maybe_offload_gpu_block_to_host(new_block, block_hash)
        return new_block

    @property
    def cache_hit_rate(self) -> float:
        """Get the percentage of prompt tokens that were retrieved from the cache."""
        if self.prompt_tokens == 0:
            return 0
        return self.cached_prompt_tokens / self.prompt_tokens

    def release_uncommitted_blocks(self, ctx: T) -> None:
        """Release the uncommitted blocks for the request."""
        req_blocks = self.current_blocks_per_request[ctx.request_id]
        num_committed_blocks = ctx.committed_idx // self.block_size
        assert len(req_blocks) >= num_committed_blocks
        num_uncommitted_blocks = len(req_blocks) - num_committed_blocks
        for _ in range(num_uncommitted_blocks):
            block = req_blocks.pop()
            self.device_block_pool.free_block(block)
        ctx.set_token_indices(start_idx=ctx.committed_idx)

    @traced
    def get_req_blocks(self, request_id: RequestID) -> list[int]:
        """Get the block ids for a request."""
        return [
            block.bid for block in self.current_blocks_per_request[request_id]
        ]

    @traced
    def assert_runtime_invariants(self, ctx: T) -> None:
        """If runtime checks are enabled, assert that the runtime checks are
        correct.
        """
        if not self.enable_runtime_checks:
            return

        # Get the active block ids
        active_block_ids = []
        for blocks in self.current_blocks_per_request.values():
            for block in blocks:
                active_block_ids.append(block.bid)
                # Check that all active blocks have a ref_cnt > 0
                assert block.ref_cnt > 0

        # Check that the block pool is consistent
        self.device_block_pool.assert_runtime_invariants(active_block_ids)

        # Get the request hashes and blocks
        req_hashes = self.req_to_hashes[ctx.request_id]
        req_blocks = self.current_blocks_per_request[ctx.request_id]

        # Check that the number of committed blocks for request is correct
        num_committed_blocks = ctx.committed_idx // self.block_size
        num_committed = 0
        for block in req_blocks:
            if block.block_hash is None:
                break
            num_committed += 1
        assert num_committed == num_committed_blocks

        # Check that the tokens in the request line up with the contents of the hashes
        for hash_idx, req_hash in enumerate(req_hashes):
            tokens = ctx.tokens[
                hash_idx * self.block_size : (hash_idx + 1) * self.block_size
            ]
            assert req_hash.token_ids == tuple(tokens)

        # Check that the req block hashes are consistent with req blocks
        for hash_value, block in zip(req_hashes, req_blocks):
            assert block.block_hash is None or block.block_hash == hash_value

        # Check that the req block hashes are consistent with parents
        for hash_idx in range(1, len(req_hashes)):
            # check that hashing parent with token ids of current block
            # yields the same hash as the parent block hash
            curr_hash = req_hashes[hash_idx]
            prev_hash = req_hashes[hash_idx - 1]
            assert curr_hash.parent_hash_value == prev_hash.value
            assert curr_hash == hash_block_tokens(
                np.array(curr_hash.token_ids), prev_hash.value
            )
