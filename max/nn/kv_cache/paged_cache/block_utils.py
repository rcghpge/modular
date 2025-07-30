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

"""Utilities for PagedAttention KVCache block manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from max._core_mojo import block_hasher
from max.profiler import traced


class BlockHashType(NamedTuple):
    """Hash value of a block. This is computed by hashing the hash_value of the
    parent block with the token ids of the current block.

    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. But please note that
    hash collisions can still theoretically occur, albeit with an extremely
    low probability.

    Additional values needed to uniquely identify a block can be added here in
    the future, eg: model name or multimodal image ids.
    """

    # Hashed value returned by hash()
    value: int

    # The hash of the parent block.
    parent_hash_value: int

    # The token ids of the block.
    token_ids: tuple[int, ...]

    def __repr__(self) -> str:
        token_ids_str = ", ".join(str(x) for x in self.token_ids[:5])
        return f"BlockHashType({self.value}, [{token_ids_str}, ...])"


# Note that we use 'None' as a string here instead of None because
# as of Python 3.12, hash(None) returns a constant predictable value.
# This could possibly make it easier to find and exploit hash
# collisions. 'None' as a string will be hashed differently per process,
# but consistently within the same process. This is the same as the
# behavior of None prior to Python 3.12.
DEFAULT_PARENT_HASH = hash("None")

ROOT_BLOCK_HASH = BlockHashType(DEFAULT_PARENT_HASH, -1, ())

ENABLE_MOJO_BLOCK_HASHER = True


@traced
def _hash_block_tokens_mojo(
    block_token_ids: np.ndarray,
    parent_hash: int | None = None,
) -> BlockHashType:
    """Hash the tokens of a block using the Mojo implementation."""
    block_size = len(block_token_ids)
    if parent_hash is None:
        parent_hash = DEFAULT_PARENT_HASH
    hash_vals = block_hasher(block_token_ids, block_size, parent_hash)
    assert len(hash_vals) == 1
    return BlockHashType(hash_vals[0], parent_hash, tuple(block_token_ids))


@traced
def _hash_request_tokens_mojo(
    token_ids: np.ndarray,
    block_size: int,
    parent_hash: int | None = None,
) -> list[BlockHashType]:
    """Hash the tokens of a request using the Mojo implementation."""
    if parent_hash is None:
        parent_hash = DEFAULT_PARENT_HASH

    hash_vals = block_hasher(token_ids, block_size, parent_hash)
    assert len(hash_vals) == len(token_ids) // block_size

    prev_hash = parent_hash
    ret = []
    for i, hash_val in enumerate(hash_vals):
        start = i * block_size
        end = start + block_size
        block_token_ids = token_ids[start:end]
        ret.append(BlockHashType(hash_val, prev_hash, tuple(block_token_ids)))
        prev_hash = hash_val

    return ret


@traced
def hash_block_tokens(
    token_ids: np.ndarray,
    parent_hash: int | None = None,
) -> BlockHashType:
    """Compute the hash value of a block."""

    if ENABLE_MOJO_BLOCK_HASHER:
        return _hash_block_tokens_mojo(token_ids, parent_hash)

    if parent_hash is None:
        parent_hash = DEFAULT_PARENT_HASH

    token_ids_tuple = tuple(token_ids)
    tuple_to_hash = (token_ids_tuple, parent_hash)
    hash_value = hash(tuple_to_hash)
    return BlockHashType(hash_value, parent_hash, token_ids_tuple)


@traced
def hash_request_tokens(
    token_ids: np.ndarray,
    block_size: int,
    parent_hash: int | None = None,
) -> list[BlockHashType]:
    """Hash the tokens of a request."""

    if ENABLE_MOJO_BLOCK_HASHER:
        return _hash_request_tokens_mojo(token_ids, block_size, parent_hash)

    ret = []
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break
        block_hash = hash_block_tokens(block_token_ids, parent_hash)
        ret.append(block_hash)
        parent_hash = block_hash.value
    return ret


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to total_num_blocks - 1.
    bid: int
    # Reference count.
    ref_cnt: int = 0
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    block_hash: BlockHashType | None = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None

    def __repr__(self) -> str:
        return f"KVCacheBlock(bid={self.bid}, ref_cnt={self.ref_cnt}, block_hash={self.block_hash})"


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        self.free_blocks = set(block.bid for block in blocks)

        # Initialize the doubly linked list of free blocks.
        self.free_list_head: KVCacheBlock | None = blocks[0]
        self.free_list_tail: KVCacheBlock | None = blocks[-1]
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

    def __len__(self) -> int:
        return self.num_free_blocks

    @traced
    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.
        """
        if not self.free_list_head:
            raise ValueError("No free blocks available")

        block = self.free_list_head
        self.remove(block)
        return block

    @traced
    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.
        """
        if block.prev_free_block is not None:
            # Link the previous block to the next block.
            block.prev_free_block.next_free_block = block.next_free_block
        if block.next_free_block is not None:
            # Link the next block to the previous block.
            block.next_free_block.prev_free_block = block.prev_free_block

        if block == self.free_list_head:
            # Update the head if the block is the head.
            self.free_list_head = block.next_free_block
        if block == self.free_list_tail:
            # Update the tail if the block is the tail.
            self.free_list_tail = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1
        self.free_blocks.remove(block.bid)

    @traced
    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.free_list_tail is not None:
            # Link the last block to the new block.
            self.free_list_tail.next_free_block = block
            block.prev_free_block = self.free_list_tail
            self.free_list_tail = block
        else:
            # The free list is empty.
            assert self.free_list_head is None
            self.free_list_head = self.free_list_tail = block

        block.next_free_block = None
        self.num_free_blocks += 1
        self.free_blocks.add(block.bid)

    @traced
    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.
        """
        ret = []
        curr_block = self.free_list_head
        while curr_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret
