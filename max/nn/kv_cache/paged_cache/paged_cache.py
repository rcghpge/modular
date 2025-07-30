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

"""PagedAttention-enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, TypeVar, cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    TensorType,
    TensorValue,
    _OpaqueType,
    _OpaqueValue,
    ops,
)
from max.interfaces.request import RequestID
from max.profiler import traced
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    MemoryTier,
)
from max.support.human_readable_formatter import to_human_readable_bytes
from max.support.math import ceildiv

from ..cache_params import KVCacheParams
from ..context import KVCacheAwareContext
from ..manager import (
    KVCacheInputs,
    KVCacheInputSymbols,
    KVCacheManager,
    RaggedKVCacheInputs,
)
from ..utils import build_max_lengths_tensor
from .block_copy_engine import BlockCopyEngine, BlockCopyMetrics
from .block_manager import BlockManager, SwappingStrategy

logger = logging.getLogger("max.pipelines")

T = TypeVar("T", bound=KVCacheAwareContext)


@dataclass
class PagedCacheInputSymbols(KVCacheInputSymbols):
    kv_blocks: TensorType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_lengths: TensorType


class PagedKVCacheType(_OpaqueType):
    """PagedAttention Mojo KV Cache graph type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a paged KV Cache."""
        super().__init__("PagedKVCache")


class PagedKVCacheCollectionType(_OpaqueType):
    """The graph type for a "view" of the cache for the given sequences in the
    batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our ContinuousKVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    def __init__(self) -> None:
        """Creates an opaque type containing a paged KV cache collection."""
        super().__init__("PagedKVCacheCollection")


class PagedKVCache(_OpaqueValue):
    """PagedAttention Mojo KV cache graph value."""


class PagedKVCacheCollection(_OpaqueValue):
    """The graph value for a view of the KV cache."""


class FetchPagedKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams, **kwargs: Any) -> None:
        self.kv_params = kv_params

    def __call__(
        self,
        blocks: TensorValue,  # NDBuffer[type, 6, Self.blocks_shape]
        cache_lengths: TensorValue,  # NDBuffer[DType.uint32, 1],
        lookup_table: TensorValue,  # NDBuffer[DType.uint32, 2],
        is_cache_empty: TensorValue,
    ) -> PagedKVCacheCollection:
        """Constructs a PagedKVCacheCollection for use downstream."""

        # Explicit validation.
        if blocks.dtype != self.kv_params.dtype:
            msg = (
                f"expected blocks to be dtype: {self.kv_params.dtype}, got"
                f" {blocks.dtype}"
            )
            raise ValueError(msg)

        if blocks.rank != 6:
            msg = f"expected blocks to be of rank 6, got {blocks.rank}"
            raise ValueError(msg)

        # For all tensors other than the blocks tensor, the length should be equivalent
        # to batch size, which is unknown within the graph at this stage.
        if cache_lengths.dtype != DType.uint32:
            msg = f"expected cache lengths to be dtype: uint32, got {cache_lengths.dtype}"
            raise ValueError(msg)

        if cache_lengths.rank != 1:
            msg = f"expected cache lengths to be of rank 1, got {cache_lengths.rank}"
            raise ValueError(msg)

        if lookup_table.dtype != DType.uint32:
            msg = f"expected lookup_table to be dtype: uint32, got {lookup_table.dtype}"
            raise ValueError(msg)

        if lookup_table.rank != 2:
            msg = f"expected lookup_table to be of rank 2, got {lookup_table.rank}"
            raise ValueError(msg)

        return PagedKVCacheCollection(
            ops.custom(
                "mo.kv_collection_ctor.paged",
                device=blocks.device,
                values=[blocks, cache_lengths, lookup_table, is_cache_empty],
                out_types=[PagedKVCacheCollectionType()],
                parameters={
                    "num_heads": self.kv_params.n_kv_heads_per_device,
                    "head_dim": self.kv_params.head_dim,
                    "page_size": int(blocks.shape[3]),
                },
            )[0].opaque
        )


class PagedKVCacheManager(KVCacheManager):
    page_size: int
    """Number of tokens stored per block."""

    total_num_pages: int
    """Total number of blocks allocated per device."""

    device_tensors: list[Tensor]
    """List of tensors holding the KV cache blocks, one per device."""

    host_tensor: Tensor | None
    """Tensor holding the KV cache blocks on the host for swapping (if enabled)."""

    total_num_host_pages: int
    """Total number of blocks allocated on the host for swapping (if enabled)."""

    block_manager: BlockManager
    """Manages allocation, eviction, and reuse of KV cache blocks."""

    enable_prefix_caching: bool
    """Flag indicating if prefix caching (block reuse) is enabled."""

    gpu_device: Device | None
    """An arbitrary GPU device which holds KV cache blocks, should one exist."""

    enable_kvcache_swapping_to_host: bool
    """Flag indicating if swapping blocks to host memory is enabled."""

    @traced
    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: Sequence[Device],
        session: InferenceSession,
        cache_memory: int,
        page_size: int = 128,
        enable_runtime_checks: bool = False,
    ) -> None:
        """
        Args:
            params: The KVCacheParams for the given pipeline.
            max_batch_size: The maximum number of active requests that the
                manager should support.
            max_seq_len: The maximum sequence length we will generate.
            num_layers: The number of layers in the model.
            devices: The devices on which the manager will allocate memory.
            session: The inference session to load ops from.
            cache_memory: The total amount of memory available for caching.
                This is aggregated across all devices.
            page_size: The number of tokens that will be stored in a single block.
            enable_runtime_checks: Whether to enable runtime correctness checks.
        """
        # The number of tokens in a single block.
        self.page_size = page_size

        # The number of bytes that a single page will occupy.
        single_page_size_bytes = (
            2
            * num_layers
            * params.n_kv_heads_per_device
            * params.head_dim
            * page_size
            * params.dtype.size_in_bytes
        )

        # Normalize cache_memory across all devices.
        cache_memory_per_device = cache_memory // len(devices)

        # The total number of blocks we'll have per-device.
        self.total_num_pages = int(
            cache_memory_per_device // single_page_size_bytes
        )

        # Validate that we are allocating enough blocks.
        single_page_size_bytes_str = to_human_readable_bytes(
            single_page_size_bytes
        )
        cache_memory_per_device_str = to_human_readable_bytes(
            cache_memory_per_device
        )
        if self.total_num_pages == 0:
            raise RuntimeError(
                f"Insufficient cache memory to allocate even a single page.\n"
                f"One page requires {single_page_size_bytes_str} but only {cache_memory_per_device_str} are available."
            )

        if max_batch_size > self.total_num_pages:
            memory_needed_str = to_human_readable_bytes(
                max_batch_size * single_page_size_bytes
            )
            logger.warning(
                f"Insufficient cache memory to support a batch containing {max_batch_size} requests with one token per request. "
                f"Need to allocate at least {max_batch_size} pages ({memory_needed_str}), but only have enough memory for {self.total_num_pages} pages ({cache_memory_per_device_str})."
            )

        blocks_needed_for_max_seq_len = ceildiv(max_seq_len, page_size)
        if blocks_needed_for_max_seq_len > self.total_num_pages:
            memory_needed_str = to_human_readable_bytes(
                blocks_needed_for_max_seq_len * single_page_size_bytes
            )
            logger.warning(
                f"Insufficient cache memory to support a batch containing one request at the max sequence length of {max_seq_len} tokens. "
                f"Need to allocate at least {blocks_needed_for_max_seq_len} pages ({memory_needed_str}), but only have enough memory for {self.total_num_pages} pages ({cache_memory_per_device_str})."
            )

        # call our base class constructor
        super().__init__(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
            is_ragged=True,
        )

        # Whether prefix caching is enabled.
        self.enable_prefix_caching = self.params.enable_prefix_caching

        # Whether kvcache swapping to host is enabled
        self.enable_kvcache_swapping_to_host = (
            self.params.enable_kvcache_swapping_to_host
        )

        if (
            self.enable_kvcache_swapping_to_host
            and not self.enable_prefix_caching
        ):
            raise ValueError(
                "KVCache swapping to host is only supported when prefix caching is enabled"
            )

        # Initialize the block buffers for each device.
        self.device_tensors: list[Tensor] = []
        for device in self.devices:
            self.device_tensors.append(
                Tensor(
                    shape=self.block_shape(),  # type: ignore
                    dtype=self.params.dtype,
                    device=device,
                )
            )

        logger.info(
            f"Paged KVCache Manager allocated {self.total_num_pages} device pages using {single_page_size_bytes_str} per page."
        )

        self.host_tensor = None
        self.total_num_host_pages = 0
        if params.enable_kvcache_swapping_to_host:
            GiB = 1024 * 1024 * 1024
            host_kvcache_swap_space_gb = params.host_kvcache_swap_space_gb
            assert host_kvcache_swap_space_gb is not None
            host_kvcache_swap_space_bytes = int(
                host_kvcache_swap_space_gb * GiB
            )

            self.total_num_host_pages = (
                host_kvcache_swap_space_bytes // single_page_size_bytes
            )

            if self.total_num_host_pages == 0:
                human_readable_host_swap_space_gb = to_human_readable_bytes(
                    host_kvcache_swap_space_bytes
                )
                raise RuntimeError(
                    f"Insufficient host swap space to allocate even a single page. "
                    f"One page requires {single_page_size_bytes_str} but only {human_readable_host_swap_space_gb} are available."
                )

            logger.warning(
                "Paged KVCache Manager will offload GPU blocks to host memory. This is an experimental feature and may not work correctly."
            )
            logger.info(
                f"Paged KVCache Manager allocated {self.total_num_host_pages} host pages using {single_page_size_bytes_str} per page."
            )

            # Determine if any of the devices are GPU devices.
            gpu_device: Device | None = None
            for dev in devices:
                if not dev.is_host:
                    gpu_device = dev
                    break

            # Create a kvcache allocation on the host.
            if gpu_device is not None:
                # Use pinned host memory when there is a GPU device.
                # Pinned memory has faster memory transfer speeds compared to
                # pageable memory and allows for the host to issue them
                # asynchronously to the GPU.
                self.host_tensor = Tensor(
                    shape=self.block_shape(self.total_num_host_pages),  # type: ignore
                    dtype=self.params.dtype,
                    device=gpu_device,
                    pinned=True,
                )
            else:
                # Use normal pageable memory when there is no GPU device.
                # Note that the memory transfers will be synchronous.
                self.host_tensor = Tensor(
                    shape=self.block_shape(self.total_num_host_pages),  # type: ignore
                    dtype=self.params.dtype,
                    device=Device.cpu(),
                )

        # Initialize block copy engine.
        self.block_copy_engine: BlockCopyEngine | None = None
        if self.enable_prefix_caching:
            self.block_copy_engine = BlockCopyEngine(
                block_size=self.page_size,
                num_device_blocks=self.total_num_pages,
                device_tensors=self.device_tensors,
                num_host_blocks=self.total_num_host_pages,
                host_tensor=self.host_tensor,
            )

        # Initialize block manager
        device_memory_tier = (
            MemoryTier.MEMORY_TIER_CPU
            if devices[0].is_host
            else MemoryTier.MEMORY_TIER_GPU
        )
        self.block_manager = BlockManager(
            device_memory_tier=device_memory_tier,
            total_num_blocks=self.total_num_pages,
            total_num_host_blocks=self.total_num_host_pages,
            block_size=self.page_size,
            block_copy_engine=self.block_copy_engine,
            enable_prefix_caching=self.params.enable_prefix_caching,
            enable_runtime_checks=enable_runtime_checks,
        )

    @classmethod
    def _block_size_per_token(
        cls, params: KVCacheParams, num_layers: int
    ) -> int:
        return (
            reduce(mul, cls._block_shape(params, 1, 1, num_layers), 1)
            * params.dtype.size_in_bytes
        )

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        # Determine how much size is necessary to store the full cache based on max_batch_size and max_seq_len.
        # If that's less than available_cache_memory, return that.
        # Otherwise, return available_cache_memory.
        # This is to prevent over-allocation on devices with a large amount of free memory (e.g. CPUs).
        assert params.page_size is not None
        block_size_per_token = cls._block_size_per_token(
            params, num_layers
        ) * len(devices)

        # round up our max_seq_len to the nearest page_size
        max_seq_len_round_up = (
            math.ceil(max_seq_len / params.page_size) * params.page_size
        )
        size_to_support_full_cache = (
            block_size_per_token * max_batch_size * max_seq_len_round_up
        )

        # return the minimum of the two
        return min(available_cache_memory, size_to_support_full_cache)

    @classmethod
    def infer_optimal_batch_size(
        cls,
        params: KVCacheParams,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: Sequence[Device],
        **kwargs: Any,
    ) -> int:
        # We just hard-code a default of 512 for paged attention.
        # The worst case scenario if this is too high is that we'll evict
        # requests at an elevated rate. We print warnings in that case so users
        # are aware of what needs to be tweaked/changed.
        return 512

    def block_shape(
        self,
        total_num_pages: int | None = None,
        is_parameterized: bool = False,
    ) -> list[int | str]:
        if total_num_pages is None:
            total_num_pages = self.total_num_pages
        return self._block_shape(
            self.params,
            total_num_pages,
            self.page_size,
            self.num_layers,
            is_parameterized,
        )

    @classmethod
    def _block_shape(
        cls,
        params: KVCacheParams,
        total_num_pages: int,
        page_size: int,
        num_layers: int,
        is_parameterized: bool = False,
    ) -> list[int | str]:
        # split k and v caches across a single dim
        # 0 = key
        # 1 = value
        kv_dim = 2
        return [
            "total_num_pages" if is_parameterized else total_num_pages,
            kv_dim,
            num_layers,
            page_size,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]

    def _get_num_req_slots(self, request_id: RequestID) -> int:
        """Get the number of KV slots available for a request."""
        return (
            len(self.block_manager.current_blocks_per_request[request_id])
            * self.page_size
        )

    @traced
    def _does_req_need_more_blocks(self, ctx: T, num_steps: int) -> bool:
        """Determines if a request needs additional blocks."""
        seq_len = ctx.current_length + num_steps - 1
        return seq_len > self._get_num_req_slots(ctx.request_id)

    @traced
    def prefetch(self, data: T, num_steps: int = 1) -> bool:
        """Prepares blocks for a request prior to a subsequent fetch call.

        This will reuse blocks from prefix cache and allocate new blocks for the
        request. If a request is prefetched, it is guaranteed to not OOM in a
        subsequent call to `fetch`.

        Returns `True` if the request was prefetched, `False` otherwise.
        """
        self.block_manager.reuse_blocks_from_prefix_cache(data)

        # Allocating new blocks can fail if there are insufficient blocks.
        try:
            self.block_manager.allocate_new_blocks(data, num_steps)
        except RuntimeError:
            return False
        return True

    @traced
    def fetch(self, batch: list[T], num_steps: int = 1) -> list[KVCacheInputs]:
        """Reuses blocks from prefix cache and allocates new blocks for requests in batch.

        On cache hits, the input context may have their start_idx bumped upwards in order
        to trim the prompt. Additionally, this method may launch COW memcpy kernel.

        This can fail if there are insufficient blocks to satisfy the batch. In such a case,
        we raise a RuntimeError.
        """

        if self.block_copy_engine is not None:
            self.block_copy_engine.wait_for_completion()

        max_seq_len = -1
        for batch_idx, ctx in enumerate(batch):  # noqa: B007
            # Allocate blocks for request if we need more.
            if self._does_req_need_more_blocks(ctx, num_steps):
                self.block_manager.reuse_blocks_from_prefix_cache(ctx)
                self.block_manager.allocate_new_blocks(ctx, num_steps)

            # Compute the total sequence length
            seq_len = ctx.current_length + num_steps - 1
            if seq_len > self.max_seq_len:
                raise RuntimeError(
                    f"Request has current length ({ctx.current_length}) + num_steps ({num_steps}) - 1 = {seq_len} which exceeds model max_seq_len of {self.max_seq_len}"
                )
            max_seq_len = max(max_seq_len, seq_len)

        # Allocate the buffers containing metadata about the batch.
        # [0, total_num_pages) are the valid block ids and total_num_pages
        # denotes an unassigned block.
        max_num_pages = ceildiv(max_seq_len, self.page_size)
        batch_size = len(batch)
        lut_table_np = np.full(
            (batch_size, max_num_pages), self.total_num_pages, dtype=np.uint32
        )
        cache_lengths_np = np.zeros((batch_size,), dtype=np.uint32)

        # Update cache_lengths and max_lengths.
        max_prompt_len = 0
        max_cached_len = 0
        for batch_idx, ctx in enumerate(batch):
            # Get the blocks for this request.
            blocks = self.block_manager.get_req_blocks(ctx.request_id)

            # Sanity check that we have enough blocks.
            seq_len = ctx.current_length + num_steps - 1
            num_required_blocks = ceildiv(seq_len, self.page_size)
            assert len(blocks) >= num_required_blocks
            if len(blocks) > num_required_blocks:
                blocks = blocks[:num_required_blocks]

            # Vectorized assignment of block indices to lookup table
            lut_table_np[batch_idx, : len(blocks)] = np.array(
                blocks, dtype=np.uint32
            )

            # Get the existing cache length for this sequence.
            cache_length = ctx.start_idx
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far.
            prompt_tokens = ctx.active_length
            max_prompt_len = max(max_prompt_len, prompt_tokens)
            max_cached_len = max(max_cached_len, cache_length + prompt_tokens)

        if self.block_manager.swapping_strategy == SwappingStrategy.EAGER:
            self.block_manager.eagerly_offload_recently_committed_blocks()

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row.
        max_lengths_host = build_max_lengths_tensor(
            num_steps, max_prompt_len, max_cached_len
        )

        # Convert from numpy to host tensors.
        lut_table_host = Tensor.from_numpy(lut_table_np)
        cache_lengths_host = Tensor.from_numpy(cache_lengths_np)

        ret_list = []
        for i, device in enumerate(self.devices):
            ret_list.append(
                RaggedKVCacheInputs(
                    blocks=self.device_tensors[i],
                    cache_lengths=cache_lengths_host.to(device=device),
                    lookup_table=lut_table_host.to(device=device),
                    max_lengths=max_lengths_host,
                )
            )

        return cast(list[KVCacheInputs], ret_list)

    def input_symbols(
        self,
    ) -> list[PagedCacheInputSymbols]:
        return [
            PagedCacheInputSymbols(
                kv_blocks=TensorType(
                    self.params.dtype,
                    shape=self.block_shape(is_parameterized=True),
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=["batch_size"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=["batch_size", "max_num_pages"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                max_lengths=TensorType(
                    DType.uint32,
                    shape=["steps_remaining", 2],
                    device=DeviceRef.CPU(),
                ),
            )
            for i in range(len(self.devices))
        ]

    def release(self, request_id: RequestID) -> None:
        """Release the sequence associated with :obj:`request_id`, marking this sequence as complete.
        This returns the sequence ID back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """
        # Get the sequence ID from the request ID for internal use
        if request_id not in self.request_to_seq_id:
            raise ValueError(
                f"Attempted to release request ID {request_id} but it is not claimed"
            )

        # Call the base class release method with the request_id
        super().release(request_id)
        # Call the block manager release method with the request_id
        self.block_manager.release(request_id)

    @traced
    def step(
        self,
        batch: list[T],
    ) -> None:
        """Commit new tokens into the prefix cache.

        This is a no-op if prefix caching is disabled.
        """
        for ctx in batch:
            # We possibly commit new blocks into the prefix cache.
            self.block_manager.step(ctx)

    @property
    def free_blocks(self) -> set[int]:
        """Get the set of free blocks."""
        return self.block_manager.device_block_pool.free_blocks

    @property
    def used_blocks_pct(self) -> float:
        """Get the percentage of blocks that are in usee."""
        pct = (
            self.total_num_pages - len(self.free_blocks)
        ) / self.total_num_pages
        assert 0 <= pct <= 1
        return pct

    @property
    def host_committed_block_pct(self) -> float:
        """Get the percentage of host blocks that are committed."""
        if self.block_manager.host_block_pool is None:
            return 0
        host_committed_blocks = len(
            self.block_manager.host_block_pool.hash_to_committed_block
        )
        pct = host_committed_blocks / self.total_num_host_pages
        assert 0 <= pct <= 1
        return pct

    @property
    def free_blocks_pct(self) -> float:
        """Get the percentage of blocks that are free."""
        pct = len(self.free_blocks) / self.total_num_pages
        assert 0 <= pct <= 1
        return pct

    @property
    def cache_hit_rate(self) -> float:
        """Get the percentage of prompt tokens that were retrieved from the cache."""
        pct = self.block_manager.cache_hit_rate
        assert 0 <= pct <= 1
        return pct

    def get_req_blocks(self, request_id: RequestID) -> list[int]:
        """Get the block ids for a request."""
        return self.block_manager.get_req_blocks(request_id)

    @property
    def num_blocks_copied(self) -> BlockCopyMetrics:
        """Get the number of blocks copied for each type."""
        if self.block_copy_engine is None:
            return BlockCopyMetrics()
        return self.block_copy_engine.blocks_copied

    def reset_num_blocks_copied(self) -> None:
        """Reset the number of blocks copied for each type."""
        if self.block_copy_engine is None:
            return
        self.block_copy_engine.blocks_copied.reset()
