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

"""Continuous Batching enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

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

from .cache_params import KVCacheParams
from .context import KVCacheAwareContext
from .manager import (
    KVCacheInputs,
    KVCacheInputSymbols,
    KVCacheManager,
    RaggedKVCacheInputs,
)
from .utils import build_max_lengths_tensor

T = TypeVar("T", bound=KVCacheAwareContext)


@dataclass
class ContinuousBatchingKVCacheInputSymbols(KVCacheInputSymbols):
    kv_blocks: TensorType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_lengths: TensorType


class ContinuousBatchingKVCacheType(_OpaqueType):
    """Continuous Mojo KV Cache graph type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a continuous batching KV Cache."""
        super().__init__("ContinuousBatchingKVCache")


class ContinuousBatchingKVCacheCollectionType(_OpaqueType):
    """The graph type for a "view" of the cache for the given sequences in the
    batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our ContinuousKVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    def __init__(self) -> None:
        """Creates an opaque type containing a continuous batching KV cache collection."""
        super().__init__("ContinuousBatchingKVCacheCollection")


class ContinuousBatchingKVCache(_OpaqueValue):
    """Continuous Mojo KV cache graph value."""


class ContinuousBatchingKVCacheCollection(_OpaqueValue):
    """The graph value for a view of the KV cache."""


class FetchContinuousBatchingKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams, **kwargs: Any) -> None:
        if kv_params.enable_prefix_caching:
            raise ValueError(
                "Prefix caching is not supported for continuous batching cache."
            )
        self.kv_params = kv_params

    def __call__(
        self,
        blocks: TensorValue,  # NDBuffer[type, 6, Self.blocks_shape]
        cache_lengths: TensorValue,  # NDBuffer[DType.uint32, 1],
        lookup_table: TensorValue,  # NDBuffer[DType.uint32, 1],
        max_lengths: TensorValue,
    ) -> ContinuousBatchingKVCacheCollection:
        """Constructs a ContinuousBatchingKVCacheCollection for use downstream."""

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
            msg = (
                "expected cache lengths to be dtype: uint32, got"
                f" {cache_lengths.dtype}"
            )
            raise ValueError(msg)

        if lookup_table.dtype != DType.uint32:
            msg = (
                "expected lookup_table to be dtype: uint32, got"
                f" {lookup_table.dtype}"
            )
            raise ValueError(msg)

        return ContinuousBatchingKVCacheCollection(
            ops.custom(
                "mo.kv_collection_ctor.continuous_batching",
                device=blocks.device,
                values=[
                    blocks,
                    cache_lengths,
                    lookup_table,
                    max_lengths,
                ],
                out_types=[ContinuousBatchingKVCacheCollectionType()],
                parameters={
                    "num_heads": self.kv_params.n_kv_heads_per_device,
                    "head_dim": self.kv_params.head_dim,
                },
            )[0].opaque
        )


class ContinuousBatchingKVCacheManager(KVCacheManager):
    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: Sequence[Device],
        session: InferenceSession,
    ) -> None:
        super().__init__(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
            is_ragged=True,
        )

        # Allocate memory for the KV cache blocks.
        self.blocks: list[Tensor] = []
        for i in range(len(self.devices)):
            self.blocks.append(
                Tensor(
                    shape=self.block_shape(self.max_batch_size),
                    dtype=self.params.dtype,
                    device=self.devices[i],
                )
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
        cache_size = (
            reduce(
                mul,
                cls._block_shape(
                    params, max_batch_size, max_seq_len, num_layers
                ),
            )
            * params.dtype.size_in_bytes
        )
        lengths_size = max_batch_size * DType.uint32.size_in_bytes
        lookup_table_size = max_batch_size * DType.uint32.size_in_bytes
        size = cache_size + lengths_size + lookup_table_size
        return size * len(devices)

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
        cache_size_per_sequence = (
            reduce(mul, cls._block_shape(params, 1, max_seq_len, num_layers))
            * params.dtype.size_in_bytes
        )
        return int(available_cache_memory // cache_size_per_sequence)

    def fetch(
        self,
        batch: list[T],
        num_steps: int = 1,
    ) -> list[KVCacheInputs]:
        """Fetches the KV cache state for the given sequence IDs.

        This method retrieves the current cache state for a batch of sequences, including their
        cache lengths and lookup information. It's used during token generation to access
        previously cached key/value pairs.

        Args:
            batch: List of KVCacheAwareContext for which to fetch cache state for.
            num_steps: Number of steps to run for multi-step scheduling.

        Returns:
            List of tuples for each device containing:
            - blocks: Tensor containing the KV cache blocks
            - cache_lengths: Tensor of current cache lengths for each sequence
            - lookup_table: Tensor mapping sequence IDs to cache positions
            - max_lengths: Tensor containing [max_seq_length, max_cache_length]

        Raises:
            ValueError: If any seq_id exceeds max_batch_size or doesn't exist in cache
        """
        active_batch_size = len(batch)

        # Lookup table and seq_ids are redundant identical tensors.
        lookup_table_values = []
        cache_lengths_np = np.zeros(active_batch_size, np.uint32)

        max_seq_length = 0
        max_context_length = 0

        for i, ctx in enumerate(batch):
            if not self.contains(ctx.request_id):
                raise ValueError(
                    f"request_id: {ctx.request_id} not currently in cache."
                )

            seq_id = self.request_to_seq_id[ctx.request_id]
            lookup_table_values.append(seq_id)
            prompt = ctx.next_tokens
            if seq_id > self.max_batch_size:
                msg = (
                    f"seq_id: {seq_id}, beyond max_batch_size, you may"
                    " want to increase `max_batch_size` in the pipeline"
                    " config."
                )
                raise ValueError(msg)

            cache_len = ctx.start_idx

            assert (
                cache_len + len(prompt) + num_steps - 1 <= self.max_seq_len
            ), (
                f"seq_id: {seq_id} would overrun the max cache length of {self.max_seq_len} "
                f"with {len(prompt)} new tokens and {num_steps} steps. Existing length: {cache_len}"
            )

            cache_lengths_np[i] = cache_len

            # Update the maximum lengths seen so far.
            max_seq_length = max(max_seq_length, len(prompt))
            max_context_length = max(
                max_context_length, cache_len + len(prompt)
            )

        cache_lengths = [
            Tensor.from_numpy(cache_lengths_np).to(d) for d in self.devices
        ]
        lookup_table_tensor = Tensor.from_numpy(
            np.array(lookup_table_values, np.uint32)
        )
        lookup_table_tensor_list = [
            lookup_table_tensor.to(self.devices[i])
            for i in range(len(self.devices))
        ]

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row.
        max_lengths_host = build_max_lengths_tensor(
            num_steps, max_seq_length, max_context_length
        )

        result = [
            RaggedKVCacheInputs(
                blocks=self.blocks[i],
                cache_lengths=cache_lengths[i],
                lookup_table=lookup_table_tensor_list[i],
                max_lengths=max_lengths_host,
            )
            for i in range(len(self.devices))
        ]
        return cast(list[KVCacheInputs], result)

    def block_shape(self, n_sequences: int) -> list[int]:
        """Returns the shape of the KV cache blocks for the given number of sequences.

        Defines the 6-dimensional shape of the cache blocks used to store key and value
        tensors for transformer attention. The dimensions represent:
        [n_sequences, 2, num_layers, max_seq_len, n_kv_heads_per_device, head_dim]
        where 2 represents separate storage for keys and values.

        Args:
            n_sequences: Number of sequences that will be cached

        Returns:
            List describing the shape of the cache blocks with dimensions for:
            sequences, key/value split, layers, sequence length, attention heads, and head dimension
        """
        return self._block_shape(
            self.params, n_sequences, self.max_seq_len, self.num_layers
        )

    @staticmethod
    def _block_shape(
        params: KVCacheParams,
        n_sequences: int,
        max_seq_len: int,
        num_layers: int,
    ) -> list[int]:
        return [
            n_sequences,
            2,
            num_layers,
            max_seq_len,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]

    def input_symbols(
        self,
    ) -> list[ContinuousBatchingKVCacheInputSymbols]:
        """Returns the expected input tensor types for `fetch` on each device.

        Defines the tensor specifications needed by the cache implementation,
        including shapes and data types. This is used for graph construction
        and validation.

        Returns:
            List of ContinuousBatchingKVCacheInputSymbols for each device
            containing TensorTypes for:
            - KV cache blocks: 6D tensor for storing keys and values
            - Cache lengths: 1D tensor tracking sequence lengths
            - Lookup table: 1D tensor mapping sequence IDs to cache positions
            - Maximum lengths: 2D tensor tracking maximum sequence and cache lengths per step.
        """
        return [
            ContinuousBatchingKVCacheInputSymbols(
                kv_blocks=TensorType(
                    self.params.dtype,
                    shape=[
                        "num_blocks",
                        2,
                        "num_layers",
                        "max_seq_len",
                        "num_kv_heads",
                        "head_dim",
                    ],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=["batch_size"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=["batch_size"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                # max_lengths (on host)
                max_lengths=TensorType(
                    DType.uint32,
                    shape=["steps_remaining", 2],
                    device=DeviceRef.CPU(),
                ),
            )
            for i in range(len(self.devices))
        ]
