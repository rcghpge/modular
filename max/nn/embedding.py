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

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationEncoding
from max.nn.comm.allreduce import Allreduce

from .layer import Layer, Module


@dataclass
class EmbeddingV1(Layer):
    """A lookup table for embedding integer indices into dense vectors.

    Deprecated: Use `Embedding` instead.
    """

    weights: TensorValueLike
    device: DeviceRef

    def __call__(self, indices: TensorValueLike) -> TensorValue:
        self.weights = TensorValue(self.weights).to(self.device)
        indices = TensorValue(indices).to(self.device)

        result = ops.gather(self.weights, indices, axis=0)
        if (
            isinstance(self.weights, Weight)
            and self.weights.quantization_encoding is not None
        ):
            result = ops.dequantize(self.weights.quantization_encoding, result)
        return result


class Embedding(Module):
    """
    A lookup table for embedding integer indices into dense vectors.

    This layer maps each integer index to a dense vector of fixed size.
    Embedding weights are stored on the CPU but are moved to the specified
    device during the model init phase.

    Example:

    .. code-block:: python

        embedding_layer = Embedding(
            vocab_size=1000,
            hidden_dim=256,
            dtype=DType.float32,
            device=DeviceRef.GPU(),
            name="embeddings",
        )

        # Token indices of shape: [batch, ..., num_indices].
        token_indices: TensorValueLike
        embeddings = embedding_layer(token_indices)
    """

    weight: Weight
    """The embedding weight matrix stored on the CPU.
    Model init moves weights to the device specified in :obj:`device`."""

    device: DeviceRef
    """The device on which embedding lookup is performed."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        dtype: DType,
        device: DeviceRef,
        quantization_encoding: Optional[QuantizationEncoding] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initializes the embedding layer with the given arguments.

        Args:
            vocab_size: The number of unique items in the vocabulary.
                Indices must be in the range ``[0, vocab_size)``.
            hidden_dim: The dimensionality of each embedding vector.
            dtype: The data type of the embedding weights.
            device: The device where embedding lookups are executed.
                Model init transfers the initially CPU-resident weights to this
                device.
            name: The name identifier for the embedding weight matrix.
        """
        super().__init__()

        self.device = device
        self.weight = Weight(
            name or "weight",
            dtype,
            shape=(vocab_size, hidden_dim),
            device=device,
            quantization_encoding=quantization_encoding,
        )

    def __call__(self, indices: TensorValueLike) -> TensorValue:
        """Embeds the input indices by looking up corresponding vectors.

        Args:
            indices: A tensor of integer indices to look up.
                Each index must be in the range ``[0, vocab_size)``.

        Returns:
            A tensor containing the embeddings corresponding to the input
            indices.
            The result resides on the device specified in :obj:`device`.
        """
        indices = TensorValue(indices).to(self.device)
        weight = TensorValue(self.weight).to(self.device)
        result = ops.gather(weight, indices, axis=0)
        if self.weight.quantization_encoding is not None:
            result = ops.dequantize(self.weight.quantization_encoding, result)
        return result


class VocabParallelEmbedding(Module):
    """
    A lookup table for embedding integer indices into dense vectors.

    This layer works like `nn.Embedding` except the embedding table is sharded
    on the vocabulary dimension across all devices.

    Example:

    .. code-block:: python

        embedding_layer = VocabParallelEmbedding(
            vocab_size=1000,
            hidden_dim=256,
            dtype=DType.float32,
            device=[DeviceRef.GPU(0), DeviceRef.GPU(1)],
            name="embeddings",
        )

        # Token indices of shape: [batch, ..., num_indices].
        token_indices: TensorValueLike
        embeddings = embedding_layer(token_indices)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        dtype: DType,
        devices: list[DeviceRef],
        quantization_encoding: Optional[QuantizationEncoding] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            vocab_size: The number of unique items in the vocabulary.
                Indices must be in the range ``[0, vocab_size)``.
            hidden_dim: The dimensionality of each embedding vector.
            dtype: The data type of the embedding weights.
            devices: The devices where embedding lookups are executed.
                Model init transfers the initially CPU-resident weights to this
                device.
            name: The name identifier for the embedding weight matrix.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.devices = devices

        self.num_devices = len(self.devices)
        self.shard_size = math.ceil(self.vocab_size / self.num_devices)

        # The weight is loaded in with a single op, then copied to each device
        # in __call__.
        self.weight = Weight(
            name or "weight",
            dtype,
            shape=(vocab_size, hidden_dim),
            device=DeviceRef.CPU(),
            quantization_encoding=quantization_encoding,
        )
        self.allreduce = Allreduce(num_accelerators=self.num_devices)

    def __call__(
        self, indices: TensorValueLike, signal_buffers: Iterable[BufferValue]
    ) -> list[TensorValue]:
        """Embeds the input indices by looking up corresponding vectors.

        Args:
            indices: A tensor of integer indices to look up.
                Each index must be in the range ``[0, vocab_size)``.
            signal_buffers: Buffers for peer-to-peer communication in allreduce.

        Returns:
            A tensor containing the embeddings corresponding to the input
            indices.
            The result resides on the device specified in :obj:`device`.
        """
        # Shard the weight onto each device.
        input = TensorValue(indices)
        outputs = [
            self._per_device_call(input, n) for n in range(self.num_devices)
        ]

        return self.allreduce(outputs, signal_buffers)

    def _per_device_call(
        self, indices: TensorValue, device_idx: int
    ) -> TensorValue:
        """Computes the embeddings for the input indices, for a single device."""
        # Copy a shard from the embedding weights onto the device.
        device = self.devices[device_idx]
        vocab_start_index = self.shard_size * device_idx
        vocab_end_index = min(
            self.shard_size * (device_idx + 1), self.vocab_size
        )
        embedding_shard = self.weight[vocab_start_index:vocab_end_index].to(
            device
        )

        indices = indices.to(device)
        # Process indices so that all tokens are between 0 and the shard size.

        # Set up mask so that the 1=tokens within range, 0=tokens out of range.
        input_mask = ops.logical_and(
            indices >= vocab_start_index, indices < vocab_end_index
        )

        # Tokens that are out of this range are masked out.
        indices -= vocab_start_index

        # Apply mask to avoid searching for out-of-bound tokens
        indices *= input_mask

        result = ops.gather(embedding_shard, indices, axis=0)
        if self.weight.quantization_encoding is not None:
            result = ops.dequantize(self.weight.quantization_encoding, result)
        result *= ops.cast(
            ops.unsqueeze(input_mask, -1), result.dtype
        )  # Apply input mask again
        return result
