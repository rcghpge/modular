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

"""Multi-layer Perceptron."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.ops.allreduce import matmul_allreduce
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import Weights

from .clamp import clamp
from .comm import Allreduce
from .kernels import (
    dynamic_scaled_matmul,
    matmul_static_scaled_float8,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
    swish_glu,
)
from .layer import Layer, Module, Shardable


class Float8ScaleGranularity(Enum):
    """Specifies the granularity of the quantization scale factor.

    Determines whether a scale factor applies per-tensor, per-row (often for
    weights), per-column, or per-block within a tensor.
    """

    TENSOR = "tensor"
    ROWWISE = "rowwise"
    COLWISE = "colwise"
    BLOCK = "block"


class Float8ScaleOrigin(Enum):
    """Specifies whether the quantization scale is determined statically or dynamically.

    STATIC scales are pre-computed and loaded with the model weights.
    DYNAMIC scales are computed at runtime based on the input data.
    """

    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass
class Float8WeightScaleSpec:
    """Specifies how weights are scaled for float8 quantization."""

    granularity: Float8ScaleGranularity
    """The granularity of the weight scale factor application."""

    dtype: DType
    """The data type of the weight scale factor(s)."""

    @property
    def is_tensor(self) -> bool:
        """Whether the weight scale granularity is per-tensor."""
        return self.granularity == Float8ScaleGranularity.TENSOR

    @property
    def is_rowwise(self) -> bool:
        """Whether the weight scale granularity is row-wise."""
        return self.granularity == Float8ScaleGranularity.ROWWISE

    @property
    def is_colwise(self) -> bool:
        """Whether the weight scale granularity is column-wise."""
        return self.granularity == Float8ScaleGranularity.COLWISE

    @property
    def is_block(self) -> bool:
        """Whether the weight scale granularity is block-wise."""
        return self.granularity == Float8ScaleGranularity.BLOCK


@dataclass
class Float8InputScaleSpec:
    """Specifies how input activations are scaled for float8 quantization."""

    granularity: Float8ScaleGranularity
    """The granularity of the input scale factor application."""

    origin: Float8ScaleOrigin
    """The origin (static or dynamic) of the input scale factor."""

    dtype: DType
    """The data type of the input scale factor(s)."""

    activation_scale_ub: float | None = None
    """An optional upper bound for dynamic activation scaling."""


@dataclass
class Float8Config:
    """Configures float8 quantization settings for a layer or model section."""

    input_scale: Float8InputScaleSpec
    """Specification for input activation scaling."""

    weight_scale: Float8WeightScaleSpec
    """Specification for weight scaling."""

    mlp_in_float8: set[int]
    """Set of layer indices with MLPs in float8.

    MLPs are considered to be either "all quantized" or all not quantized per
    layer.
    So either all of gate proj, down proj, and up proj are float8, or all bfloat16.
    """

    attn_qkv_in_float8: set[int]
    """Set of layer indices with attention QKV projections in float8.

    QKV projections are considered to be either "all quantized" or all not
    quantized per layer.
    So either all of {q,k,v,o}_proj are float8, or all bfloat16.
    """

    embedding_output_dtype: DType | None = None
    """The data type of the output from the embedding layer."""

    quant_method: str | None = None
    """The quantization method used (e.g., "fbgemm_fp8")."""

    @property
    def is_static(self) -> bool:
        """Returns true if this input scale is static."""
        return self.input_scale.origin == Float8ScaleOrigin.STATIC

    @property
    def is_dynamic(self) -> bool:
        """Returns true if this input scale is dynamic."""
        return self.input_scale.origin == Float8ScaleOrigin.DYNAMIC


class Linear(Module, Shardable):
    """
    Applies a linear transformation to incoming data: :math:`y = xW^T + b`.

    This layer implements a fully connected layer where inputs are multiplied
    by a weight matrix and optionally added with a bias vector.
    Both weights and bias initially reside on CPU, and the model init phase
    moves them to :obj:`device`.

    Example:

    .. code-block:: python

        linear_layer = Linear(
            in_dim=256,
            out_dim=128,
            dtype=DType.float32,
            device=DeviceRef.GPU(),
            name="linear",
            has_bias=True
        )

        # Input tensor of shape: [batch, ..., 256]
        input_tensor: TensorValue
        output = linear_layer(input_tensor)
    """

    weight: Weight
    """The weight matrix stored on CPU with shape (out_dim, in_dim).
    Model init transposes the weight and moves it to :obj:`device`."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_dim,).
    Model init moves the bias to :obj:`device` if present."""

    input_scale: Weight | None = None
    """The optional input scale stored on CPU with shape ().
    Model init moves the input_scale to :obj:`device` if present."""

    weight_scale: Weight | None = None
    """The optional weight scale stored on CPU with shape () or (N,).
    Model init moves the weight_scale to :obj:`device` if present."""

    device: DeviceRef
    """The device where matrix operations are performed."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        float8_config: Float8Config | None = None,
        name: str | None = None,
        clip_weight: float | None = None,
    ) -> None:
        """Initializes the linear layer with weights and optional bias.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The data type for both weights and bias.
            device: The target device for computation.
                Weights remain on CPU until moved during computation.
            name: Base name for weights (appended with ``.weight`` and
                ``.bias`` if applicable).
            has_bias: When :obj:`True`, adds a bias vector to the layer.
                Defaults to :obj:`False`.
        """
        super().__init__()

        self.device = device
        self.clip_weight = clip_weight
        self.float8_config = float8_config

        self.weight = Weight(
            name=f"{name}.weight" if name else "weight",
            dtype=dtype,
            shape=(out_dim, in_dim),
            device=device,
            quantization_encoding=quantization_encoding,
        )

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=dtype,
                shape=(out_dim,),
                device=device,
                quantization_encoding=quantization_encoding,
            )

            if self.bias.device != self.weight.device:
                raise ValueError(
                    f"Bias is on device {self.bias.device} while weight is on {self.weight.device}."
                )

        if float8_config:
            if float8_config.is_static:
                self.input_scale = Weight(
                    name=f"{name}.input_scale" if name else "input_scale",
                    dtype=float8_config.input_scale.dtype,
                    shape=(),
                    device=DeviceRef.CPU(),
                    quantization_encoding=quantization_encoding,
                )

            if float8_config.input_scale.granularity not in (
                Float8ScaleGranularity.TENSOR,
                Float8ScaleGranularity.COLWISE,
            ):
                raise ValueError(
                    f"unsupported input scale granularity {float8_config.input_scale.granularity}. "
                    "Only tensor and col-wise are supported, currently"
                )

            weight_scale_shape: tuple[int, ...]
            weight_scale = float8_config.weight_scale
            if weight_scale.is_rowwise:
                weight_scale_shape = (int(self.weight.shape[0]), 1)
            elif weight_scale.is_tensor:
                weight_scale_shape = ()
            else:
                raise ValueError(
                    "only row-wise and tensor scaling are "
                    f"supported currently, but got {weight_scale.granularity}"
                )

            self.weight_scale = Weight(
                name=f"{name}.weight_scale" if name else "weight_scale",
                dtype=weight_scale.dtype,
                # TODO: Pass a per-layer quantization type.
                # For now since we only support row-wise
                shape=weight_scale_shape,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the weight sharding strategy."""
        return self.weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the weight sharding strategy.

        Args:
            strategy: The strategy describing the weight sharding.
        """
        self.weight.sharding_strategy = strategy

        if self.weight_scale:
            # Weight scale should only be added when a float8 config is passed.
            assert self.float8_config

            # When the weight scale is rowwise of shape (M, 1), or tensor of
            # shape (1,), replicate it across devices when weight sharding is
            # colwise.
            should_replicate = self.float8_config.weight_scale.is_tensor or (
                (strategy.is_colwise or strategy.is_head_aware_colwise)
                and self.float8_config.weight_scale.is_rowwise
            )
            self.weight_scale.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
                if should_replicate
                else strategy
            )

        if self.bias:
            # Only truly shard the bias across devices when the weight sharding
            # is rowwise.
            # Otherwise, when the weight sharding is columnwise, set the bias to
            # replicate so that it is complete on device 0.
            # Linear.shard handles setting bias to None on devices >= 1 to
            # prevent bias duplication, which would be incorrect.
            self.bias.sharding_strategy = (
                strategy
                if strategy.is_rowwise
                else ShardingStrategy.replicate(strategy.num_devices)
            )

    def shard(self, shard_idx: int, device: DeviceRef) -> Linear:
        """Creates a sharded view of this Linear layer for a specific device.

        Args:
            shard_idx: The index of the shard (0 to num_devices-1).
            device: The device where this shard should reside.

        Returns:
            A sharded Linear instance.
        """
        if not self.weight.sharding_strategy:
            raise ValueError(
                "Linear layer cannot be sharded because no sharding strategy was provided."
            )

        # Calculate sharded dimensions.
        out_dim = (
            int(self.weight.shape[0])
            // self.weight.sharding_strategy.num_devices
            if self.weight.sharding_strategy.is_rowwise
            else int(self.weight.shape[0])
        )

        # Create new Linear with same configuration.
        sharded = Linear(
            in_dim=int(self.weight.shape[1]),
            out_dim=out_dim,
            dtype=self.weight.dtype,
            device=device,
            has_bias=self.bias is not None,
            float8_config=self.float8_config,
            clip_weight=self.clip_weight,
        )

        # Replace the weights with sharded versions.
        sharded.weight = self.weight.shard(shard_idx, device)

        # Handle bias sharding
        if self.bias is not None:
            # For columnwise sharding with allreduce.sum, only add bias on device 0
            # to avoid adding it multiple times.
            is_colwise = (
                self.weight.sharding_strategy.is_colwise
                or self.weight.sharding_strategy.is_head_aware_colwise
            )
            if is_colwise and (shard_idx > 0):
                sharded.bias = None
            else:
                sharded.bias = self.bias.shard(shard_idx, device)

        # Handle float8 scales.
        if self.float8_config:
            if self.input_scale is not None:
                # Input scale is always shared (scalar), which should be
                # checked upstream.
                assert len(self.input_scale.shape) == 0
                sharded.input_scale = self.input_scale

            if self.weight_scale is not None:
                # Share a reference to the original weight scale if scalar, and
                # shard if on device.
                # This is because scalars are always on CPU by convention.
                sharded.weight_scale = (
                    self.weight_scale
                    if len(self.weight_scale.shape) == 0
                    else self.weight_scale.shard(shard_idx, device)
                )

        return sharded

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies a linear transformation to the input data.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensor must reside on :obj:`device`.

        Returns:
            Output tensor of shape ``(..., out_dim)``.
            The result resides on the device specified in :obj:`device`.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        weight: TensorValue = self.weight
        if self.clip_weight:
            weight = clamp(weight, -self.clip_weight, self.clip_weight)

        if self.weight.quantization_encoding:
            res = ops.qmatmul(
                self.weight.quantization_encoding, None, x, weight
            )
        elif self.float8_config:
            assert self.weight_scale is not None
            weight_scale: TensorValue = self.weight_scale
            if self.input_scale is not None:
                x = quantize_static_scaled_float8(x, self.input_scale)

                input_scale: TensorValue = self.input_scale
                res = matmul_static_scaled_float8(
                    x, weight, input_scale, weight_scale
                )
            else:
                x, x_scales = quantize_dynamic_scaled_float8(
                    x, scales_type=weight_scale.dtype
                )

                if self.device:
                    weight_scale = weight_scale.to(self.device)

                res = dynamic_scaled_matmul(
                    x, weight, x_scales, weight_scale, out_type=DType.bfloat16
                )
        else:
            res = x @ weight.T

        if self.bias is not None:
            res += self.bias
        return res


class ColumnParallelLinear(Linear):
    """A Linear layer where the weight and bias are sharded onto multiple devices.

    This layer first computes :math:`y = xW_i^T + b_i` for each device `i` in
    `[0,..., num_devices]`:

    .. code-block::

        +-----+       +-----+ T     +-----+       +-----+
        |     |       | W_0 |       | b_0 |       | y_0 | GPU0
        |     |       +-----+       +-----+       +-----+
        |     |       | W_1 |       | b_1 |       | y_1 | GPU1
        |  x  |   @   +-----+   +   +-----+   =   +-----+
        |     |       | W_2 |       | b_2 |       | y_2 | GPU2
        |     |       +-----+       +-----+       +-----+
        |     |       | W_3 |       | b_3 |       | y_3 | GPU3
        +-----+       +-----+       +-----+       +-----+

    The values are then collected using an Allgather op, producing the same
    output tensor :math:`y = xW^T + b` on each device:

    .. code-block::

        GPU0  GPU1  GPU2  GPU3                      GPU0  GPU1  GPU2  GPU3
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+
        | y_0 |  -  |  -  |  -  |                   | y_0 | y_0 | y_0 | y_0 |
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+
        |  -  | y_1 |  -  |  -  |                   | y_1 | y_1 | y_1 | y_1 |
        +-----+-----+-----+-----+  -- Allgather --> +-----+-----+-----+-----+
        |  -  |  -  | y_2 |  -  |                   | y_2 | y_2 | y_2 | y_2 |
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+
        |  -  |  -  |  -  | y_3 |                   | y_3 | y_3 | y_3 | y_3 |
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+

    Example usage:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef
        from max.nn import ColumnParallelLinear

        num_devices = 4
        distributed_linear = ColumnParallelLinear(
            in_dim,
            out_dim,
            DType.float32,
            devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        )
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        devices: Sequence[DeviceRef],
        tied_weight: Weight | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The data type for both weights and bias.
            devices: The target devices for computation.
                Weights remain on CPU until sharded and moved to device during
                computation.
        """
        if len(devices) == 0:
            raise ValueError(
                "ColumnParallelLinear requires a non-empty devices argument"
            )

        if tied_weight and (
            kwargs.get("float8_config") is not None
            or kwargs.get("has_bias") is not None
        ):
            raise ValueError(
                "float8 and bias are both unsupported by "
                "ColumnParallelLinear currently"
            )

        super().__init__(in_dim, out_dim, dtype, devices[0], **kwargs)

        if tied_weight:
            # Overwrite the weight we just constructed with the tied weight.
            # In contrast with overriding outside the constructor, this ensures
            # that the sharding strategy captures the tied weight correctly.
            self.weight = tied_weight
            self.set_shared_weight("weight", tied_weight)

        self.devices = devices
        self.num_devices = len(self.devices)

        self.sharding_strategy = ShardingStrategy.rowwise(self.num_devices)

        # Create normal Linear layers for each device. These layers and weights
        # are not recorded by the nn.Module and do not appear in the state dict.
        self.distributed_linear_layers = []
        for n, device in enumerate(self.devices):
            layer = Linear(in_dim, out_dim, dtype, device, **kwargs)
            layer.device = device
            layer.weight = self.weight.shard(n, device)
            if self.bias is not None:
                layer.bias = self.bias.shard(n, device)
            self.distributed_linear_layers.append(layer)

    def __call__(  # type: ignore[override]
        self, x: Sequence[TensorValue], signal_buffers: Iterable[BufferValue]
    ) -> list[TensorValue]:
        """Applies a linear transformation to the input data.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensor must reside on :obj:`device`.
            signal_buffers: Buffers for peer-to-peer communication in allgather.

        Returns:
            Output tensor of shape ``(..., out_dim)``.
            The result resides on the device specified in :obj:`device`.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        linear_outs = [
            self.distributed_linear_layers[i](x[i])
            for i in range(self.num_devices)
        ]
        return ops.allgather(linear_outs, signal_buffers, axis=-1)


def _allocate_if_needed(value: Weights | Weight, dtype, shape) -> Weight:
    if isinstance(value, Weight):
        return value
    else:
        return value.weight.allocate(dtype, shape)


@dataclass
class LinearV1(Layer):
    """A unified linear layer that delegates to either regular or quantized implementation.

    Deprecated: Use `Linear` instead.
    """

    weight: TensorValueLike
    bias: TensorValueLike | None = None

    def __call__(self, x: TensorValue) -> TensorValue:
        weight = TensorValue(self.weight)
        if weight.type.device != x.type.device:
            weight = weight.to(x.type.device)
        res = x @ weight.T
        if self.bias is not None:
            bias = TensorValue(self.bias)
            if bias.type.device != x.type.device:
                bias = bias.to(x.type.device)
            res += bias
        return res

    @classmethod
    def create(
        cls,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        in_features: int,
        out_features: int,
        weights: Weights | Weight,
        bias: Weights | Weight | None = None,
        quantization_config: QuantizationConfig | None = None,
    ) -> LinearV1:
        """Factory method to create a Linear layer with appropriate implementation."""
        if not quantization_encoding:
            weight = _allocate_if_needed(
                weights, dtype, [in_features, out_features]
            )
            bias_weight = (
                _allocate_if_needed(bias, dtype, [out_features])
                if bias
                else None
            )
            return LinearV1(weight=weight, bias=bias_weight)
        else:
            return QLinearV1._create(
                dtype,
                quantization_encoding,
                in_features,
                out_features,
                weights,
                bias,
                quantization_config,
            )


@dataclass
class QLinearV1(LinearV1):
    """A quantized fully connected layer."""

    # Because Linear.bias is optional and Linear is a dataclass and we inherit from Linear, all our fields must be optional even if it doesn't make logical sense
    quantization_encoding: QuantizationEncoding | None = None

    @classmethod
    def _create(
        cls,
        dtype: DType,
        quantization_encoding: QuantizationEncoding,
        in_features: int,
        out_features: int,
        weights: Weights | Weight,
        bias: Weights | Weight | None,
        quantization_config: QuantizationConfig | None,
    ) -> LinearV1:
        if quantization_encoding != QuantizationEncoding.GPTQ:
            weight = _allocate_if_needed(
                weights, dtype, [in_features, out_features]
            )
            bias_weight = (
                _allocate_if_needed(bias, dtype, [out_features])
                if bias
                else None
            )
            return QLinearV1(
                weight=weight,
                bias=bias_weight,
                # GGUF weights can have different quantization per weight
                quantization_encoding=weight.quantization_encoding,
            )
        else:
            return GPTQLinearV1._create(
                dtype,
                quantization_encoding,
                in_features,
                out_features,
                weights,
                bias,
                quantization_config,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        assert self.quantization_encoding is not None
        weight = TensorValue(self.weight)
        weight = weight.to(x.type.device)
        res = ops.qmatmul(self.quantization_encoding, None, x, weight)
        if self.bias is not None:
            bias = TensorValue(self.bias).to(x.type.device or DeviceRef.CPU())
            res += bias
        return res


@dataclass
class GPTQLinearV1(QLinearV1):
    "A Linear layer for GPTQ encoding"

    # Because QLinear has optional fields, so must we, since we subclass QLinear
    quantization_config: QuantizationConfig | None = None
    perm_idx: TensorValueLike | None = None

    @classmethod
    def _create(
        cls,
        dtype: DType,
        quantization_encoding: QuantizationEncoding,
        in_features: int,
        out_features: int,
        weights: Weights | Weight,
        bias: Weights | Weight | None,
        quantization_config: QuantizationConfig | None,
    ) -> LinearV1:
        """Internal method to create a Linear layer from GPTQ weights."""

        assert quantization_config, (
            "QuantizationConfig must be provided for GPTQLinear"
        )

        assert quantization_config.sym, "GPTQ with sym=False is not supported."

        desc_act = quantization_config.desc_act

        perm_idx = None

        if isinstance(weights, Weights) and weights.qweight.exists():
            orig_quantized_weights = [weights.qweight, weights.scales]
            quantized_weights = []
            for idx, qw in enumerate(orig_quantized_weights):
                orig = qw.allocate()
                # TODO(AITLIB-135): allocate_as_bytes is only available for
                # safetensors. This isn't a problem right now because gptq is
                # only present for safetensors
                weight_bytes = qw.allocate_as_bytes()  # type: ignore
                assert len(orig.shape) == 2
                reshaped = ops.reshape(
                    weight_bytes,
                    (orig.shape[0] * orig.dtype.size_in_bytes, orig.shape[1]),
                ).transpose(0, 1)
                quantized_weights.append(reshaped)

            weight = ops.concat(
                (quantized_weights[0], quantized_weights[1]), axis=1
            ).transpose(0, 1)

            if desc_act:
                perm_idx = weights.g_idx.allocate(DType.int32, [out_features])
                # hack: argsort the perm_idx array
                weights._allocated[perm_idx.name] = np.argsort(  # type: ignore
                    weights._allocated[perm_idx.name]  # type: ignore
                ).astype(np.int32)

            return GPTQLinearV1(
                weight=weight,
                bias=None,
                quantization_encoding=quantization_encoding,
                quantization_config=quantization_config,
                perm_idx=perm_idx,
            )

        else:
            weight = _allocate_if_needed(
                weights, DType.bfloat16, [in_features, out_features]
            )
            bias_weight = (
                _allocate_if_needed(bias, dtype, [out_features])
                if bias
                else None
            )
            return LinearV1(weight, bias_weight)

    def __call__(self, x: TensorValue) -> TensorValue:
        assert self.quantization_encoding is not None
        weight = TensorValue(self.weight)
        if self.perm_idx is not None:
            perm_idx = TensorValue(self.perm_idx)
            res = ops.qmatmul(
                self.quantization_encoding,
                self.quantization_config,
                ops.gather(x, perm_idx, axis=(x.rank - 1)),
                weight,
                perm_idx,
            )
        else:
            res = ops.qmatmul(
                self.quantization_encoding, self.quantization_config, x, weight
            )
        if self.bias is not None:
            res += TensorValue(self.bias)
        return res


@dataclass
class GPTQLinear(Linear):
    "A Linear layer for GPTQ encoding"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        quantization_config: QuantizationConfig | None = None,
        float8_config: Float8Config | None = None,
    ) -> None:
        """Initializes the linear layer with weights and optional bias with
        GPTQ quantization.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The data type for both weights and bias.
            device: The target device for computation.
                Weights remain on CPU until moved during computation.
            has_bias: When :obj:`True`, adds a bias vector to the layer.
                Defaults to :obj:`False`.
            quantization_encoding: The quantization encoding of the weights.
            quantization_config: Extra config for the weight quantization.
        """
        del out_dim, dtype  # Unused.
        if has_bias:
            raise ValueError("has_bias=True is not supported in GPTQLinear.")
        if float8_config:
            raise ValueError("Float8 is not supported in GPTQLinear.")

        # Skip Linear initialization.
        Module.__init__(self)
        self.device = device
        self.qweight = Weight(
            name="qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            device=device,
            quantization_encoding=quantization_encoding,
        )
        self.scales = Weight(
            name="scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            device=device,
            quantization_encoding=quantization_encoding,
        )

        assert quantization_config, (
            "QuantizationConfig must be provided for GPTQLinear"
        )
        assert quantization_config.sym, "GPTQ with sym=False is not supported."

        self.quantization_config = quantization_config

        desc_act = self.quantization_config.desc_act
        self.perm_idx = None
        if desc_act:
            self.perm_idx = Weight(
                "perm_idx", DType.int32, [in_dim], device=device
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        assert self.qweight.quantization_encoding is not None
        qweight_dtype, qweight_shape = self.qweight.original_dtype_and_shape
        qweight = ops.reshape(
            self.qweight,
            (qweight_shape[0] * qweight_dtype.size_in_bytes, qweight_shape[1]),
        ).transpose(0, 1)

        scales_dtype, scales_shape = self.scales.original_dtype_and_shape
        scales = ops.reshape(
            self.scales,
            (scales_shape[0] * scales_dtype.size_in_bytes, scales_shape[1]),
        ).transpose(0, 1)
        weight = ops.concat((qweight, scales), axis=1).transpose(0, 1)
        if self.device:
            weight = weight.to(self.device)
        if self.perm_idx is not None:
            perm_idx: TensorValue = self.perm_idx
            if self.device:
                perm_idx = perm_idx.to(self.device)
            res = ops.qmatmul(
                self.qweight.quantization_encoding,
                self.quantization_config,
                ops.gather(x, perm_idx, axis=(x.rank - 1)),
                weight,
                perm_idx,
            )
        else:
            res = ops.qmatmul(
                self.qweight.quantization_encoding,
                self.quantization_config,
                x,
                weight,
            )
        if self.bias is not None:
            res += TensorValue(self.bias)
        return res


@dataclass
class MLPV1(Layer):
    """
    Simple multi-layer perceptron composed of three linear layers.
    Uses SiLU activation function.
    """

    gate_proj: LinearV1
    down_proj: LinearV1
    up_proj: LinearV1

    def __call__(self, x: TensorValueLike) -> TensorValue:
        if (
            self.gate_proj.bias is None
            and self.up_proj.bias is None
            and TensorValue(x).rank == 2
            and TensorValue(x).device is not None
            and TensorValue(x).device != DeviceRef.CPU()
            and False  # GEX-1476: This causes elaboration errors - disable swish_glu pathway.
        ):
            return self.down_proj(
                swish_glu(x, self.gate_proj.weight, self.up_proj.weight)
            )

        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))  # type: ignore


_ACTIVATION_FUNCTIONS = {
    "silu": ops.silu,
    "gelu": ops.gelu,
    "gelu_tanh": partial(ops.gelu, approximate="tanh"),
    "relu": ops.relu,
    "tanh": ops.tanh,
    "sigmoid": ops.sigmoid,
}


@dataclass
class DistributedGemmConfig:
    """Configure how distributed GEMM is executed"""

    # Required fields

    # If True, use the matmul + all_reduce kernel
    enable_matmul_allreduce: bool

    @staticmethod
    def generate() -> DistributedGemmConfig | None:
        """Returns the default DistributedGemmConfig"""
        opts_env = os.getenv("LLAMA_ENABLE_DIST_GEMM_KERNELS")
        if opts_env is None:
            return DistributedGemmConfig(True)

        enable_matmul_allreduce = bool(opts_env)
        return DistributedGemmConfig(enable_matmul_allreduce)


class MLP(Module):
    """
    Simple multi-layer perceptron composed of three linear layers.
    Defaults to SiLU activation function.
    """

    def __init__(
        self,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        hidden_dim: int,
        feed_forward_length: int,
        devices: Sequence[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        has_bias: bool = False,
        activation_function: str = "silu",
        float8_config: Float8Config | None = None,
        dist_gemm_config: DistributedGemmConfig | None = None,
    ) -> None:
        """
        Args:
            dtype: DType to use for the layer weights, which should match the
                input dtype.
            quantization_encoding: Quantization encoding of the layer weights.
            hidden_dim: The last dimension of the layer input.
            feed_forward_length: Size of dimension used to project the inputs.
            linear_cls: Linear class to use to create the projection layers.
            devices: Devices to run the `MLP` layer. If multiple are provided,
                the first device is used instead. Use `DistributedMLP` to use
                all devices.
            activation_function: Activation function to use. Options are:
                - "silu"
                - "gelu"
                - "gelu_tanh"
                - "relu"
                - "tanh"
                - "sigmoid"
        """
        super().__init__()
        self.devices = devices
        self.dist_gemm_config = dist_gemm_config
        self.gate_proj = linear_cls(  # [ffl, hidden]
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.down_proj = linear_cls(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.up_proj = linear_cls(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.quantization_encoding = quantization_encoding
        self.float8_config = float8_config
        assert activation_function in _ACTIVATION_FUNCTIONS.keys()
        self.activation_function = _ACTIVATION_FUNCTIONS[activation_function]

    def __call__(self, x: TensorValueLike) -> TensorValue:
        if (
            self.gate_proj.bias is None
            and self.up_proj.bias is None
            and TensorValue(x).rank == 2
            and TensorValue(x).device is not None
            and TensorValue(x).device != DeviceRef.CPU()
            and False  # GEX-1476: This causes elaboration errors - disable swish_glu pathway.
        ):
            return self.down_proj(
                swish_glu(x, self.gate_proj.weight, self.up_proj.weight)
            )
        if self.quantization_encoding or self.float8_config:
            return self.down_proj(
                self.activation_function(self.gate_proj(TensorValue(x)))
                * self.up_proj(TensorValue(x))
            )
        else:
            # Optimization to compute a single matmul by merging the
            # gate and up projection weights.
            feed_forward_length = self.gate_proj.weight.shape[0]
            gate_proj_weight: TensorValue = self.gate_proj.weight
            if self.gate_proj.device:
                gate_proj_weight = gate_proj_weight.to(self.gate_proj.device)
            up_proj_weight: TensorValue = self.up_proj.weight
            if self.up_proj.device:
                up_proj_weight = up_proj_weight.to(self.up_proj.device)

            bias = None
            if (
                self.gate_proj.bias is not None
                and self.up_proj.bias is not None
            ):
                gate_proj_bias: TensorValue = self.gate_proj.bias
                if self.gate_proj.device:
                    gate_proj_bias = gate_proj_bias.to(self.gate_proj.device)
                up_proj_bias: TensorValue = self.up_proj.bias
                if self.up_proj.device:
                    up_proj_bias = up_proj_bias.to(self.up_proj.device)
                bias = ops.concat((gate_proj_bias, up_proj_bias))

            if bias is not None:
                output = (
                    x @ ops.concat((gate_proj_weight, up_proj_weight)).T
                ) + bias
            else:
                output = x @ ops.concat((gate_proj_weight, up_proj_weight)).T

            gate_out, up_out = ops.split(
                output, [feed_forward_length, feed_forward_length], axis=1
            )

            hidden = self.activation_function(gate_out) * up_out
            # If we overlap GEMM / AllReduce, the last linear layer is skipped.
            if (
                self.dist_gemm_config is None
                or not self.dist_gemm_config.enable_matmul_allreduce
            ):
                return self.down_proj(hidden)
            else:
                return hidden


class DistributedMLP(MLP):
    """A distributed multi-layer perceptron.

    This class has the same state keys as the non-distributed MLP Layer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if kwargs.get("has_bias"):
            raise ValueError(
                "has_bias=True is not supported in DistributedMLP."
            )

        self.num_devices = len(self.devices)

        self.gate_proj.sharding_strategy = ShardingStrategy.rowwise(
            self.num_devices
        )
        self.down_proj.sharding_strategy = ShardingStrategy.columnwise(
            self.num_devices
        )
        self.up_proj.sharding_strategy = ShardingStrategy.rowwise(
            self.num_devices
        )

        # Create normal MLP layers for each device. These layers and weights are
        # not recorded by the nn.Module and do not appear in the state dict.
        self.list_of_mlps = []
        for n, device in enumerate(self.devices):
            layer = MLP(*args, **kwargs)

            layer.gate_proj = self.gate_proj.shard(n, device)
            layer.down_proj = self.down_proj.shard(n, device)
            layer.up_proj = self.up_proj.shard(n, device)

            self.list_of_mlps.append(layer)

        self.allreduce = Allreduce(num_accelerators=len(self.devices))

    def __call__(  # type: ignore[override]
        self, x: Sequence[TensorValue], signal_buffers: Iterable[BufferValue]
    ) -> list[TensorValue]:
        """Applies a linear transformation to the input data.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensor must reside on :obj:`device`.
            signal_buffers: Buffers for peer-to-peer communication in allreduce.

        Returns:
            Output tensor of shape ``(..., out_dim)``.
            The result resides on the device specified in :obj:`device`.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        mlp_outs = [self.list_of_mlps[i](x[i]) for i in range(self.num_devices)]

        dist_gemm_cfg = self.list_of_mlps[0].dist_gemm_config
        if dist_gemm_cfg is None or not dist_gemm_cfg.enable_matmul_allreduce:
            return self.allreduce(mlp_outs, signal_buffers)

        # Special matmul + allreduce split version
        # extract the sharded weights from the last linear layers
        weights = [layer.down_proj.weight for layer in self.list_of_mlps]
        return matmul_allreduce(
            mlp_outs,
            weights,
            signal_buffers,
        )
