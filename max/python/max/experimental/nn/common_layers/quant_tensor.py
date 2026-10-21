# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Quantized tensor wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Any, TypeAlias, TypeGuard

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.module import module_dataclass
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import Shape, TensorValue
from max.nn.kernels import quantize_dynamic_block_scaled
from typing_extensions import Self

MAX_E2M1 = 6.0
MAX_E4M3 = 448.0


def _ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


class QTensor(Module[[], None]):
    """Base class for quantized tensor wrappers.

    QTensors are :class:`~max.experimental.nn.Module` subclasses so that
    parameter discovery finds the inner tensors. They are data wrappers, not
    callable layers — pass them to a quantized kernel op
    (such as ``quant_ops.matmul``).
    """

    def forward(self) -> None:
        """Raises, since a QTensor is a data wrapper rather than a layer."""
        raise NotImplementedError("QTensor is not a callable layer")

    @property
    def local_shards(self) -> tuple[Self, ...]:
        """Per-device shards of this tensor; subclasses implement it."""
        raise NotImplementedError("QTensor does not support local_shards")

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh the tensor is placed on; subclasses implement it."""
        raise NotImplementedError("mesh is not implemented for QTensor")

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        """Returns a copy placed on ``target``; subclasses implement it."""
        raise NotImplementedError("to() is not implemented for QTensor")


class _QuantizedWeight(QTensor):
    """A quantized weight whose fields carry the checkpoint's names.

    A checkpoint stores a quantized projection's packed data as its
    ``weight``, beside the scales, and the fields follow suit. Held as
    ``proj``, the weight loads ``proj.weight`` and ``proj.weight_scale``;
    held by a :class:`~.quant_linear.QuantizedLinear`, which adds no path
    segment for it, it loads the same names.
    """

    weight: Tensor


class FP8BlockTensor(_QuantizedWeight):
    """FP8 block-scaled quantized tensor.

    Args:
        weight: Packed ``float8_e4m3fn`` data of shape ``[rows, cols]``.
        weight_scale_inv: Per-block inverse ``float32`` scales of shape
            ``[ceil(rows / block_m), ceil(cols / block_k)]``.
        block_size: Per-block size as ``(block_m, block_k)``.
    """

    weight_scale_inv: Tensor

    def __init__(
        self,
        *,
        weight: Tensor,
        weight_scale_inv: Tensor,
        block_size: tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.weight = weight
        self.weight_scale_inv = weight_scale_inv
        self._block_size = block_size

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        *,
        block_size: tuple[int, int] = (128, 128),
    ) -> FP8BlockTensor:
        """Builds a :class:`FP8BlockTensor` of the given shape, with all zeros."""
        rows, cols = int(shape[0]), int(shape[1])
        block_m, block_k = block_size
        return cls(
            weight=Tensor.zeros((rows, cols), dtype=DType.float8_e4m3fn),
            weight_scale_inv=Tensor.zeros(
                (_ceildiv(rows, block_m), _ceildiv(cols, block_k)),
                dtype=DType.float32,
            ),
            block_size=block_size,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        """Quantization block shape as ``(rows, cols)``."""
        return self._block_size

    def shard(self, axis: int, mesh: DeviceMesh) -> FP8BlockTensor:
        """Co-shard both leaves along ``axis`` onto ``mesh``.

        Args:
            axis: Tensor axis to shard along (same index into both
                ``weight.shape`` and ``weight_scale_inv.shape``).
            mesh: 1-D :class:`DeviceMesh` to scatter onto.

        Returns:
            A new :class:`FP8BlockTensor` whose ``weight`` and
            ``weight_scale_inv`` leaves are each distributed
            ``Sharded(axis=axis)`` across ``mesh``.
        """
        mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
        return FP8BlockTensor(
            weight=self.weight.to(mapping),
            weight_scale_inv=self.weight_scale_inv.to(mapping),
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[FP8BlockTensor, ...]:
        """Per-device shards, pairing each ``weight`` shard with its scales."""
        return tuple(
            FP8BlockTensor(
                weight=self.weight.local_shards[i],
                weight_scale_inv=self.weight_scale_inv.local_shards[i],
                block_size=self.block_size,
            )
            for i in range(self.weight.num_shards)
        )

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh of the packed ``weight`` tensor."""
        return self.weight.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.weight.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.weight._mapping = mapping
        self.weight_scale_inv._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> FP8BlockTensor:
        """Returns a copy with ``weight`` and its scales placed on ``target``."""
        return FP8BlockTensor(
            weight=self.weight.to(target),
            weight_scale_inv=self.weight_scale_inv.to(target),
            block_size=self.block_size,
        )


class NVFP4Tensor(_QuantizedWeight):
    """NVFP4 (modelopt) block-scaled quantized tensor.

    NVFP4 packs two ``float4_e2m1fn`` values per ``uint8`` and applies two
    scale levels: a per-16-element block scale (``float8_e4m3fn``) and a
    per-tensor global scale (``weight_scale_2``, ``float32``). The activation
    scale (``input_scale``, ``float32``) is *static* — loaded from the
    checkpoint — and used to normalize activations before dynamic block
    quantization (see :func:`~.quant_ops._matmul_nvfp4`).

    Args:
        weight: Packed ``uint8`` weight data of shape ``[rows, cols // 2]``.
        weight_scale: Per-block ``float8_e4m3fn`` scales of shape
            ``[ceil(rows / block_m), ceil(cols / block_k)]``.
        weight_scale_2: Per-tensor ``float32`` global scale, scalar ``()``
            (or ``[num_experts]`` when stacked for grouped MoE).
        input_scale: Static per-tensor ``float32`` activation scale, scalar
            ``()`` (or ``[num_experts]`` when stacked).
        block_size: Per-block size as ``(block_m, block_k)``; ``(1, 16)`` for
            NVFP4.
    """

    weight_scale: Tensor
    weight_scale_2: Tensor
    input_scale: Tensor

    def __init__(
        self,
        *,
        weight: Tensor,
        weight_scale: Tensor,
        weight_scale_2: Tensor,
        input_scale: Tensor,
        block_size: tuple[int, int] = (1, 16),
    ) -> None:
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self.weight_scale_2 = weight_scale_2
        self.input_scale = input_scale
        self._block_size = block_size

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int],
        *,
        block_size: tuple[int, int] = (1, 16),
    ) -> NVFP4Tensor:
        """Builds a zero-filled :class:`NVFP4Tensor` of logical ``shape``.

        ``shape`` is the *unpacked* Linear-convention ``[out_dim, in_dim]``.
        The packed ``weight`` leaf has half the trailing dimension.
        """
        rows, cols = int(shape[0]), int(shape[1])
        block_m, block_k = block_size
        return cls(
            weight=Tensor.zeros((rows, cols // 2), dtype=DType.uint8),
            weight_scale=Tensor.zeros(
                (_ceildiv(rows, block_m), _ceildiv(cols, block_k)),
                dtype=DType.float8_e4m3fn,
            ),
            weight_scale_2=Tensor.zeros((), dtype=DType.float32, device=CPU()),
            input_scale=Tensor.zeros((), dtype=DType.float32, device=CPU()),
            block_size=block_size,
        )

    @property
    def block_size(self) -> tuple[int, int]:
        """Quantization block shape as ``(rows, cols)``."""
        return self._block_size

    def shard(self, axis: int, mesh: DeviceMesh) -> NVFP4Tensor:
        """Co-shard ``weight``/``weight_scale`` along ``axis`` onto ``mesh``."""
        mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
        return NVFP4Tensor(
            weight=self.weight.to(mapping),
            weight_scale=self.weight_scale.to(mapping),
            weight_scale_2=self.weight_scale_2,
            input_scale=self.input_scale,
            block_size=self.block_size,
        )

    @property
    def local_shards(self) -> tuple[NVFP4Tensor, ...]:
        """Per-device shards, pairing each ``weight`` shard with its scales."""
        n = self.weight.num_shards
        return tuple(
            NVFP4Tensor(
                weight=self.weight.local_shards[i],
                weight_scale=self.weight_scale.local_shards[i],
                weight_scale_2=self.weight_scale_2,
                input_scale=self.input_scale,
                block_size=self.block_size,
            )
            for i in range(n)
        )

    @property
    def mesh(self) -> DeviceMesh:
        """Device mesh of the packed ``weight`` tensor."""
        return self.weight.mesh

    @property
    def _mapping(self) -> DeviceMapping:
        return self.weight.mapping

    @_mapping.setter
    def _mapping(self, mapping: DeviceMapping) -> None:
        self.weight._mapping = mapping
        self.weight_scale._mapping = mapping

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> NVFP4Tensor:
        """Returns a copy with ``weight`` and ``weight_scale`` on ``target``."""
        return NVFP4Tensor(
            weight=self.weight.to(target),
            weight_scale=self.weight_scale.to(target),
            # weight_scale_2 and input_scale stay on CPU
            weight_scale_2=self.weight_scale_2,
            input_scale=self.input_scale,
            block_size=self.block_size,
        )


class NVFP4TensorQuantizer(QTensor):
    """FakeQuant-style weight that quantizes to NVFP4 on first use.

    Unlike :class:`NVFP4Tensor`, which holds a checkpoint's pre-quantized
    weight and scales, this holds the checkpoint's native-precision
    ``weight``. :attr:`quant_tensor` quantizes it on first use and caches the
    :class:`NVFP4Tensor`, which quant ops dispatch on, so repeated matmuls,
    eager or traced, quantize once.
    """

    weight: Tensor

    def __init__(
        self,
        tensor: Tensor,
        block_size: tuple[int, int] = (1, 16),
    ):
        if int(tensor.shape[-1]) % block_size[-1]:
            raise ValueError(
                f"NVFP4 needs a K that is a multiple of {block_size[-1]}, got "
                f"{tensor.shape[-1]}"
            )
        self.weight = tensor
        self._block_size = block_size

    @property
    def shape(self) -> Shape:
        """Delegates shape to underlying tensor."""
        return self.weight.shape

    def __tree_flatten__(self) -> tuple[dict[str, Any], None]:
        # Not vars(self): that also holds the cached quant_tensor once it is
        # computed, which is no weight to load.
        return {"weight": self.weight, "_block_size": self._block_size}, None

    def to(
        self, target: Device | DeviceMesh | DeviceMapping
    ) -> NVFP4TensorQuantizer:
        """Delegates placement to underlying tensor."""
        self.weight = self.weight.to(target)
        return self

    @property
    def block_size(self) -> tuple[int, int]:
        """Quantization block shape as ``(rows, cols)``."""
        return self._block_size

    def _quantize(self) -> NVFP4Tensor:
        amax = (
            F.abs(self.weight)
            .max(axis=None)
            .reshape([])
            .cast(DType.float32)
            .to(CPU())
        )
        encode_scale = (MAX_E2M1 * MAX_E4M3) / amax
        decode_scale = 1.0 / encode_scale

        packed_value, scales_value = quantize_dynamic_block_scaled(
            TensorValue(self.weight),
            tensor_sf=TensorValue(encode_scale),
            sf_vector_size=self.block_size[1],
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,
        )
        packed = Tensor.from_graph_value(packed_value)
        scales = Tensor.from_graph_value(scales_value)
        rows = self.weight.shape[0]
        scale_cols = self.weight.shape[1] // self.block_size[1]
        weight_scale = scales.permute([0, 3, 2, 1, 4]).reshape(
            [rows, scale_cols]
        )
        input_scale = Tensor.ones((), dtype=DType.float32, device=CPU())

        return NVFP4Tensor(
            weight=packed,
            weight_scale=weight_scale,
            weight_scale_2=decode_scale,
            input_scale=input_scale,
        )

    @cached_property
    def quant_tensor(self) -> NVFP4Tensor:
        """The NVFP4 quantization of ``weight``, built on first use."""
        return self._quantize()


@module_dataclass(frozen=True)
class NVFP4Activation(QTensor):
    """NVFP4-quantized activation tensor.

    Args:
        data: Packed ``uint8`` FP4 tokens of shape ``[total_tokens, K / 2]``.
        scales: Block scales in the padded SF-atom layout.
        input_scale: The ``float32`` scale these tokens were quantized with.
        scales_offset: Per-expert offset into the padded ``scales``.
    """

    data: Tensor
    scales: Tensor
    input_scale: Tensor
    scales_offset: Tensor


@module_dataclass(frozen=True)
class FP8BlockActivation(QTensor):
    """FP8 block-quantized activation tensor.

    Args:
        data: ``float8_e4m3fn`` tokens of shape ``[total_tokens, K]``.
        scales: ``float32`` scales, one per ``(1, block_k)`` block.
    """

    data: Tensor
    scales: Tensor


QuantAwareTensor: TypeAlias = (
    Tensor
    | FP8BlockTensor
    | NVFP4Tensor
    | NVFP4TensorQuantizer
    | NVFP4Activation
    | FP8BlockActivation
)


def all_fp8_block(
    weights: Sequence[QuantAwareTensor],
) -> TypeGuard[Sequence[FP8BlockTensor]]:
    """Narrow a sequence of mixed weights to a homogeneous FP8 sequence."""
    return all(isinstance(w, FP8BlockTensor) for w in weights)


def all_nvfp4(
    weights: Sequence[QuantAwareTensor],
) -> TypeGuard[Sequence[NVFP4Tensor]]:
    """Narrow a sequence of mixed weights to a homogeneous NVFP4 sequence."""
    return all(isinstance(w, NVFP4Tensor) for w in weights)
