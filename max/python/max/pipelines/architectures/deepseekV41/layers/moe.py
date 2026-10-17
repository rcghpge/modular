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
"""DeepSeek-V4.1 mixture-of-experts feed-forward block.

Each layer routes every token to a few quantized experts, chosen by a scored
gate with a separate bias for image tokens, and adds one always-active shared
expert. Experts can be split across devices (expert parallelism).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.activation import GatedActivation
from max.nn.comm.ep.ep_kernels import fused_silu_quantized
from max.nn.kernels import (
    block_scales_interleave,
    grouped_matmul_block_scaled,
    grouped_quantize_dynamic_block_scaled,
)
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

from ...deepseekV4.layers.moe import (
    DeepseekV4Gate,
    DeepseekV4RoutedExperts,
    _PackedExpertWeight,
    sqrt_softplus,
)
from ..layer_resolver import LayerInfo
from ..model_config import (
    NVFP4_SF_ATOM_COLS,
    NVFP4_SF_ATOM_ROWS,
    DeepseekV41Config,
    nvfp4_sf_preinterleaved,
)

# One E4M3 scale per 16 K-elements (``quantization_config.group_size``).
NVFP4_BLOCK = 16

# Routed experts' down-proj input: the fused SwiGLU + NVFP4 quantize kernel
# (default) or the unfused graph-op SwiGLU then the grouped quantize ("0").
_FUSED_SILU_QUANT = os.environ.get("MAX_DSV41_FUSED_SILU_QUANT", "1") != "0"

# The fused kernel's grid needs an SM per expert group. The driver does not
# expose the SM count, so this is a B200's.
_NUM_SMS = 148


def _use_fused_silu_quant(n_local_experts: int) -> bool:
    """Whether the down-proj input uses the fused SwiGLU + quantize kernel.

    Falls back to the unfused chain when the local experts outnumber the SMs.
    """
    return _FUSED_SILU_QUANT and n_local_experts <= _NUM_SMS


# What `fused_silu_quantized` reads off its config: NVFP4 activations with a
# static per-expert input scale and dynamic per-16 E4M3 block scales.
_NVFP4_ACTIVATION_QUANT_CONFIG = QuantConfig(
    input_scale=InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, NVFP4_BLOCK),
    ),
    weight_scale=WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, NVFP4_BLOCK),
    ),
    mlp_quantized_layers=set(),
    attn_quantized_layers=set(),
    format=QuantFormat.NVFP4,
)

# The token id that marks image-span positions. Every image delimiter role
# shares this id (the role travels in a separate `types` tensor), so routing
# tests `==` rather than a range.
IMAGE_SENTINEL_BASE_ID = 129264


class _PackedExpertWeightNvFp4(_PackedExpertWeight):
    """One stacked NVFP4 expert projection, stacked by expert id.

    - ``weight``: ``[E, N, K/2]`` uint8, two E2M1 values per byte.
    - ``weight_scale``: ``[E, N, K/16]`` float8_e4m3fn.
    - ``weight_scale_2``: ``[E]`` float32, a dequant multiplier (not
      inverted).
    - ``input_scale``: ``[E]`` float32 static activation scale (inverted by
      ``DeepseekV41NvFp4RoutedExperts._activation_scales``).

    Skips ``_PackedExpertWeight.__init__``, whose ``K % 128`` check is
    specific to the W4A8 path; NVFP4 needs only ``K % 32``.
    """

    def __init__(
        self, n_experts: int, out_dim: int, in_dim: int, device: DeviceRef
    ) -> None:
        Module.__init__(self)
        if in_dim % 32:
            raise ValueError(f"nvfp4 expert K={in_dim} is not a multiple of 32")
        self.n_experts = n_experts
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.weight = Weight(
            name="weight",
            dtype=DType.uint8,
            shape=(n_experts, out_dim, in_dim // 2),
            device=device,
        )
        self.weight_scale = Weight(
            name="weight_scale",
            dtype=DType.float8_e4m3fn,
            shape=(n_experts, out_dim, in_dim // NVFP4_BLOCK),
            device=device,
        )
        self.weight_scale_2 = Weight(
            name="weight_scale_2",
            dtype=DType.float32,
            shape=(n_experts,),
            device=device,
        )
        self.input_scale = Weight(
            name="input_scale",
            dtype=DType.float32,
            shape=(n_experts,),
            device=device,
        )

    @property
    def sf_preinterleaved(self) -> bool:
        """Whether ``weight_scale`` was interleaved on the host before loading.

        For these shapes the scale bytes arrive already in the grouped
        matmul's interleaved layout; the buffer keeps its ``[E, N, K/16]``
        shape and only its byte order differs.
        """
        return nvfp4_sf_preinterleaved(self.out_dim, self.in_dim // NVFP4_BLOCK)

    def __call__(self, device: DeviceRef) -> TensorValue:
        """Returns the block scales in the grouped matmul's interleaved layout.

        The result is ``[E, N/128, K/64, 32, 4, 4]``.
        """
        scales = self.weight_scale.to(device)
        if self.sf_preinterleaved:
            # A free view. Interleaving in the graph would make the compiler
            # keep a second resident copy of every expert scale.
            return ops.reshape(
                scales,
                [
                    scales.shape[0],
                    self.out_dim // NVFP4_SF_ATOM_ROWS,
                    self.in_dim // NVFP4_BLOCK // NVFP4_SF_ATOM_COLS,
                    32,
                    4,
                    4,
                ],
            )
        per_expert = ops.split(scales, [1] * self.n_experts, axis=0)
        return ops.stack(
            [
                block_scales_interleave(
                    ops.reshape(e, [self.out_dim, self.in_dim // NVFP4_BLOCK]),
                    NVFP4_BLOCK,
                )
                for e in per_expert
            ],
            axis=0,
        )


class DeepseekV41NvFp4RoutedExperts(DeepseekV4RoutedExperts):
    """Trunk routed experts quantized NVFP4 (W4A4).

    The per-16 E4M3 block scales are computed from each call's own amax, so
    :meth:`_grouped` ignores the ``scale_slot`` argument the base pipeline
    passes.
    """

    # The routing weight multiplies GEMM2's output, not its input, so it does
    # not change how the down projection's activations are quantized.
    _weight_after_down_proj = True

    def __init__(
        self,
        n_experts: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts_per_tok: int,
        swiglu_limit: float,
        device: DeviceRef,
    ) -> None:
        Module.__init__(self)
        self.n_experts = n_experts
        self.topk = num_experts_per_tok
        self.hidden = hidden_size
        self.inter = moe_intermediate_size
        self.swiglu_limit = swiglu_limit
        self.gate_up_proj: _PackedExpertWeight = _PackedExpertWeightNvFp4(
            n_experts, 2 * moe_intermediate_size, hidden_size, device
        )
        self.down_proj: _PackedExpertWeight = _PackedExpertWeightNvFp4(
            n_experts, hidden_size, moe_intermediate_size, device
        )
        self.expert_offset = 0
        self.n_global_experts = n_experts

    def shard_experts(self, num_devices: int) -> None:
        """Splits the experts and their per-expert scales across devices."""
        super().shard_experts(num_devices)
        for proj in (self.gate_up_proj, self.down_proj):
            assert isinstance(proj, _PackedExpertWeightNvFp4)
            proj.weight_scale_2.sharding_strategy = ShardingStrategy.axiswise(
                0, num_devices
            )
            proj.input_scale.sharding_strategy = ShardingStrategy.axiswise(
                0, num_devices
            )

    @staticmethod
    def _activation_scales(
        proj: _PackedExpertWeightNvFp4, device: DeviceRef
    ) -> tuple[TensorValue, TensorValue]:
        """A projection's per-expert static activation scales.

        Each expert's rows are quantized against that expert's own
        ``input_scale``, not one max over all experts: a single shared scale
        would change which blocks saturate and clip.

        Returns:
            ``(sf_tensor, input_scale)``, both ``[n_experts]`` float32 for this
            replica's experts: ``sf_tensor = 1 / input_scale`` is the quantize
            kernel's tensor-wise scale, and ``input_scale`` feeds the matmul
            epilogue.
        """
        input_scale = proj.input_scale.to(device)
        return 1.0 / input_scale, input_scale

    def _grouped(
        self,
        proj: _PackedExpertWeight,
        x: TensorValue,
        a_offsets: TensorValue,
        scale_offsets: TensorValue,
        scale_slot: TensorValue,
        expert_ids: TensorValue,
    ) -> TensorValue:
        del scale_slot  # NVFP4's block scales need no reslice.
        assert isinstance(proj, _PackedExpertWeightNvFp4)
        device = x.device
        sf_tensor, input_scale = self._activation_scales(proj, device)
        # The down projection's input arrives float32; the quantize kernel
        # requires bf16.
        x = ops.cast(x, DType.bfloat16)
        quantized, a_scales = grouped_quantize_dynamic_block_scaled(
            x,
            row_offsets=a_offsets,
            scales_offsets=scale_offsets,
            expert_ids=expert_ids,
            sf_tensor=sf_tensor,
            sf_vector_size=NVFP4_BLOCK,
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,
        )
        return self._nvfp4_matmul(
            proj,
            quantized,
            a_scales,
            input_scale,
            a_offsets,
            scale_offsets,
            expert_ids,
        )

    def _nvfp4_matmul(
        self,
        proj: _PackedExpertWeightNvFp4,
        quantized: TensorValue,
        a_scales: TensorValue,
        input_scale: TensorValue,
        a_offsets: TensorValue,
        scale_offsets: TensorValue,
        expert_ids: TensorValue,
    ) -> TensorValue:
        """The grouped NVFP4 GEMM over rows quantized with ``input_scale``."""
        device = quantized.device
        # Descale = weight_scale_2 / (1 / input_scale): a per-expert product.
        expert_scales = proj.weight_scale_2.to(device) * input_scale
        usage = ops.concat(
            [
                ops.cast(
                    ops.shape_to_tensor([quantized.shape[0]]), DType.uint32
                ),
                ops.constant(
                    np.array([self.n_experts], np.uint32),
                    DType.uint32,
                    DeviceRef.CPU(),
                ),
            ],
            axis=0,
        )
        return grouped_matmul_block_scaled(
            quantized,
            proj.weight.to(device),
            a_scales,
            proj(device),
            a_offsets,
            scale_offsets,
            expert_ids,
            expert_scales,
            usage,
            out_type=DType.bfloat16,
        )

    def _activate_and_down(
        self,
        gate_up: TensorValue,
        slot_weight: TensorValue,
        a_offsets: TensorValue,
        scale_offsets: TensorValue,
        scale_slot: TensorValue,
        expert_ids: TensorValue,
    ) -> TensorValue:
        """The clamped SwiGLU and its NVFP4 quantize in one kernel, then GEMM2.

        Computes the base class's clamped SwiGLU (``min(gate, limit)``,
        ``clamp(up, -limit, limit)``) and quantizes each expert's rows against
        that expert's ``input_scale`` (group ``g`` is local expert ``g``).
        Runs the unfused chain instead when :func:`_use_fused_silu_quant` is
        false (``MAX_DSV41_FUSED_SILU_QUANT=0`` or more local experts than SMs).

        Not bitwise equal to the unfused chain: it skips the intermediate bf16
        round and uses an approximate reciprocal in ``silu``, so an
        occasional NVFP4 code or block scale differs by one step.
        """
        if not _use_fused_silu_quant(self.n_experts):
            return super()._activate_and_down(
                gate_up,
                slot_weight,
                a_offsets,
                scale_offsets,
                scale_slot,
                expert_ids,
            )
        assert self._weight_after_down_proj
        proj = self.down_proj
        assert isinstance(proj, _PackedExpertWeightNvFp4)
        input_scale = proj.input_scale.to(gate_up.device)
        quantized, a_scales = fused_silu_quantized(
            ops.cast(gate_up, DType.bfloat16),
            a_offsets,
            _NVFP4_ACTIVATION_QUANT_CONFIG,
            DType.uint8,
            input_scales=input_scale,
            scales_offsets=scale_offsets,
            activation=GatedActivation(
                "silu", limit=max(float(self.swiglu_limit), 0.0)
            ),
        )
        return self._nvfp4_matmul(
            proj,
            quantized,
            a_scales,
            input_scale,
            a_offsets,
            scale_offsets,
            expert_ids,
        )


class DeepseekV41Gate(DeepseekV4Gate):
    """Scored top-k routing gate with a separate bias for image tokens.

    Never hash-routed, so ``is_hash_routed`` is always false. Image-span
    tokens use ``bias_vl`` instead of ``bias`` for expert selection, and the
    renormalization sum is floored. A checkpoint without ``bias_vl`` loads it
    as zeros.
    """

    def __init__(
        self,
        n_routed_experts: int,
        num_experts_per_tok: int,
        routed_scaling_factor: float,
        hidden_size: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        Module.__init__(self)
        self.is_hash_routed = False
        self.topk = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.route_scale = routed_scaling_factor
        self.weight = Weight(
            name="weight",
            dtype=dtype,
            shape=(n_routed_experts, hidden_size),
            device=device,
        )
        self.bias = Weight(
            name="bias",
            dtype=DType.float32,
            shape=(n_routed_experts,),
            device=device,
        )
        self.bias_vl = Weight(
            name="bias_vl",
            dtype=DType.float32,
            shape=(n_routed_experts,),
            device=device,
        )

    def __call__(
        self, x: TensorValue, token_ids: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Routes each token to its top-k experts.

        The bias added for top-k selection is ``bias_vl`` where ``token_ids``
        is the image sentinel and ``bias`` elsewhere; the returned weights are
        the unbiased ``sqrtsoftplus`` scores, renormalized (an all-zero row
        divides by 1) and scaled by ``route_scale``.

        Args:
            x: ``[b, s, d]`` hidden states.
            token_ids: ``[b, s]`` token ids.

        Returns:
            ``(weights, indices)``: ``[b, s, k]`` float32 and ``[b, s, k]``
            int32.
        """
        scores = sqrt_softplus(
            ops.matmul(
                ops.cast(x, DType.float32),
                ops.transpose(ops.cast(self.weight, DType.float32), 0, 1),
            )
        )
        is_image = ops.unsqueeze(token_ids == IMAGE_SENTINEL_BASE_ID, -1)
        bias = ops.where(is_image, self.bias_vl, self.bias)
        _, indices = ops.top_k(scores + bias, self.topk, axis=-1)
        indices = ops.cast(indices, DType.int32)

        weights = ops.gather_nd(
            scores, ops.unsqueeze(indices, -1), batch_dims=2
        )
        weight_sum = ops.sum(weights, axis=-1)
        denom = ops.where(
            weight_sum > 0,
            weight_sum,
            ops.constant(1.0, DType.float32, weight_sum.device),
        )
        return weights * (self.route_scale / denom), indices


# One E8M0 scale per 32 elements (MXFP8).
_MXFP8_BLOCK_SIZE = 32


def _shared_expert_quant_config() -> QuantConfig:
    """The MXFP8 ``QuantConfig`` for the shared expert's projections.

    One dynamic per-32 activation scale and one static per-32 weight scale.
    """
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.DYNAMIC,
            dtype=DType.float32,
            block_size=(1, _MXFP8_BLOCK_SIZE),
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e8m0fnu,
            block_size=(1, _MXFP8_BLOCK_SIZE),
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.MXFP8,
    )


class DeepseekV41SharedExpert(Module):
    """The always-active dense FFN, added to the routed output unweighted.

    A SwiGLU with an asymmetric clamp. Weights are MXFP8 and replicated
    across devices. ``w1`` (gate) and ``w3`` (up)
    are fused on the host into one ``gate_up_proj`` (``[w1 rows; w3 rows]``),
    so one activation quantize feeds one GEMM; this is bitwise equal to
    running them separately.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        swiglu_limit: float,
        device: DeviceRef,
    ) -> None:
        """Declares ``gate_up_proj`` (``w1`` then ``w3``) and ``w2``.

        Args:
            hidden_size: Model hidden width.
            moe_intermediate_size: Expert intermediate width.
            swiglu_limit: SwiGLU clamp limit; ``<= 0`` disables the clamp.
            device: Device the weights live on.
        """
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.moe_intermediate_size = moe_intermediate_size

        def linear(in_dim: int, out_dim: int) -> Linear:
            return Linear(
                in_dim=in_dim,
                out_dim=out_dim,
                dtype=DType.float8_e4m3fn,
                device=device,
                quant_config=_shared_expert_quant_config(),
            )

        self.gate_up_proj = linear(hidden_size, 2 * moe_intermediate_size)
        self.w2 = linear(moe_intermediate_size, hidden_size)

    def __call__(self, x: TensorValue) -> TensorValue:
        """``[..., hidden]`` -> ``[..., hidden]``, no routing weight applied."""
        inter = self.moe_intermediate_size
        gate_up = self.gate_up_proj(x)
        # Slices, not `ops.split`: `ops.split` slows the fused GEMM ~25% at
        # prefill sizes.
        gate = ops.cast(gate_up[..., :inter], DType.float32)
        up = ops.cast(gate_up[..., inter:], DType.float32)
        if self.swiglu_limit > 0:
            limit = float(self.swiglu_limit)
            # Asymmetric: `up` is clamped on both sides, `gate` only above.
            up = ops.min(ops.max(up, -limit), limit)
            gate = ops.min(gate, limit)
        h = ops.silu(gate) * up
        return self.w2(ops.cast(h, x.dtype))


class ExpertParallelFFN(Protocol):
    """One device's replica of an expert-parallel FFN."""

    @property
    def shared_experts(self) -> Callable[[TensorValue], TensorValue]:
        """The replicated shared expert."""
        ...

    def routed(self, x: TensorValue, token_ids: TensorValue) -> TensorValue:
        """``[b, s, hidden]`` -> float32 ``[b, s, hidden]``."""
        ...


class DeepseekV41MoE(Module):
    """The trunk FFN: gate, NVFP4 routed experts and MXFP8 shared expert.

    Under expert parallelism the shared expert is added after the routed
    all-reduce (:meth:`tensor_parallel`), so it is counted once.
    """

    def __init__(
        self, config: DeepseekV41Config, layer_idx: int, device: DeviceRef
    ) -> None:
        super().__init__()
        del layer_idx  # every trunk layer is scored-routed, no hash routing
        # The gate weight is unquantized bf16; `config.dtype` is the packed
        # expert dtype.
        self.gate = DeepseekV41Gate(
            n_routed_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            routed_scaling_factor=config.routed_scaling_factor,
            hidden_size=config.hidden_size,
            dtype=DType.bfloat16,
            device=device,
        )
        self.experts = DeepseekV41NvFp4RoutedExperts(
            n_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            swiglu_limit=config.swiglu_limit,
            device=device,
        )
        if config.n_shared_experts != 1:
            raise ValueError(
                "DeepSeek-V4.1 assumes exactly one shared expert, got "
                f"{config.n_shared_experts}"
            )
        self.shared_experts = DeepseekV41SharedExpert(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            swiglu_limit=config.swiglu_limit,
            device=device,
        )

    def shard_experts(self, num_devices: int) -> None:
        """Declares the expert-parallel split of the routed experts.

        Each device owns a contiguous run of whole experts; the gate and the
        shared expert are replicated.

        Raises:
            ValueError: If the experts do not divide across ``num_devices``.
        """
        n_experts = self.experts.n_global_experts
        if n_experts % num_devices:
            raise ValueError(
                f"{n_experts} routed experts do not split across "
                f"{num_devices} devices"
            )
        self.experts.shard_experts(num_devices)

    def keep_local_experts(self, num_devices: int, rank: int) -> None:
        """Makes this replica compute only its run of :meth:`shard_experts`."""
        self.experts.keep_local_experts(num_devices, rank)

    def routed(self, x: TensorValue, token_ids: TensorValue) -> TensorValue:
        """Returns this replica's float32 routed-expert output.

        Maps ``[b, s, hidden]`` to ``[b, s, hidden]``, before any all-reduce.
        """
        weights, indices = self.gate(x, token_ids)
        hidden = int(x.shape[-1])
        batch, seq = x.shape[0], x.shape[1]
        routed = self.experts(
            ops.reshape(x, [batch * seq, hidden]),
            ops.reshape(weights, [batch * seq, self.experts.topk]),
            ops.reshape(indices, [batch * seq, self.experts.topk]),
        )
        return ops.reshape(
            ops.cast(routed, DType.float32), [batch, seq, hidden]
        )

    @staticmethod
    def tensor_parallel(
        moes: Sequence[ExpertParallelFFN],
        xs: Sequence[TensorValue],
        token_ids: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """Runs the MoE expert-parallel, one replica per device in ``moes``.

        Each device's routed share is all-reduced in float32, then each device
        adds its replicated shared expert.

        Returns:
            One ``[b, s, hidden]`` output per device, in ``x``'s dtype.
        """
        partials = [
            moe.routed(x, tok)
            for moe, x, tok in zip(moes, xs, token_ids, strict=True)
        ]
        reduced = ops.allreduce.sum(partials, signal_buffers)
        return [
            ops.cast(
                y + ops.cast(moe.shared_experts(x), DType.float32), x.dtype
            )
            for moe, x, y in zip(moes, xs, reduced, strict=True)
        ]

    def __call__(self, x: TensorValue, token_ids: TensorValue) -> TensorValue:
        """``[1, s, hidden]``, ``[1, s]`` token ids -> ``[1, s, hidden]``."""
        y = ops.cast(self.shared_experts(x), DType.float32) + self.routed(
            x, token_ids
        )
        return ops.cast(y, x.dtype)


def nvfp4_moe_ffn_factory(
    config: DeepseekV41Config, layer_info: LayerInfo, device: DeviceRef
) -> Module:
    """Builds the NVFP4-routed MoE for one trunk layer."""
    return DeepseekV41MoE(config, layer_info.layer_id, device)
