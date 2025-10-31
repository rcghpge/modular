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
"""An opaque KV Cache optimized attention mechanism with Rope."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.support.math import ceildiv

from ..float8_config import Float8Config
from ..float8_ops import matmul_float8
from ..kernels import (
    flare_mla_prefill_plan,
    flare_mla_prefill_ragged,
    fused_qk_ragged_rope,
    k_cache_to_buffer,
    kv_cache_get_max_seq_len,
    matmul_k_cache_ragged_scaled_float8,
    quantize_dynamic_scaled_float8,
    rms_norm_key_cache,
)
from ..kv_cache import KVCacheParams, PagedCacheValues
from ..layer import Module, Shardable
from ..linear import Linear
from ..norm import RMSNorm
from ..rotary_embedding import RotaryEmbedding
from .mask_config import MHAMaskVariant


class LatentAttentionWithRopeFp8(Module, Shardable):
    """Implementation of Latent Attention with Rope with FP8 weights."""

    rope: RotaryEmbedding

    _sharding_strategy: ShardingStrategy | None = None
    """The sharding strategy for the module."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        float8_config: Float8Config,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        buffer_size: int = 16384,
        graph_mode: str | None = None,
    ) -> None:
        """Initializes the latent attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the
                head dim, and data type.
            dtype: DType of the weights, currently only bfloat16 is supported.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            q_lora_rank: Optional LoRA rank for Q projection.
            kv_lora_rank: LoRA rank for KV projections.
            qk_nope_head_dim: Head dimension for non-positional encoding part.
            qk_rope_head_dim: Head dimension for rope part.
            v_head_dim: Head dimension for value.
            buffer_size: Buffer size for storing the temporal results during
                prefill, in unit of tokens.
            graph_mode: Pipeline role to use for the attention layer. Should be
                "prefill", "decode", or "auto".
        """
        super().__init__()

        _role = graph_mode or "auto"
        if _role not in ("prefill", "decode", "auto"):
            raise ValueError(
                f"Invalid graph_mode '{_role}'. Use 'prefill', 'decode', or 'auto'."
            )
        if (
            not float8_config.weight_scale.is_block
            or not float8_config.input_scale.is_block
        ):
            raise ValueError(
                "Weight scale and input scale must be block-wise for LatentAttentionWithRopeFp8"
            )

        # TODO: Support decode path for FP8.
        if _role != "prefill":
            raise ValueError("MLA decode is not yet supported for FP8")

        self.graph_mode = _role
        self.float8_config = float8_config

        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.linear_cls = linear_cls

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.cache_head_dim = kv_lora_rank + qk_rope_head_dim

        self.BUFFER_TOK_SIZE = buffer_size

        self._scale = scale if scale else math.sqrt(1.0 / self.qk_head_dim)
        self.scale = self.rope.compute_scale(self._scale)
        self.devices = devices or [DeviceRef.CPU()]
        assert float8_config.weight_scale.block_size is not None
        assert float8_config.input_scale.block_size is not None
        if (
            float8_config.input_scale.block_size[1]
            != float8_config.weight_scale.block_size[1]
        ):
            raise ValueError(
                "Input scale and weight scale must have the same K block size"
            )
        self.scales_granularity_mnk = (
            float8_config.input_scale.block_size[0],
            float8_config.weight_scale.block_size[0],
            float8_config.weight_scale.block_size[1],
        )

        if self.q_lora_rank is not None:
            self.q_a_proj = Weight(
                name="q_a_proj.weight",
                dtype=DType.float8_e4m3fn,
                shape=(self.q_lora_rank, self.hidden_size),
                device=self.devices[0],
            )
            self.q_a_proj_scale = Weight(
                name="q_a_proj.weight_scale",
                dtype=float8_config.weight_scale.dtype,
                shape=(
                    ceildiv(
                        int(self.q_a_proj.shape[0]),
                        float8_config.weight_scale.block_size[0],
                    ),
                    ceildiv(
                        int(self.q_a_proj.shape[1]),
                        float8_config.weight_scale.block_size[1],
                    ),
                ),
                device=self.devices[0],
            )

            self.q_a_layernorm = RMSNorm(
                dim=self.q_lora_rank,
                dtype=DType.bfloat16,
                eps=1e-6,
                multiply_before_cast=False,
            )

            self.q_b_proj = Weight(
                name="q_b_proj.weight",
                dtype=DType.float8_e4m3fn,
                shape=(self.n_heads * self.qk_head_dim, self.q_lora_rank),
                device=self.devices[0],
            )
            self.q_b_proj_scale = Weight(
                name="q_b_proj.weight_scale",
                dtype=float8_config.weight_scale.dtype,
                shape=(
                    ceildiv(
                        int(self.q_b_proj.shape[0]),
                        float8_config.weight_scale.block_size[0],
                    ),
                    ceildiv(
                        int(self.q_b_proj.shape[1]),
                        float8_config.weight_scale.block_size[1],
                    ),
                ),
                device=self.devices[0],
            )

        else:
            self.q_proj = Weight(
                name="q_proj.weight",
                dtype=DType.float8_e4m3fn,
                shape=(self.n_heads * self.qk_head_dim, self.hidden_size),
                device=self.devices[0],
            )
            self.q_proj_scale = Weight(
                name="q_proj.weight_scale",
                dtype=float8_config.weight_scale.dtype,
                shape=(
                    ceildiv(
                        int(self.q_proj.shape[0]),
                        float8_config.weight_scale.block_size[0],
                    ),
                    ceildiv(
                        int(self.q_proj.shape[1]),
                        float8_config.weight_scale.block_size[1],
                    ),
                ),
                device=self.devices[0],
            )

        self.kv_a_proj_layernorm = Weight(
            name="kv_a_layernorm.weight",
            dtype=DType.bfloat16,
            shape=(self.kv_lora_rank,),
            device=self.devices[0],
        )

        self.kv_a_proj_with_mqa = Weight(
            name="kv_a_proj_with_mqa.weight",
            dtype=DType.float8_e4m3fn,
            shape=(self.cache_head_dim, self.hidden_size),
            device=self.devices[0],
        )
        self.kv_a_proj_with_mqa_scale = Weight(
            name="kv_a_proj_with_mqa.weight_scale",
            dtype=float8_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.kv_a_proj_with_mqa.shape[0]),
                    float8_config.weight_scale.block_size[0],
                ),
                ceildiv(
                    int(self.kv_a_proj_with_mqa.shape[1]),
                    float8_config.weight_scale.block_size[1],
                ),
            ),
            device=self.devices[0],
        )

        self.kv_b_proj = Weight(
            name="kv_b_proj.weight",
            dtype=DType.float8_e4m3fn,
            shape=(
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                self.kv_lora_rank,
            ),
            device=self.devices[0],
        )
        self.kv_b_proj_scale = Weight(
            name="kv_b_proj.weight_scale",
            dtype=float8_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.kv_b_proj.shape[0]),
                    float8_config.weight_scale.block_size[0],
                ),
                ceildiv(
                    int(self.kv_b_proj.shape[1]),
                    float8_config.weight_scale.block_size[1],
                ),
            ),
            device=self.devices[0],
        )

        self.o_proj = linear_cls(
            in_dim=self.n_heads * self.v_head_dim,
            out_dim=self.hidden_size,
            dtype=DType.float8_e4m3fn,
            device=self.devices[0],
            float8_config=float8_config,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the Module sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the Module sharding strategy.

        Args:
            strategy: The strategy describing the Module sharding.
        """
        if strategy.is_replicate:
            # Data parallelism: replicate the entire module's weights to each device.
            self._sharding_strategy = strategy

            weights = [
                self.kv_a_proj_layernorm,
                self.kv_a_proj_with_mqa,
                self.kv_a_proj_with_mqa_scale,
                self.kv_b_proj,
                self.kv_b_proj_scale,
                self.o_proj.weight,
            ]
            if self.q_lora_rank is not None:
                weights.append(self.q_a_proj)
                weights.append(self.q_a_proj_scale)
                weights.append(self.q_a_layernorm.weight)
                weights.append(self.q_b_proj)
                weights.append(self.q_b_proj_scale)
            else:
                weights.append(self.q_proj)
                weights.append(self.q_proj_scale)

            if self.o_proj.input_scale is not None:
                weights.append(self.o_proj.input_scale)
            if self.o_proj.weight_scale is not None:
                weights.append(self.o_proj.weight_scale)

            for weight in weights:
                weight.sharding_strategy = ShardingStrategy.replicate(
                    strategy.num_devices
                )
        else:
            raise ValueError(
                "Only tensor parallel or replicate sharding strategies are supported for LatentAttentionWithRope"
            )

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[LatentAttentionWithRopeFp8]:
        """Creates sharded views of this Module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded LatentAttentionWithRope instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "LatentAttentionWithRope layer cannot be sharded because no sharding strategy was provided."
            )

        if self.sharding_strategy.is_replicate:
            # Replicate full weights to each device (no head split).
            if self.q_lora_rank is not None:
                q_a_proj_shards = self.q_a_proj.shard(devices)
                q_a_proj_scale_shards = self.q_a_proj_scale.shard(devices)
                q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(
                    devices
                )
                q_b_proj_shards = self.q_b_proj.shard(devices)
                q_b_proj_scale_shards = self.q_b_proj_scale.shard(devices)
            else:
                q_proj_shards = self.q_proj.shard(devices)
                q_proj_scale_shards = self.q_proj_scale.shard(devices)

            kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
            kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
            kv_a_proj_with_mqa_scale_shards = (
                self.kv_a_proj_with_mqa_scale.shard(devices)
            )
            kv_b_proj_shards = self.kv_b_proj.shard(devices)
            kv_b_proj_scale_shards = self.kv_b_proj_scale.shard(devices)
            o_proj_weight_shards = self.o_proj.weight.shard(devices)

            if self.o_proj.input_scale is not None:
                o_proj_scale_shards = self.o_proj.input_scale.shard(devices)
            if self.o_proj.weight_scale is not None:
                o_proj_weight_scale_shards = self.o_proj.weight_scale.shard(
                    devices
                )

            replicas: list[LatentAttentionWithRopeFp8] = []
            for shard_idx, device in enumerate(devices):
                replica = LatentAttentionWithRopeFp8(
                    rope=self.rope,
                    num_attention_heads=self.n_heads,  # DP keeps full heads
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    float8_config=self.float8_config,
                    devices=[device],
                    graph_mode=self.graph_mode,
                    linear_cls=self.linear_cls,
                    scale=self._scale,
                    q_lora_rank=self.q_lora_rank,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    buffer_size=self.BUFFER_TOK_SIZE,
                )

                if self.q_lora_rank is not None:
                    replica.q_a_proj = q_a_proj_shards[shard_idx]
                    replica.q_a_proj_scale = q_a_proj_scale_shards[shard_idx]
                    replica.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                        shard_idx
                    ]
                    replica.q_b_proj = q_b_proj_shards[shard_idx]
                    replica.q_b_proj_scale = q_b_proj_scale_shards[shard_idx]
                else:
                    replica.q_proj = q_proj_shards[shard_idx]
                    replica.q_proj_scale = q_proj_scale_shards[shard_idx]

                replica.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[
                    shard_idx
                ]
                replica.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[
                    shard_idx
                ]
                replica.kv_a_proj_with_mqa_scale = (
                    kv_a_proj_with_mqa_scale_shards[shard_idx]
                )
                replica.kv_b_proj = kv_b_proj_shards[shard_idx]
                replica.kv_b_proj_scale = kv_b_proj_scale_shards[shard_idx]
                replica.o_proj.weight = o_proj_weight_shards[shard_idx]
                if self.o_proj.input_scale is not None:
                    replica.o_proj.input_scale = o_proj_scale_shards[shard_idx]
                if self.o_proj.weight_scale is not None:
                    replica.o_proj.weight_scale = o_proj_weight_scale_shards[
                        shard_idx
                    ]

                replicas.append(replica)

            return replicas
        else:
            raise ValueError(
                "Only tensor parallel or replicate sharding strategies are supported for LatentAttentionWithRope"
            )

    @property
    def w_uk_uv(self) -> list[TensorValue]:
        """The concatenation of q, k, and v weight vectors."""
        kv_b_proj_weight: TensorValue = self.kv_b_proj.transpose(0, 1)

        kv_b_proj_weight = kv_b_proj_weight.reshape(
            (
                self.kv_lora_rank,
                self.n_heads,
                (self.qk_nope_head_dim + self.v_head_dim),
            )
        )

        w_uk, w_uv = ops.split(
            kv_b_proj_weight, [self.qk_nope_head_dim, self.v_head_dim], axis=2
        )

        w_uv = w_uv.transpose(0, 1)

        w_uk_t = w_uk.permute([1, 2, 0])

        return [w_uk_t, w_uv]

    def _mla_impl(
        self,
        xq_nope: TensorValue,
        xq_rope: TensorValue,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # These weights are going to be used in the decode path.
        # Move the creation of these weights outside of the decode subgraph so the
        # ConstantFold Pass can optimize them out, avoiding the need to re-calculate
        # them each time we enter the decode subgraph.
        # w_uk, w_uv = self.w_uk_uv

        def _mla_prefill() -> TensorValue:
            xq = ops.concat([xq_nope, xq_rope], axis=2)

            (buffer_row_offsets, cache_offsets, buffer_lengths) = (
                flare_mla_prefill_plan(
                    self.kv_params,
                    input_row_offsets,
                    kv_collection,
                    layer_idx,
                    self.BUFFER_TOK_SIZE,
                )
            )
            buffer_lengths_host = buffer_lengths.to(DeviceRef.CPU())

            k_latent_buffer = k_cache_to_buffer(
                self.kv_params,
                buffer_row_offsets[0],
                cache_offsets[0],
                kv_collection,
                layer_idx,
                buffer_lengths_host[0],
                self.BUFFER_TOK_SIZE,
                int(self.kv_b_proj.shape[1]),
            )
            kv_buffer = matmul_float8(
                x=k_latent_buffer,
                weight=self.kv_b_proj,
                weight_scale=self.kv_b_proj_scale,
                input_scale=None,  # Dynamic scaling
                float8_config=self.float8_config,
                group_size_or_per_token=self.scales_granularity_mnk[2],
            )

            kv_buffer = kv_buffer.reshape(
                (-1, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            )
            k_nope, v = ops.split(
                kv_buffer, [self.qk_nope_head_dim, self.v_head_dim], axis=2
            )

            result, _ = flare_mla_prefill_ragged(
                self.kv_params,
                xq,
                k_nope,
                v,
                input_row_offsets,
                buffer_row_offsets[0],
                cache_offsets[0],
                kv_collection,
                layer_idx,
                MHAMaskVariant.CAUSAL_MASK,
                self.scale,
                self.qk_rope_head_dim,
            )

            return result

        # def _mla_decode() -> TensorValue:
        #     # from [B, H, D] to [H, B, D]
        #     xq_nope_t = xq_nope.transpose(0, 1)

        #     # batched matmul
        #     xq_nope_proj = xq_nope_t @ w_uk
        #     xq_nope_proj = xq_nope_proj.transpose(0, 1)
        #     xq = ops.concat([xq_nope_proj, xq_rope], axis=2)

        #     # Calculate Flash Attention.
        #     attn_out = flare_mla_decode_ragged(
        #         self.kv_params,
        #         input=xq,
        #         kv_collection=kv_collection,
        #         layer_idx=layer_idx,
        #         input_row_offsets=input_row_offsets,
        #         mask_variant=MHAMaskVariant.CAUSAL_MASK,
        #         scale=self.scale,
        #     )

        #     # from [B, H, D] to [H, B, D]
        #     attn_out_latent = attn_out.transpose(0, 1)

        #     # batched matmul
        #     attn_out = attn_out_latent @ w_uv
        #     return attn_out.transpose(0, 1)

        # TODO: use max_lengths[0, 0] cause a CUDA_INVALID_MEMORY_ACCESS error,
        # as the graph compiler assumes it is a GPU tensor, and inserts a DtoH copy.
        max_seq_len = kv_cache_get_max_seq_len(self.kv_params, kv_collection)

        if self.graph_mode == "prefill":
            result = _mla_prefill()
        elif self.graph_mode == "decode":
            raise ValueError("MLA decode is not yet supported for FP8")
            # result = _mla_decode()
        else:
            raise ValueError("MLA auto mode is not yet supported for FP8")
            # result = ops.cond(
            #     max_seq_len > 1,
            #     [
            #         TensorType(
            #             dtype=xq_nope.dtype,
            #             shape=[
            #                 xq_nope.shape[0],
            #                 self.n_heads,
            #                 self.v_head_dim,
            #             ],
            #             device=xq_nope.device,
            #         )
            #     ],
            #     _mla_prefill,
            #     _mla_decode,
            # )[0].tensor

        result = ops.reshape(
            result, shape=[result.shape[0], self.n_heads * self.v_head_dim]
        )

        return result

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        if self.q_lora_rank is not None:
            # First FP8 matmul: x @ q_a_proj.T
            q_a_out = matmul_float8(
                x=x,
                weight=self.q_a_proj,
                weight_scale=self.q_a_proj_scale,
                input_scale=None,  # Dynamic scaling
                float8_config=self.float8_config,
                group_size_or_per_token=self.scales_granularity_mnk[2],
            )
            # Apply layer norm
            q_a_normed = self.q_a_layernorm(q_a_out)

            # Second FP8 matmul: q_a_normed @ q_b_proj.T
            xq = matmul_float8(
                x=q_a_normed,
                weight=self.q_b_proj,
                weight_scale=self.q_b_proj_scale,
                input_scale=None,  # Dynamic scaling
                float8_config=self.float8_config,
                group_size_or_per_token=self.scales_granularity_mnk[2],
            )
        else:
            # Single FP8 matmul: x @ q_proj.T
            xq = matmul_float8(
                x=x,
                weight=self.q_proj,
                weight_scale=self.q_proj_scale,
                input_scale=None,  # Dynamic scaling
                float8_config=self.float8_config,
                group_size_or_per_token=self.scales_granularity_mnk[2],
            )
        x, x_scales = quantize_dynamic_scaled_float8(
            x,
            self.float8_config.input_scale,
            self.float8_config.weight_scale,
            scales_type=self.kv_a_proj_with_mqa_scale.dtype,
            group_size_or_per_token=self.scales_granularity_mnk[2],
            out_type=self.kv_a_proj_with_mqa.dtype,
        )

        matmul_k_cache_ragged_scaled_float8(
            self.kv_params,
            hidden_states=x,
            input_row_offsets=input_row_offsets,
            weight=self.kv_a_proj_with_mqa,
            input_scale=x_scales,
            weight_scale=self.kv_a_proj_with_mqa_scale,
            kv_collection=kv_collection,
            scales_granularity_mnk=self.scales_granularity_mnk,
            layer_idx=layer_idx,
        )

        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.kv_a_proj_layernorm,
            epsilon=1e-6,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            rms_norm_cols=self.kv_lora_rank,
            weight_offset=0.0,
            multiply_before_cast=False,
        )

        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))

        xq_nope, xq_rope = ops.split(
            xq, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=2
        )

        # Apply rope.
        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)

        xq_rope = fused_qk_ragged_rope(
            self.kv_params,
            xq_rope,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=True,
        )

        attn_out = self._mla_impl(
            xq_nope, xq_rope, kv_collection, layer_idx, input_row_offsets
        )

        return self.o_proj(attn_out)


class DataParallelLatentAttentionWithRopeFp8(LatentAttentionWithRopeFp8):
    """Data-parallel implementation of Latent Attention with RoPE.

    This replicates the attention module across devices and runs each replica on
    its local inputs (x, kv, freqs_cis, input_row_offsets). No collective ops
    are required; KV-cache remains local to each device.

    Notes:
      - `signal_buffers` is accepted for interface parity with the distributed
        implementation but is not used here.
      - Assumes the caller has already distributed `xs`, `kv_collections`,
        `freqs_cis`, and `input_row_offsets` so that index i corresponds to
        device i, with `input_row_offsets[i]` rebased to start at 0.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)
        self.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, freqs_cis, and input_row_offsets must all have "
                f"length equal to number of devices ({n})"
            )

        outs: list[TensorValue] = []
        for i in range(n):
            if xs[i].shape[0] == 0:
                outs.append(xs[i])
                continue

            outs.append(
                self.list_of_attentions[i](
                    layer_idx,
                    xs[i],
                    kv_collections[i],
                    freqs_cis=freqs_cis[i],
                    input_row_offsets=input_row_offsets[i],
                )
            )
        return outs
