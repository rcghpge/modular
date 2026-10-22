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
"""The MiMo-V2.6-Flash text model on ModuleV3."""

from __future__ import annotations

import enum
from collections.abc import Sequence

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import (
    Embedding,
    Module,
    ModuleList,
    RMSNorm,
    as_subgraph,
)
from max.experimental.nn.common_layers.linear import ColumnParallelLinear
from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
)
from max.experimental.nn.norm import rms_norm
from max.experimental.tensor import Tensor
from max.nn.comm.ep import EPBatchManager, EPCommBuffers, EPConfig
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import ModuleV3Outputs

from .layers.attention import MiMoV2Attention
from .layers.moe import (
    MiMoV2ExpertParallelMoE,
    MiMoV2MLP,
    MiMoV2MoE,
)
from .model_config import FULL, SLIDING, MiMoV2Config


class ParallelismMode(enum.Enum):
    """How one decoder block shards its attention and its MoE.

    Each mode picks the MoE class and the residual collectives of a block;
    attention stays tensor-parallel in every multi-device mode until the
    DP-attn work lands.
    """

    TP_TP = "tp_tp"
    """TP attention + channel-sharded experts, one allreduce per residual
    add. The single-device and default multi-device mode."""

    TP_EP = "tp_ep"
    """TP attention + whole experts per device on the allreduce EP
    backend. The residual collectives are TP_TP's: the combine emits each
    device's partial top-k sum and the residual add's allreduce completes
    it."""

    DP_EP = "dp_ep"
    """DP attention + EP MoE, no residual collectives. Not built yet; the
    mode exists so the selection below names it."""


def select_parallelism_mode(
    num_devices: int,
    data_parallel_degree: int,
    ep_config: EPConfig | None,
) -> ParallelismMode:
    """Select the block-level parallelism from the device/DP/EP settings.

    Shared by the model and its decoder blocks so the two views of the mode
    cannot drift:

    - single device, or no EP        -> ``TP_TP``
    - ``data_parallel_degree == 1`` and EP -> ``TP_EP``
    - ``data_parallel_degree > 1``   -> ``DP_EP``
    """
    if num_devices <= 1 or (ep_config is None and data_parallel_degree <= 1):
        return ParallelismMode.TP_TP
    if data_parallel_degree > 1:
        return ParallelismMode.DP_EP
    assert ep_config is not None
    return ParallelismMode.TP_EP


class MiMoV2RMSNorm(RMSNorm):
    """RMSNorm that scales in float32 before the cast, as the reference does.

    The reference casts to BF16 before it scales; scaling in float32 and
    casting once is the order of the kernel that fuses a norm with the
    allreduce and residual add in front of it.
    """

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps, multiply_before_cast=True)


class MiMoV2DecoderLayer(Module[..., Tensor]):
    """Pre-norm attention and MLP (dense or MoE), tensor parallel."""

    def __init__(
        self,
        attention: MiMoV2Attention,
        mlp: MiMoV2MLP | MiMoV2MoE,
        input_layernorm: MiMoV2RMSNorm,
        post_attention_layernorm: MiMoV2RMSNorm,
    ) -> None:
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm

    def forward(
        self,
        h: Tensor,
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        comm: EPCommBuffers | None = None,
    ) -> Tensor:
        """Runs the layer.

        Args:
            h: The replicated residual stream, ``[tokens, hidden]``.
            kv_collections: The layer's KV group, one per device.
            freqs_cis: The group's RoPE table, replicated.
            input_row_offsets: Ragged sequence boundaries, replicated.
            comm: The EP communication buffers, when the MoE runs
                expert-parallel.

        Returns:
            The layer's output, replicated.
        """
        h = h + self.self_attn(
            self.input_layernorm(h),
            kv_collections,
            freqs_cis,
            input_row_offsets,
        ).to(h.mapping)
        mlp_out: Tensor
        if isinstance(self.mlp, MiMoV2ExpertParallelMoE):
            assert comm is not None
            # The shared compiled body resolves each layer's weights and
            # its EP buffers per call.
            mlp_out = as_subgraph(self.mlp, name="mimo_v2_moe")(
                self.post_attention_layernorm(h), comm
            )
        elif isinstance(self.mlp, MiMoV2MoE):
            # Every MoE layer shares one compiled body; each call resolves
            # its own layer's weights. The one dense layer runs inline.
            mlp_out = as_subgraph(self.mlp, name="mimo_v2_moe")(
                self.post_attention_layernorm(h)
            )
        else:
            mlp_out = self.mlp(self.post_attention_layernorm(h))
        return h + mlp_out.to(h.mapping)


class MiMoV2(Module[..., ModuleV3Outputs]):
    """MiMo-V2.6-Flash: 48 hybrid SWA/full decoder layers and an untied head.

    Weight names are the checkpoint's without ``model.``, as the weight
    adapter emits them. Attention runs inline in the root graph: each layer
    bakes its KV-group index in as a constant, so attention layers cannot
    share a compiled subgraph. The MoE layers share one.

    Under expert parallelism the graph's inputs grow an EP tail after the
    KV leaves: the per-device atomic counters and the buffer-pointer
    tensors the dispatch/combine kernels read. Every MoE call threads them
    through its shared subgraph.

    Hidden states are not returned: the ModuleV3 outputs hold one copy, and
    the graph-API model's per-device taps are a speculative-decode concern.
    """

    def __init__(
        self,
        config: MiMoV2Config,
        ep_batch_manager: EPBatchManager | None = None,
    ) -> None:
        if config.return_hidden_states != ReturnHiddenStates.NONE:
            raise NotImplementedError(
                "MiMo-V2 on ModuleV3 returns no hidden states; the "
                "speculative lane owns them"
            )
        self.mode = select_parallelism_mode(
            len(config.devices),
            config.data_parallel_degree,
            config.ep_config,
        )
        if self.mode is ParallelismMode.DP_EP:
            raise NotImplementedError(
                "Data-parallel attention for MiMo-V2 on ModuleV3 is not "
                "built yet."
            )
        if self.mode is ParallelismMode.TP_EP and ep_batch_manager is None:
            raise ValueError(
                "EP parallelism is selected but no EP batch manager was "
                f"given: mode={self.mode}, ep_config={config.ep_config}"
            )
        if ep_batch_manager is not None and (
            self.mode is not ParallelismMode.TP_EP
        ):
            raise ValueError(
                "An EP batch manager arrives without an EP parallelism "
                f"mode: mode={self.mode}, ep_config={config.ep_config}"
            )
        self.ep_batch_manager = ep_batch_manager
        self.devices = config.devices
        # Annotated, not inferred: the NONE guard above narrows the right
        # side to the literal, and an inferred-literal attribute would make
        # the spec target's SELECTED_LAYERS widening an error.
        self.return_logits: ReturnLogits = config.return_logits
        self.return_hidden_states: ReturnHiddenStates = (
            config.return_hidden_states
        )

        assert isinstance(config.kv_params, MultiKVCacheParams)
        self.kv_params = config.kv_params
        kv_params: dict[str, KVCacheParams] = {}
        for group, params in config.kv_params.children.items():
            assert isinstance(params, KVCacheParams)
            kv_params[group] = params
        self.ropes = {
            group: RotaryEmbedding(
                dim=config.hidden_size,
                n_heads=config.num_attention_heads,
                theta=theta,
                max_seq_len=config.max_seq_len,
                device=config.devices[0].to_device(),
                head_dim=config.rotary_dim,
                interleaved=False,
            )
            for group, theta in config.rope_thetas.items()
        }

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)
        self.norm = MiMoV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False
        )

        group_layer_counts = {SLIDING: 0, FULL: 0}
        layers = []
        for i, group in enumerate(config.layer_types):
            attention = MiMoV2Attention(
                layout=config.qkv_layouts[group],
                head_dim=config.head_dim,
                v_head_dim=config.v_head_dim,
                hidden_size=config.hidden_size,
                kv_params=kv_params[group],
                layer_idx=group_layer_counts[group],
                scale=config.attention_scale,
                value_scale=config.attention_value_scale,
                sliding_window=(
                    config.sliding_window if group == SLIDING else None
                ),
                has_sinks=config.sinks[group],
                quant_config=config.quant.dense,
            )
            group_layer_counts[group] += 1
            mlp: MiMoV2MLP | MiMoV2MoE
            if i in config.moe_layers:
                if ep_batch_manager is not None:
                    # Whole experts per device; the dispatch/combine move
                    # the tokens (see MiMoV2ExpertParallelMoE).
                    mlp = MiMoV2ExpertParallelMoE(
                        hidden_dim=config.hidden_size,
                        num_experts=config.n_routed_experts,
                        num_experts_per_tok=config.num_experts_per_tok,
                        moe_dim=config.moe_intermediate_size,
                        norm_topk_prob=config.norm_topk_prob,
                        quant_config=config.quant.experts,
                        ep_batch_manager=ep_batch_manager,
                    )
                else:
                    mlp = MiMoV2MoE(
                        hidden_dim=config.hidden_size,
                        num_experts=config.n_routed_experts,
                        num_experts_per_tok=config.num_experts_per_tok,
                        moe_dim=config.moe_intermediate_size,
                        norm_topk_prob=config.norm_topk_prob,
                        quant_config=config.quant.experts,
                    )
            else:
                mlp = MiMoV2MLP(
                    hidden_dim=config.hidden_size,
                    intermediate_dim=config.intermediate_size,
                )
            layers.append(
                MiMoV2DecoderLayer(
                    attention=attention,
                    mlp=mlp,
                    input_layernorm=MiMoV2RMSNorm(
                        config.hidden_size, eps=config.rms_norm_eps
                    ),
                    post_attention_layernorm=MiMoV2RMSNorm(
                        config.hidden_size, eps=config.rms_norm_eps
                    ),
                )
            )
        self.layers = ModuleList(layers)
        self._layer_groups = list(config.layer_types)

    def _logits(self, h: Tensor) -> Tensor:
        logits = self.lm_head(self.norm(h))
        # Gather the vocabulary before selecting the sampler's device.
        logits = F.allgather(logits, tensor_axis=-1)
        return logits.cast(DType.float32)

    def _place(
        self, tokens: Tensor, input_row_offsets: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Replicates the single-device tokens and offsets across the mesh."""
        mesh = self.norm.weight.mesh
        if mesh.num_devices > 1:
            # Device graph capture records each device's stream on its own,
            # and a peer copy makes one stream wait on another, which
            # invalidates the capture. The collective syncs on the devices.
            return (
                F.distributed_broadcast(tokens, mesh),
                F.distributed_broadcast(input_row_offsets, mesh),
            )
        return tokens.to(mesh), input_row_offsets.to(mesh)

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *kv_inputs: Tensor,
    ) -> ModuleV3Outputs:
        if self.return_logits == ReturnLogits.VARIABLE:
            # Only `verify` builds the variable-logits range, from a host
            # count. This graph's count would land on the device, which
            # `F.range` rejects.
            raise NotImplementedError(
                "MiMo-V2 returns a variable number of logits only from its"
                " speculative verify pass"
            )
        del return_n_logits
        row_offsets = input_row_offsets
        kv_iter = iter(x._graph_value for x in kv_inputs)
        # One entry per device, in declaration order: sliding, then full.
        sliding_kv, full_kv = self.kv_params.unflatten_basic_kv_tree(kv_iter)
        # Whatever the KV tree left in the tail is the EP communication
        # buffers, in the order EPBatchManager.input_types declares them.
        comm: EPCommBuffers | None = None
        if self.ep_batch_manager is not None:
            comm = self.ep_batch_manager.comm_buffers(list(kv_iter))
        tokens, input_row_offsets = self._place(tokens, input_row_offsets)
        kv_by_group = {SLIDING: list(sliding_kv), FULL: list(full_kv)}

        h = self.embed_tokens(tokens)
        freqs_cis = {
            group: rope.freqs_cis.cast(h.dtype).to(h.mesh)
            for group, rope in self.ropes.items()
        }
        for i, layer in enumerate(self.layers):
            group = self._layer_groups[i]
            h = layer(
                h,
                kv_by_group[group],
                freqs_cis[group],
                input_row_offsets,
                comm,
            )

        # One projection and one gather serve both outputs when ALL logits
        # are requested: the head runs once on the full hidden states and
        # the last rows are gathered for the sampler. For LAST_TOKEN the
        # head runs once on just the rows the sampler needs.
        logits: Tensor | None = None
        offsets: Tensor | None = None
        last: Tensor
        if self.return_logits == ReturnLogits.ALL:
            all_logits = self._logits(h)
            last = F.gather(
                all_logits, input_row_offsets[1:] - 1, axis=0
            ).local_shards[0]
            logits = all_logits.local_shards[0]
            offsets = row_offsets
        else:
            last = self._logits(
                F.gather(h, input_row_offsets[1:] - 1, axis=0)
            ).local_shards[0]
        return ModuleV3Outputs(
            next_token_logits=last, logits=logits, logit_offsets=offsets
        )
