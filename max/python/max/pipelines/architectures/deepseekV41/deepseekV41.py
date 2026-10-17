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
"""The DeepSeek-V4.1 model: decoder blocks, the trunk and the full model.

Each decoder block wraps attention and a feed-forward network in
hyper-connections, which keep several copies of the residual stream and learn
how to mix them. Some blocks first add an Engram value: an embedding looked up
from hashed n-grams of the input tokens.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

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
from max.nn.embedding import Embedding
from max.nn.hyper_connection import HyperConnection
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm
from max.nn.transformer import logits_postprocess
from typing_extensions import Self

from ..deepseekV4.layers import (
    DeepseekV4Expert,
    DeepseekV4Gate,
    DeepseekV4MoE,
    DeepseekV4RotaryEmbedding,
    apply_rope_tail,
    expand_copies,
    hc_post,
    rope_for_layer,
)
from ..deepseekV4.layers.quantization import (
    fp4_qat_quantize,
    fp8_qat_quantize,
)
from .layer_resolver import LayerInfo
from .layer_resolver import resolve as resolve_layers
from .layers import final_collapse
from .layers.attention import (
    CSA2Attention,
    CSA2Cache,
    make_csa2_attn_factory,
)
from .layers.dspark_moe import DeepseekV41DSparkMoE
from .layers.engram import Engram
from .layers.engram_hash import EngramLayout
from .layers.engram_lookup import EngramLookup, EngramOperand, EngramTokens
from .layers.indexer import DeepseekV41Indexer
from .layers.moe import DeepseekV41MoE, nvfp4_moe_ffn_factory
from .layers.ragged import V41RaggedRows as RaggedRows
from .model_config import DeepseekV41Config, head_bf16_enabled

# Re-exports the V4 layers this model uses unchanged.
__all__ = [
    "DeepseekV4Expert",
    "DeepseekV4Gate",
    "DeepseekV4MoE",
    "DeepseekV4RotaryEmbedding",
    "DeepseekV41",
    "DeepseekV41Block",
    "DeepseekV41Trunk",
    "RaggedRows",
    "apply_rope_tail",
    "fp4_qat_quantize",
    "fp8_qat_quantize",
    "rope_for_layer",
]


def _adopt_shards(
    source: Module, replicas: Sequence[Module], devices: list[DeviceRef]
) -> None:
    """Points each replica's weights at the matching shard of ``source``'s.

    ``replicas`` are structurally identical to ``source`` and walked in
    step. A weight with no strategy is replicated; a ``Linear``'s goes
    through its setter so a block-scaled ``weight_scale`` is sharded with the
    weight.
    """
    n = len(devices)
    if isinstance(source, Linear) and source.sharding_strategy is None:
        source.sharding_strategy = ShardingStrategy.replicate(n)
    for name, value in list(vars(source).items()):
        if not isinstance(value, Weight):
            continue
        if value.sharding_strategy is None:
            value.sharding_strategy = ShardingStrategy.replicate(n)
        for replica, shard in zip(replicas, value.shard(devices), strict=True):
            setattr(replica, name, shard)
    for name, child in source.sublayers.items():
        _adopt_shards(
            child, [replica.sublayers[name] for replica in replicas], devices
        )


def _hyper_connection(
    config: DeepseekV41Config, device: DeviceRef
) -> HyperConnection:
    """Builds the hyper-connection around a block's attention or FFN."""
    return HyperConnection(
        config.hidden_size,
        config.hc_mult,
        device,
        DType.float32,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        rms_norm_eps=config.rms_norm_eps,
    )


def _hc_pre(
    x: TensorValue, hc: HyperConnection, pre_mix: TensorValue | None
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
    """Merges the ``[1, s, hc, d]`` stream copies into one ``[1, s, d]`` input.

    The copies are merged with the ``pre`` weights computed by the previous
    sublayer, never with this sublayer's own, so each sublayer's mixing
    weights are computed once and carried forward (Single-Pass mHC).
    ``pre_mix=None`` marks the first sublayer of a stack, whose copies are
    still identical; it selects copy 0.

    Returns:
        The merged stream and this sublayer's ``post``, ``comb`` and ``pre``
        mixing weights.
    """
    b, s = x.shape[0], x.shape[1]
    if pre_mix is None:
        folded_pre_mix = ops.constant(
            np.eye(hc.hc_mult, dtype=np.float32)[0],
            DType.float32,
            device=x.device,
        )
    else:
        folded_pre_mix = ops.reshape(pre_mix, [-1, hc.hc_mult])
    post, comb, y, pre = hc(
        ops.reshape(x, [-1, hc.hc_mult, hc.hidden_size]), folded_pre_mix
    )
    return (
        ops.reshape(y, [b, s, hc.hidden_size]),
        ops.reshape(post, [b, s, hc.hc_mult]),
        ops.reshape(comb, [b, s, hc.hc_mult, hc.hc_mult]),
        ops.reshape(pre, [b, s, hc.hc_mult]),
    )


def _hc_mean(x: TensorValue) -> TensorValue:
    """Averages ``[1, s, hc, d]`` over the copies, accumulating in float32."""
    mean = ops.mean(ops.cast(x, DType.float32), axis=2)
    return ops.cast(ops.squeeze(mean, 2), x.dtype)


def _apply_engram(
    x: TensorValue,
    engram_fn: Engram,
    kv: TensorValue,
    token_mask: TensorValue | None,
) -> TensorValue:
    """Gates the Engram value ``kv`` into the ``[1, s, hc, d]`` stream.

    Reshapes to ``[s, hc, d]`` for :meth:`Engram.gate` and back.
    """
    b, s, hc_mult, hidden_size = (
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
    )
    y = engram_fn.gate(
        ops.reshape(x, [-1, hc_mult, hidden_size]), kv, token_mask
    )
    return ops.reshape(y, [b, s, hc_mult, hidden_size])


class DeepseekV41Block(Module):
    """A decoder block: attention then FFN, each inside a hyper-connection.

    The block takes its ``attn`` and ``ffn`` modules and owns the
    hyper-connection mixing and the optional Engram step. Each call takes the
    previous sublayer's ``pre`` mixing weights and returns this block's FFN
    ``pre`` for the next block.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        layer_info: LayerInfo,
        device: DeviceRef,
        *,
        attn: Module,
        ffn: Module,
        engram_fn: Engram | None = None,
    ) -> None:
        super().__init__()
        if layer_info.has_engram and engram_fn is None:
            raise ValueError(
                f"layer {layer_info.layer_id} is an Engram layer "
                "(has_engram=True) but no engram_fn was given"
            )
        if not layer_info.has_engram and engram_fn is not None:
            raise ValueError(
                f"layer {layer_info.layer_id} is not an Engram layer "
                "(has_engram=False) but an engram_fn was given"
            )
        self.layer_info = layer_info
        self.attn = attn
        self.ffn = ffn
        self.engram_fn = engram_fn
        self.attn_norm = RMSNorm(
            config.hidden_size, DType.float32, config.rms_norm_eps
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size, DType.float32, config.rms_norm_eps
        )
        self.hc_attn = _hyper_connection(config, device)
        self.hc_ffn = _hyper_connection(config, device)

    @staticmethod
    def tensor_parallel(
        blocks: Sequence[DeepseekV41Block],
        xs: Sequence[TensorValue],
        pre_mixes: Sequence[TensorValue | None],
        rows: Sequence[RaggedRows],
        token_ids: Sequence[TensorValue],
        caches: Sequence[object],
        signal_buffers: Sequence[BufferValue],
        *,
        engram_operands: Sequence[EngramOperand] | None = None,
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        """Runs :meth:`__call__` on every device, one replica per device.

        Attention is split by heads, and its output-projection partial sums
        are all-reduced before the hyper-connection merge; for
        :class:`CSA2Attention`, indexer scores are also all-reduced before the
        top-k. The FFN all-reduces its experts' sum. The Engram projection is
        split by columns and all-gathered before the gate. The gate and both
        hyper-connections run in full on every device.

        Returns:
            Each device's output stream and FFN ``pre``.
        """
        moes: list[DeepseekV41MoE | DeepseekV41DSparkMoE] = []
        for block in blocks:
            assert isinstance(block.ffn, (DeepseekV41MoE, DeepseekV41DSparkMoE))
            moes.append(block.ffn)

        if blocks[0].engram_fn is not None:
            if engram_operands is None:
                raise ValueError(
                    f"layer {blocks[0].layer_info.layer_id} is an Engram "
                    "layer but no engram_operands was given"
                )
            engrams: list[Engram] = []
            for block in blocks:
                assert block.engram_fn is not None
                engrams.append(block.engram_fn)
            kvs = [
                engram.project(operand)
                for engram, operand in zip(
                    engrams, engram_operands, strict=True
                )
            ]
            if len(kvs) > 1:
                kvs = ops.allgather(kvs, signal_buffers, axis=-1)
            xs = [
                _apply_engram(x, engram, kv, None)
                for x, engram, kv in zip(xs, engrams, kvs, strict=True)
            ]

        attn_contracted = [
            _hc_pre(x, block.hc_attn, pre_mix)
            for block, x, pre_mix in zip(blocks, xs, pre_mixes, strict=True)
        ]
        if not isinstance(blocks[0].attn, CSA2Attention):
            partials = [
                block.attn(block.attn_norm(h), row, cache)
                for block, (h, _, _, _), row, cache in zip(
                    blocks, attn_contracted, rows, caches, strict=True
                )
            ]
        else:
            partials = DeepseekV41Block._csa2_partials(
                blocks, attn_contracted, rows, caches, signal_buffers
            )
        attended = ops.allreduce.sum(partials, signal_buffers)
        xs = [
            hc_post(h, residual, post, comb)
            for h, residual, (_, post, comb, _) in zip(
                attended, xs, attn_contracted, strict=True
            )
        ]
        return DeepseekV41Block._ffn_tensor_parallel(
            blocks, xs, attn_contracted, moes, token_ids, signal_buffers
        )

    @staticmethod
    def _csa2_partials(
        blocks: Sequence[DeepseekV41Block],
        attn_contracted: Sequence[
            tuple[TensorValue, TensorValue, TensorValue, TensorValue]
        ],
        rows: Sequence[RaggedRows],
        caches: Sequence[object],
        signal_buffers: Sequence[BufferValue],
    ) -> list[TensorValue]:
        """Returns each device's CSA2 output-projection partial sum.

        Indexer scores are all-reduced between :meth:`CSA2Attention.begin`
        and :meth:`CSA2Attention.end`.
        """
        attns: list[CSA2Attention] = []
        for block in blocks:
            assert isinstance(block.attn, CSA2Attention)
            attns.append(block.attn)
        states = [
            attn.begin(block.attn_norm(h), row, cache)
            for attn, block, (h, _, _, _), row, cache in zip(
                attns, blocks, attn_contracted, rows, caches, strict=True
            )
        ]
        if states[0].index_score is not None:
            scores = []
            for state in states:
                assert state.index_score is not None
                scores.append(state.index_score)
            # Row-tiled so each all-reduce fits the signal buffers.
            reduced = DeepseekV41Indexer.reduce_scores(
                scores, signal_buffers, max_rows=attns[0].max_seq_len
            )
            for state, score in zip(states, reduced, strict=True):
                state.index_score = score
        return [
            attn.end(state, row)
            for attn, state, row in zip(attns, states, rows, strict=True)
        ]

    @staticmethod
    def _ffn_tensor_parallel(
        blocks: Sequence[DeepseekV41Block],
        xs: Sequence[TensorValue],
        attn_contracted: Sequence[
            tuple[TensorValue, TensorValue, TensorValue, TensorValue]
        ],
        moes: Sequence[DeepseekV41MoE | DeepseekV41DSparkMoE],
        token_ids: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> tuple[list[TensorValue], list[TensorValue]]:
        """Runs the FFN half of :meth:`tensor_parallel`.

        The attention sublayer's ``pre`` weights merge the FFN's input.
        """
        ffn_contracted = [
            _hc_pre(x, block.hc_ffn, attn_pre)
            for block, x, (_, _, _, attn_pre) in zip(
                blocks, xs, attn_contracted, strict=True
            )
        ]
        moved = DeepseekV41MoE.tensor_parallel(
            moes,
            [
                block.ffn_norm(h)
                for block, (h, _, _, _) in zip(
                    blocks, ffn_contracted, strict=True
                )
            ],
            token_ids,
            signal_buffers,
        )
        return (
            [
                hc_post(h, x, post, comb)
                for h, x, (_, post, comb, _) in zip(
                    moved, xs, ffn_contracted, strict=True
                )
            ],
            [ffn_pre for _, _, _, ffn_pre in ffn_contracted],
        )

    def __call__(
        self,
        x: TensorValue,
        pre_mix: TensorValue | None,
        rows: RaggedRows,
        token_ids: TensorValue,
        cache: object | None = None,
        *,
        engram_operand: EngramOperand | None = None,
        engram_token_mask: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Runs the block on the ``[1, s, hc, d]`` stream and carried ``pre``.

        Args:
            x: The ``[1, s, hc, d]`` stream entering this block.
            pre_mix: The previous block's FFN ``pre``, or ``None`` for the
                first block.
            rows: Ragged-batch bookkeeping, forwarded to ``attn``.
            token_ids: Forwarded to ``ffn``.
            cache: Forwarded to ``attn``; opaque to this block.
            engram_operand: This layer's Engram table rows, gathered for
                these tokens. Required iff ``self.engram_fn is not None``.
            engram_token_mask: ``[s]`` bool; ``False`` turns Engram off at
                that position. ``None`` keeps it on everywhere.

        Returns:
            The block's output stream and its FFN ``pre``, carried into the
            next block or into the final merge after the last block.

        Raises:
            ValueError: If this is an Engram layer and ``engram_operand`` is
                missing.
        """
        if self.engram_fn is not None:
            if engram_operand is None:
                raise ValueError(
                    f"layer {self.layer_info.layer_id} is an Engram layer "
                    "but no engram_operand was given"
                )
            x = _apply_engram(
                x,
                self.engram_fn,
                self.engram_fn.project(engram_operand),
                engram_token_mask,
            )

        residual = x
        h, post, comb, attn_pre = _hc_pre(x, self.hc_attn, pre_mix)
        h = self.attn(self.attn_norm(h), rows, cache)
        x = hc_post(h, residual, post, comb)

        residual = x
        h, post, comb, ffn_pre = _hc_pre(x, self.hc_ffn, attn_pre)
        h = self.ffn(self.ffn_norm(h), token_ids)
        x = hc_post(h, residual, post, comb)

        return x, ffn_pre


class DeepseekV41Trunk(Module):
    """Embeds tokens, runs the decoder blocks and merges the stream copies.

    Attention, FFN and Engram are built per layer by ``attn_factory``,
    ``ffn_factory`` and ``engram_factory`` (the last only for ``has_engram``
    layers). Holds the target model's layers only, not the drafter's.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        device: DeviceRef,
        max_seq_len: int,
        *,
        attn_factory: Callable[
            [DeepseekV41Config, LayerInfo, DeviceRef, int], Module
        ],
        ffn_factory: Callable[
            [DeepseekV41Config, LayerInfo, DeviceRef], Module
        ],
        engram_factory: Callable[
            [DeepseekV41Config, LayerInfo, DeviceRef], Engram
        ]
        | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        # bf16, not `config.dtype` (the packed quantized dtype): the
        # embedding is stored unquantized.
        self.embed = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=DType.bfloat16,
            device=device,
        )
        self.layer_infos = [
            row for row in resolve_layers(config) if row.side != "drafter"
        ]
        self.layers = LayerList(
            [
                DeepseekV41Block(
                    config,
                    row,
                    device,
                    attn=attn_factory(config, row, device, max_seq_len),
                    ffn=ffn_factory(config, row, device),
                    engram_fn=(
                        engram_factory(config, row, device)
                        if row.has_engram and engram_factory is not None
                        else None
                    ),
                )
                for row in self.layer_infos
            ]
        )

    def __call__(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        cache: object | None = None,
        *,
        engram_operands: Mapping[int, EngramOperand] | None = None,
        engram_token_mask: TensorValue | None = None,
    ) -> TensorValue:
        """Runs the trunk and returns the pre-norm ``[1, s, d]`` stream.

        Args:
            tokens: ``[1, s]`` token ids.
            rows: Ragged-batch bookkeeping, forwarded to every ``attn``.
            cache: Forwarded to every ``attn`` unchanged; opaque here.
            engram_operands: Each Engram layer's table rows gathered for
                these tokens, keyed by ``layer_id``; required for every
                ``has_engram`` layer.
            engram_token_mask: ``[s]`` bool; ``False`` turns Engram off at
                that position. ``None`` keeps it on everywhere.

        Returns:
            ``[1, s, d]`` pre-norm hidden states.
        """
        x, _ = self.forward_with_taps(
            tokens,
            rows,
            cache,
            tap_layer_ids=(),
            engram_operands=engram_operands,
            engram_token_mask=engram_token_mask,
        )
        return x

    def forward_with_taps(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        cache: object | None = None,
        *,
        tap_layer_ids: Sequence[int],
        engram_operands: Mapping[int, EngramOperand] | None = None,
        engram_token_mask: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue | None]:
        """Runs :meth:`__call__`, also returning hidden states at some layers.

        A tap is the ``[1, s, hc, d]`` stream entering a layer in
        ``tap_layer_ids``, averaged over its copies. ``tap_layer_ids=()``
        adds nothing to the graph.

        Returns:
            The pre-norm ``[1, s, d]`` stream, and the taps concatenated in
            layer order, ``[1, s, d * len(tap_layer_ids)]`` (``None`` for no
            taps).

        Raises:
            ValueError: If a tap id is not a trunk layer, repeats, or is an
                Engram layer (its tap would have to follow the Engram step,
                which happens inside the block).
        """
        tap_ids = set(tap_layer_ids)
        if len(tap_ids) != len(tap_layer_ids):
            raise ValueError(f"repeated tap layer ids: {list(tap_layer_ids)}")
        trunk_ids = {row.layer_id for row in self.layer_infos}
        if not tap_ids <= trunk_ids:
            raise ValueError(
                f"tap layer ids {sorted(tap_ids - trunk_ids)} are not trunk "
                "layers"
            )
        taps: list[TensorValue] = []
        x = expand_copies(self.embed(tokens), self.config.hc_mult)
        pre_mix: TensorValue | None = None
        for layer_info, layer in zip(
            self.layer_infos, self.layers, strict=True
        ):
            if layer_info.layer_id in tap_ids:
                if layer_info.has_engram:
                    raise ValueError(
                        f"tap layer {layer_info.layer_id} is an Engram layer"
                    )
                taps.append(_hc_mean(x))
            operand = None
            if layer_info.has_engram:
                if (
                    engram_operands is None
                    or layer_info.layer_id not in engram_operands
                ):
                    raise ValueError(
                        f"layer {layer_info.layer_id} is an Engram layer "
                        "but engram_operands has no entry for it"
                    )
                operand = engram_operands[layer_info.layer_id]
            x, pre_mix = layer(
                x,
                pre_mix,
                rows,
                tokens,
                cache,
                engram_operand=operand,
                engram_token_mask=engram_token_mask,
            )
        assert pre_mix is not None  # at least one trunk layer always runs
        return final_collapse(x, pre_mix), (
            ops.concat(taps, axis=-1) if taps else None
        )

    @staticmethod
    def tensor_parallel(
        trunks: Sequence[DeepseekV41Trunk],
        tokens: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        *,
        engram_operands: Sequence[Mapping[int, EngramOperand]],
    ) -> TensorValue:
        """Runs :meth:`__call__` over one trunk replica per device.

        Every argument but ``signal_buffers`` is per device.

        Returns:
            Device 0's ``[1, s, d]`` stream; every device holds the same one.
        """
        x, _ = DeepseekV41Trunk.tensor_parallel_with_taps(
            trunks,
            tokens,
            rows,
            caches,
            signal_buffers,
            tap_layer_ids=(),
            engram_operands=engram_operands,
        )
        return x

    @staticmethod
    def tensor_parallel_with_taps(
        trunks: Sequence[DeepseekV41Trunk],
        tokens: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        *,
        tap_layer_ids: Sequence[int],
        engram_operands: Sequence[Mapping[int, EngramOperand]],
    ) -> tuple[TensorValue, list[TensorValue] | None]:
        """Runs :meth:`forward_with_taps` over one trunk replica per device.

        Returns:
            Device 0's ``[1, s, d]`` stream and, per device, the identical
            ``[1, s, d * len(tap_layer_ids)]`` taps (``None`` for no taps).
        """
        lead = trunks[0]
        tap_ids = set(tap_layer_ids)
        if len(tap_ids) != len(tap_layer_ids):
            raise ValueError(f"repeated tap layer ids: {list(tap_layer_ids)}")
        trunk_ids = {row.layer_id for row in lead.layer_infos}
        if not tap_ids <= trunk_ids:
            raise ValueError(
                f"tap layer ids {sorted(tap_ids - trunk_ids)} are not trunk "
                "layers"
            )
        taps: list[list[TensorValue]] = [[] for _ in trunks]
        xs = [
            expand_copies(trunk.embed(tok), lead.config.hc_mult)
            for trunk, tok in zip(trunks, tokens, strict=True)
        ]
        pre_mixes: Sequence[TensorValue | None] = [None] * len(trunks)
        for i, layer_info in enumerate(lead.layer_infos):
            if layer_info.layer_id in tap_ids:
                if layer_info.has_engram:
                    raise ValueError(
                        f"tap layer {layer_info.layer_id} is an Engram layer"
                    )
                for per_dev, x in zip(taps, xs, strict=True):
                    per_dev.append(_hc_mean(x))
            blocks = []
            for trunk in trunks:
                block = trunk.layers[i]
                assert isinstance(block, DeepseekV41Block)
                blocks.append(block)
            xs, pre_mixes = DeepseekV41Block.tensor_parallel(
                blocks,
                xs,
                pre_mixes,
                rows,
                tokens,
                caches,
                signal_buffers,
                engram_operands=(
                    [
                        operands[layer_info.layer_id]
                        for operands in engram_operands
                    ]
                    if layer_info.has_engram
                    else None
                ),
            )
        assert pre_mixes[0] is not None  # at least one trunk layer always runs
        return final_collapse(xs[0], pre_mixes[0]), (
            [ops.concat(per_dev, axis=-1) for per_dev in taps]
            if tap_layer_ids
            else None
        )


def _engram_factory(
    config: DeepseekV41Config, layer_info: LayerInfo, device: DeviceRef
) -> Engram:
    """Builds one Engram layer.

    The Engram tables are not module weights: they stay in pinned host memory
    and are gathered once per forward, outside the layer.
    """
    del layer_info
    n_hash_cols = (config.engram_max_ngram_size - 1) * config.engram_n_heads
    # bf16 compute, not the packed `config.dtype`.
    return Engram(
        hidden_size=config.hidden_size,
        hc_mult=config.hc_mult,
        n_hash_cols=n_hash_cols,
        head_dim=config.engram_head_dim,
        rms_norm_eps=config.rms_norm_eps,
        device=device,
        dtype=DType.bfloat16,
    )


def _csa2_num_comp(config: DeepseekV41Config) -> int:
    """Returns the static ``comp_indices`` width of every CSA2 layer.

    Equals ``config.index_topk``; rows with fewer visible entries are
    ``-1``-padded by the top-k. Defaults to 512 when the field is unset.
    """
    return config.index_topk if config.index_topk is not None else 512


class DeepseekV41(Module):
    """The DeepSeek-V4.1 model: trunk, final norm and LM head.

    ``__call__`` takes ragged-batch bookkeeping (``rows``) because the Engram
    lookup needs each request's positions; :meth:`serve` builds ``rows`` from
    the serving inputs.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        device: DeviceRef | None = None,
        *,
        engram_host_tables: Mapping[int, tuple[TensorValue, TensorValue]],
        engram_token_map: np.ndarray | None = None,
    ) -> None:
        """Builds the model tree.

        Args:
            config: The resolved model config.
            device: The device this instance's weights/ops live on; defaults
                to ``config.devices[0]``.
            engram_host_tables: This device's pinned, host-resident
                ``(codes, scales)`` table pair per Engram layer id: the full
                tables on one device, else this rank's rows. Pass ``{}`` for
                a config with no Engram layers.
            engram_token_map: The compressed token map derived from the
                tokenizer. Required when ``config`` declares Engram layers;
                ``None`` otherwise.
        """
        super().__init__()
        self.config = config
        if device is None:
            device = config.devices[0]
        self.device = device

        attn_factory, _ = make_csa2_attn_factory(
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            window=config.sliding_window,
            num_comp=_csa2_num_comp(config),
        )
        self.trunk = DeepseekV41Trunk(
            config,
            device,
            config.max_seq_len,
            attn_factory=attn_factory,
            ffn_factory=nvfp4_moe_ffn_factory,
            engram_factory=_engram_factory,
        )
        self.norm = RMSNorm(
            config.hidden_size, DType.float32, config.rms_norm_eps
        )
        # bfloat16 by default; ``MAX_DSV41_HEAD_BF16=0`` keeps the head in
        # float32 (upcast on load).
        self.head_bf16 = head_bf16_enabled()
        self.head = Linear(
            config.hidden_size,
            config.vocab_size,
            DType.bfloat16 if self.head_bf16 else DType.float32,
            device,
        )

        self._engram_layout = EngramLayout.from_config(config)
        self._engram_token_map = (
            engram_token_map if self._engram_layout is not None else None
        )
        self._engram_host_tables = engram_host_tables
        self._engram_lookup_cache: EngramLookup | None = None

    def _lm_head(self, x: TensorValue) -> TensorValue:
        """Returns the head's float32 ``[t, vocab]`` logits.

        With a bfloat16 head (the default) the input is rounded to bfloat16
        and the bfloat16 logits are widened to float32. Otherwise the matmul
        runs in float32.
        """
        if self.head_bf16:
            return ops.cast(
                ops.matmul(
                    ops.cast(x, DType.bfloat16),
                    ops.transpose(self.head.weight, 0, 1),
                ),
                DType.float32,
            )
        return ops.matmul(
            ops.cast(x, DType.float32),
            ops.transpose(ops.cast(self.head.weight, DType.float32), 0, 1),
        )

    def _engram_lookup(self) -> EngramLookup | None:
        """Returns the Engram ops' builder, or ``None`` without Engram."""
        if self._engram_layout is None:
            return None
        if self._engram_lookup_cache is None:
            assert self._engram_token_map is not None
            self._engram_lookup_cache = EngramLookup(
                self._engram_layout, self._engram_token_map
            )
        return self._engram_lookup_cache

    def _engram_tables(self, lookup: EngramLookup) -> list[TensorValue]:
        """Returns this device's table inputs, layer-major."""
        return [
            table
            for layer_id in lookup.layer_ids
            for table in self._engram_host_tables[layer_id]
        ]

    def _engram_tokens(
        self,
        lookup: EngramLookup,
        tokens: TensorValue,
        rows: RaggedRows,
        lookback_token_ids: TensorValue | None,
    ) -> EngramTokens:
        """Returns this forward's Engram hash inputs on this device.

        ``lookback_token_ids`` is ``[b, max_ngram_size - 1]``: column ``j``
        holds the token at ``start - 1 - j``, ``-1`` before the sequence
        start. ``None`` means every request starts at position 0. No position
        is masked out of Engram.
        """
        depth = lookup.lookback_depth
        ids = ops.cast(ops.reshape(tokens, [-1]), DType.int32)
        total = ids.shape[0]
        num_reqs = rows.batch
        if lookback_token_ids is None:
            lookback = ops.broadcast_to(
                ops.constant(-1, DType.int32, self.device), [num_reqs, depth]
            )
        else:
            lookback = ops.rebind(
                ops.cast(lookback_token_ids, DType.int32), [num_reqs, depth]
            )
        return EngramTokens(
            ids=ids,
            positions=ops.cast(rows.positions, DType.int64),
            query_start_loc=ops.cast(rows.offsets, DType.int32),
            dead_mask=ops.broadcast_to(
                ops.constant(False, DType.bool, self.device), [total]
            ),
            lookback=lookback,
            lookback_dead=ops.broadcast_to(
                ops.constant(False, DType.bool, self.device), [num_reqs, depth]
            ),
        )

    def _engram_operands(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        lookback_token_ids: TensorValue | None = None,
    ) -> Mapping[int, EngramOperand]:
        """Returns each Engram layer's gathered rows from the full tables.

        ``lookback_token_ids``: see :meth:`_engram_tokens`.
        """
        lookup = self._engram_lookup()
        if lookup is None:
            return {}
        return lookup.full(
            self._engram_tokens(lookup, tokens, rows, lookback_token_ids),
            self._engram_tables(lookup),
            self.device,
        )

    @staticmethod
    def _engram_operands_tensor_parallel(
        replicas: Sequence[DeepseekV41],
        tokens: Sequence[TensorValue],
        rows: Sequence[RaggedRows],
        lookback_token_ids: Sequence[TensorValue] | None,
        signal_buffers: Sequence[BufferValue],
    ) -> list[Mapping[int, EngramOperand]]:
        """Runs :meth:`_engram_operands` over replicas holding table shards.

        Each rank gathers its hash columns from its own rows, then the ranks
        all-gather and assemble the full operands on every device.
        """
        lookup = replicas[0]._engram_lookup()
        if lookup is None:
            return [{} for _ in replicas]
        lookbacks: Sequence[TensorValue | None] = (
            lookback_token_ids
            if lookback_token_ids is not None
            else [None] * len(replicas)
        )
        return list(
            lookup.sharded(
                [
                    replica._engram_tokens(lookup, tok, row, lookback)
                    for replica, tok, row, lookback in zip(
                        replicas, tokens, rows, lookbacks, strict=True
                    )
                ],
                [replica._engram_tables(lookup) for replica in replicas],
                [replica.device for replica in replicas],
                signal_buffers,
            )
        )

    def __call__(
        self,
        tokens: TensorValue,
        rows: RaggedRows,
        cache: object | None = None,
        engram_lookback_ids: TensorValue | None = None,
    ) -> TensorValue:
        """Runs the trunk and returns the pre-norm ``[1, s, d]`` stream.

        ``engram_lookback_ids``: see :meth:`_engram_tokens`.
        """
        operands = self._engram_operands(tokens, rows, engram_lookback_ids)
        return self.trunk(tokens, rows, cache, engram_operands=operands)

    def serve(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        return_n_logits: TensorValue,
        cache: object,
        engram_lookback_ids: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Builds the serving graph body: ragged tokens in, logits out.

        ``engram_lookback_ids``: see :meth:`_engram_tokens`.

        Raises:
            TypeError: If ``cache`` is not a :class:`CSA2Cache`.
        """
        if not isinstance(cache, CSA2Cache):
            raise TypeError(
                f"DeepseekV41.serve requires a CSA2Cache, got {type(cache)!r}"
            )
        t = tokens.shape[0]
        starts = ops.cast(cache.swa_collection.cache_lengths, DType.int32)
        rows = RaggedRows.from_offsets(input_row_offsets, t, starts)
        x = self(
            ops.reshape(tokens, [1, t]),
            rows,
            cache,
            engram_lookback_ids=engram_lookback_ids,
        )
        return logits_postprocess(
            ops.reshape(x, [t, self.config.hidden_size]),
            input_row_offsets,
            return_n_logits,
            norm=self.norm,
            lm_head=self._lm_head,
            return_logits=self.config.return_logits,
            return_hidden_states=self.config.return_hidden_states,
        )

    def tensor_parallel_replicas(
        self,
        devices: Sequence[DeviceRef],
        engram_host_tables: Sequence[
            Mapping[int, tuple[TensorValue, TensorValue]]
        ],
    ) -> list[Self]:
        """Builds one copy of the model per device, for the multi-device graph.

        Each copy's weights are shards of this model's, which stays the
        source the weights are loaded into. Attention (with its indexer)
        splits heads and the MoE splits routed experts. Engram's projection
        splits columns, and each replica reads only its rank's rows of the
        host tables, so table traffic splits across ranks. All else is
        replicated. Call once, after :meth:`load_state_dict`.

        Args:
            devices: One device per replica.
            engram_host_tables: One ``engram_host_tables`` mapping (as in
                the constructor) per device, in ``devices`` order: rank
                ``i``'s rows of each table.

        Returns:
            The per-device replicas.
        """
        n = len(devices)
        for block in self.trunk.layers:
            assert isinstance(block, DeepseekV41Block)
            assert isinstance(block.attn, CSA2Attention)
            assert isinstance(block.ffn, DeepseekV41MoE)
            block.attn.shard_heads(n)
            block.ffn.shard_experts(n)
            if block.engram_fn is not None:
                block.engram_fn.shard_columns(n)
        # `type(self)` so subclasses replicate their extra submodules too.
        replicas = [
            type(self)(
                self.config,
                device,
                engram_host_tables=tables,
                engram_token_map=self._engram_token_map,
            )
            for device, tables in zip(devices, engram_host_tables, strict=True)
        ]
        _adopt_shards(self, replicas, list(devices))
        for rank, replica in enumerate(replicas):
            for block in replica.trunk.layers:
                assert isinstance(block, DeepseekV41Block)
                assert isinstance(block.attn, CSA2Attention)
                assert isinstance(block.ffn, DeepseekV41MoE)
                block.attn.keep_local_heads(n)
                block.ffn.keep_local_experts(n, rank)
        return replicas

    @staticmethod
    def serve_tensor_parallel(
        replicas: Sequence[DeepseekV41],
        tokens: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        return_n_logits: TensorValue,
        caches: Sequence[CSA2Cache],
        signal_buffers: Sequence[BufferValue],
        engram_lookback_ids: Sequence[TensorValue] | None = None,
    ) -> tuple[TensorValue, ...]:
        """Runs :meth:`serve` with one replica per device.

        ``tokens``, ``input_row_offsets``, ``caches`` and
        ``engram_lookback_ids`` are per device; logits come from device 0.
        """
        lead = replicas[0]
        t = tokens[0].shape[0]
        rows = [
            RaggedRows.from_offsets(
                offsets,
                t,
                ops.cast(cache.swa_collection.cache_lengths, DType.int32),
            )
            for offsets, cache in zip(input_row_offsets, caches, strict=True)
        ]
        tokens_2d = [ops.reshape(tok, [1, t]) for tok in tokens]
        x = DeepseekV41Trunk.tensor_parallel(
            [replica.trunk for replica in replicas],
            tokens_2d,
            rows,
            caches,
            signal_buffers,
            engram_operands=DeepseekV41._engram_operands_tensor_parallel(
                replicas, tokens_2d, rows, engram_lookback_ids, signal_buffers
            ),
        )
        return logits_postprocess(
            ops.reshape(x, [t, lead.config.hidden_size]),
            input_row_offsets[0],
            return_n_logits,
            norm=lead.norm,
            lm_head=lead._lm_head,
            return_logits=lead.config.return_logits,
            return_hidden_states=lead.config.return_hidden_states,
        )
