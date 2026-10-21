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
"""Block spec-decode adapters for a ModuleV3 MiMo-V2 target and DFlash drafter.

Both are :class:`~max.pipelines.speculative.driver.TensorAdapter` values, so
the block driver hands them ``Tensor`` values, one per device where it holds
a list. The target also grows the verify pass the base module refuses to
build: a variable-logits window scored from a host count, and the
``SELECTED_LAYERS`` taps the drafter's ``fc`` reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import DeviceMapping, DeviceMesh
from max.experimental.tensor import Tensor
from max.graph import BufferType, TensorType, TensorValue
from max.nn.kv_cache import KVCacheInputsPerDevice, PagedCacheValues
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.speculative._tensor_compat import as_graph_values
from max.pipelines.speculative.block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)
from max.pipelines.speculative.driver import (
    TensorAdapter,
    _for_adapter,
    _from_adapter,
)
from max.pipelines.speculative.spec_target import Verified

from ..deepseekV3.deepseekV3 import mask_padded_tail
from ..dflash_mimo_v2_modulev3 import DFlashMiMoV2
from ..mimo_v2_modulev3.mimo_v2 import MiMoV2
from ..mimo_v2_modulev3.model_config import FULL, SLIDING, MiMoV2Config

__all__ = [
    "DFlashMiMoV2Proposer",
    "MiMoV2BlockTarget",
    "MiMoV2SpecTarget",
    "verify_fn",
]

_Batch = BlockBatch[Tensor, Tensor, Tensor]
_Caches = list[KVCacheInputsPerDevice[Tensor, Tensor]]


def verify_fn(
    target: MiMoV2BlockTarget,
) -> Callable[[BlockBatch], Verified[Any, TensorValue]]:
    """The block target's verify behind the driver's callable argument.

    The driver hands the traced graph's values to ``verify``; the target's
    layers are ModuleV3 ``Tensor`` values. The conversion pair is the one
    the driver's per-adapter sites apply to the proposer, kept at the
    boundary here since the verify callable is this package's own.
    """

    def _verify(
        batch: BlockBatch,
    ) -> Verified[Any, TensorValue]:
        # The conversion pairs the driver applies to the proposer's calls:
        # graph values in, the adapter's Tensor results back out. The
        # driver's captured-hidden payload stays opaque to it.
        return _from_adapter(target, target.verify(_for_adapter(target, batch)))

    return _verify


class MiMoV2SpecTarget(MiMoV2):
    """The fused graph's MiMo-V2 target: a verify pass, not a forward.

    The base module returns neither variable logits nor hidden states: its
    ``F.range`` count would land on the device, and its per-device taps are a
    speculative-decode concern. Both are built here instead, where the verify
    hands the count in from the host and the taps reach the drafter's ``fc``
    without leaving the graph.
    """

    def __init__(self, config: MiMoV2Config) -> None:
        if not config.target_layer_ids:
            raise ValueError(
                "MiMo-V2 DFlash: the spec target captures its taps at the"
                " drafter's target_layer_ids, which are unset."
            )
        if config.target_layer_ids != sorted(config.target_layer_ids):
            raise ValueError(
                "MiMo-V2 DFlash: the target captures its taps in layer order;"
                f" got target_layer_ids {config.target_layer_ids}."
            )
        # Build with the base's narrow ABI and widen after: the base module
        # refuses hidden states at build, and the verify pass is the only
        # consumer of the wide ones. The base's NONE guard narrows the
        # attributes to the literal, so the widening needs the cast comments.
        super().__init__(
            replace(
                config,
                return_logits=ReturnLogits.LAST_TOKEN,
                return_hidden_states=ReturnHiddenStates.NONE,
            )
        )
        self.return_logits = ReturnLogits.VARIABLE
        self.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        self.target_layer_ids = list(config.target_layer_ids)
        self._target_layers = set(self.target_layer_ids)

    def verify(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        kv_by_group: Mapping[str, Sequence[PagedCacheValues]],
    ) -> tuple[Tensor, list[Tensor]]:
        """Runs the target over the merged rows, at verify width.

        Args:
            tokens: The single-device merged tokens, ``[merged_seq_len]``.
            return_n_logits: How many trailing rows of each request score
                logits; a host tensor, as the logit range needs.
            input_row_offsets: The single-device ragged offsets.
            kv_by_group: The target's caches, one per device per KV group,
                keyed ``sliding_attention`` and ``full_attention``.

        Returns:
            The ``[batch * count, vocab]`` float32 logits and the residual
            stream after each tapped layer, in layer order, both replicated.
        """
        tokens, input_row_offsets = self._place(tokens, input_row_offsets)
        h = self.embed_tokens(tokens)
        freqs_cis = {
            group: rope.freqs_cis.cast(h.dtype).to(h.mesh)
            for group, rope in self.ropes.items()
        }
        taps: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            group = self._layer_groups[i]
            h = layer(
                h, kv_by_group[group], freqs_cis[group], input_row_offsets
            )
            if i in self._target_layers:
                taps.append(h)
        # The count is a host tensor, so `F.range` takes it; a device count
        # it would reject. Each request scores its trailing `count` rows.
        count = return_n_logits[0]
        distance = F.range(
            count,
            0,
            -1,
            out_dim="return_n_logits_range",
            dtype=DType.int64,
            device=h.mesh,
        )
        indices = (input_row_offsets[1:].unsqueeze(-1) - distance).reshape(
            (-1,)
        )
        logits = self._logits(F.gather(h, indices, axis=0))
        return logits, taps

    def draft_logits(self, h: Tensor) -> Tensor:
        """The target's head over the drafter's own final-norm output.

        The drafter normalizes its block before this head reads it, so the
        target's final norm must not run again.
        """
        logits = self.lm_head(h)
        return F.allgather(logits, tensor_axis=-1).cast(DType.float32)


class MiMoV2BlockTarget(TensorAdapter):
    """The ModuleV3 MiMo-V2 target over the merged rows, returning its taps.

    Acceptance samples its recovered and bonus tokens from these logits, so
    they carry the ``lm_head`` padding rows at negative infinity.
    """

    def __init__(
        self,
        target: MiMoV2SpecTarget,
        *,
        vocab_size: int,
        sampleable_vocab_size: int,
    ) -> None:
        self.target = target
        self.vocab_size = vocab_size
        self.sampleable_vocab_size = sampleable_vocab_size

    def verify(self, batch: _Batch) -> Verified[list[Tensor], Tensor]:
        # The paged-cache kernels take the same cache leaves as graph values,
        # so the buffers the kernels write in place stay buffers through the
        # round trip.
        logits, taps = self.target.verify(
            batch.merged_tokens,
            batch.return_n_logits,
            batch.merged_offsets,
            {
                FULL: as_graph_values(list(batch.kv_collections)),
                SLIDING: as_graph_values(list(batch.passthrough_kv[SLIDING])),
            },
        )
        n_devices = batch.n_devs
        # Device-major, as the graph-API target's capture tail is: the
        # proposer rebuilds one replicated tensor per tap from the copies.
        hidden = [
            taps[t].local_shards[d]
            for d in range(n_devices)
            for t in range(len(taps))
        ]
        return Verified(
            logits=Tensor.from_graph_value(
                mask_padded_tail(
                    TensorValue(logits.local_shards[0]),
                    self.vocab_size,
                    self.sampleable_vocab_size,
                )
            ),
            hidden=list(hidden),
        )

    def ep_input_types(self) -> Sequence[TensorType | BufferType]:
        return ()


class DFlashMiMoV2Proposer(TensorAdapter):
    """The DFlash block drafter, reusing the target's embedding and head.

    The block is always ``block_size`` wide, its trained width; the head
    reads the first ``num_speculative_tokens`` slots after the anchor.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = False

    def __init__(
        self,
        drafter: DFlashMiMoV2,
        target: MiMoV2SpecTarget,
        *,
        num_speculative_tokens: int,
        sampleable_vocab_size: int,
    ) -> None:
        self.drafter = drafter
        self.target = target
        self.block_size = drafter.config.block_size
        self.mask_token_id = drafter.config.mask_token_id
        self.num_speculative_tokens = num_speculative_tokens
        self.sampleable_vocab_size = sampleable_vocab_size
        self.mesh: DeviceMesh = target.norm.weight.mesh

    def materialize(
        self,
        batch: _Batch,
        target_hidden: list[Tensor],
        ctx_kv: _Caches,
    ) -> None:
        self.drafter.write_context(
            self._taps(target_hidden, batch.n_devs),
            self._replicated(batch.merged_offsets_per_dev),
            as_graph_values(list(ctx_kv)),
        )

    def embed_block(self, batch: _Batch, block_ids: Tensor) -> list[Tensor]:
        # The mask slots take the drafter's trained embedding; only the
        # anchor is a token.
        anchors = block_ids.reshape((-1, self.block_size))[:, 0]
        embeds = self.target.embed_tokens(self._replicate(anchors))
        # A broadcast over the mesh, not a GPU-to-GPU transfer, which
        # device graph capture cannot record.
        block = self.drafter.block_embeddings(embeds)
        return [block.local_shards[d] for d in range(self.mesh.num_devices)]

    def forward_block(
        self,
        batch: _Batch,
        embeds: list[Tensor],
        offsets: list[Tensor],
        block_kv: _Caches,
    ) -> list[Tensor]:
        hidden = self.drafter.forward_block(
            self._replicated(embeds),
            block_kv_with_dispatch(
                as_graph_values(list(block_kv)), self.block_size
            ),
            self._replicated(offsets),
        )
        return [hidden.local_shards[d] for d in range(self.mesh.num_devices)]

    def head(
        self,
        batch: _Batch,
        block_hs: list[Tensor],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> Tensor:
        del accepted
        block, drafts = self.block_size, self.num_speculative_tokens
        hidden_size = self.drafter.config.hidden_size
        # Drop the anchor slot: it carries the committed token, not a draft.
        drafted = (
            self._replicated(block_hs)
            .reshape([-1, block, hidden_size])[:, 1 : 1 + drafts, :]
            .reshape([-1, hidden_size])
        )
        logits = self.target.draft_logits(drafted)
        vocab = int(logits.shape[-1])
        # A padding id could never be accepted, so it would waste a slot.
        base = mask_padded_tail(
            TensorValue(logits.local_shards[0]),
            vocab,
            self.sampleable_vocab_size,
        )
        tokens = sampler.sample_all(
            base.reshape([-1, drafts, vocab]).rebind(
                ["batch_size", drafts, vocab]
            )
        )
        return Tensor.from_graph_value(tokens)

    def _taps(self, hidden: list[Tensor], n_devices: int) -> list[Tensor]:
        """One replicated tensor per tap, from the device-major copies."""
        per_device = len(hidden) // n_devices
        return [
            Tensor.from_shard_values(
                [hidden[d * per_device + t] for d in range(n_devices)],
                DeviceMapping.replicated(self.mesh),
            )
            for t in range(per_device)
        ]

    def _replicated(self, values: list[Tensor]) -> Tensor:
        """One replicated tensor from each device's copy."""
        return Tensor.from_shard_values(
            list(values), DeviceMapping.replicated(self.mesh)
        )

    def _replicate(self, value: Tensor) -> Tensor:
        """Places a single-device tensor on every device of the mesh."""
        if self.mesh.num_devices > 1:
            return F.distributed_broadcast(value, self.mesh)
        return value.to(self.mesh)
