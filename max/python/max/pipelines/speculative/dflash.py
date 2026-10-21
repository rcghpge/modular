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
"""The DFlash proposer: a block of drafts from one parallel forward."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from max.dtype import DType
from max.graph import BufferValue, TensorValue, ops
from max.nn.embedding import Embedding, VocabParallelEmbedding
from max.nn.kv_cache import PagedCacheValues
from max.nn.linear import ColumnParallelLinear, Linear

from .block_driver import (
    Accepted,
    BlockBatch,
    DraftSampler,
    block_kv_with_dispatch,
)

__all__ = ["DFlashDraft", "DFlashProposer"]


@runtime_checkable
class DFlashDraft(Protocol):
    """What the DFlash proposer calls on its draft module.

    Every argument and result is per device, so a draft on one device takes
    and returns lists of one. The draft has no embedding or head of its own;
    it holds its target's.
    """

    @property
    def embed_tokens(self) -> object:
        """The target's embedding."""
        ...

    @property
    def lm_head(self) -> object:
        """The target's head."""
        ...

    def materialize(
        self,
        target_hidden: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        signal_buffers: Sequence[BufferValue],
        /,
    ) -> None:
        """Writes the context's K/V into the draft's cache.

        ``target_hidden`` holds the target's captured layers, concatenated.
        """
        ...

    def forward_block(
        self,
        input_embeds: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        /,
    ) -> list[TensorValue]:
        """Runs the draft over one embedded block."""
        ...


class DFlashProposer:
    """Drafts a block through the target's embedding and head.

    The target's scales are applied here, since neither its embedding nor
    its head applies them.
    """

    samples_from_anchor = False
    supports_zero_draft_rows = True

    def __init__(
        self,
        draft: DFlashDraft,
        *,
        block_size: int,
        mask_token_id: int,
        hidden_size: int,
        embedding_multiplier: float = 1.0,
        logits_scaling: float = 1.0,
        final_logit_softcapping: float | None = None,
    ) -> None:
        self.draft = draft
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.hidden_size = hidden_size
        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.final_logit_softcapping = final_logit_softcapping

    def materialize(
        self,
        batch: BlockBatch,
        target_hidden: list[TensorValue],
        ctx_kv: list[PagedCacheValues],
    ) -> None:
        """Writes the target's context K/V into the draft's cache."""
        self.draft.materialize(
            target_hidden,
            batch.merged_offsets_per_dev,
            ctx_kv,
            batch.signal_buffers,
        )

    def embed_block(
        self, batch: BlockBatch, block_ids: TensorValue
    ) -> list[TensorValue]:
        """Embeds the block, scaled as the target scales its own."""
        embed = self.draft.embed_tokens
        if isinstance(embed, VocabParallelEmbedding):
            embeds = embed(block_ids, batch.signal_buffers)
        else:
            assert isinstance(embed, Embedding)
            embeds = [embed(block_ids)]
        if self.embedding_multiplier != 1.0:
            embeds = [e * float(self.embedding_multiplier) for e in embeds]
        return embeds

    def forward_block(
        self,
        batch: BlockBatch,
        embeds: list[TensorValue],
        offsets: list[TensorValue],
        block_kv: list[PagedCacheValues],
    ) -> list[TensorValue]:
        """Runs the draft over the embedded block."""
        return self.draft.forward_block(
            embeds,
            block_kv_with_dispatch(block_kv, self.block_size),
            offsets,
            batch.signal_buffers,
        )

    def head(
        self,
        batch: BlockBatch,
        block_hs: list[TensorValue],
        accepted: Accepted,
        sampler: DraftSampler,
    ) -> TensorValue:
        """Scores the drafted slots through the head and samples them."""
        del accepted
        k, hidden = self.block_size, self.hidden_size
        seqs = batch.num_draft_seqs
        # Drop the anchor slot: it carries the committed token, not a proposal.
        drafted = [
            hs.rebind([seqs * k, hidden]).reshape((seqs, k, hidden))[:, 1:, :]
            for hs in block_hs
        ]
        lm_head = self.draft.lm_head
        if isinstance(lm_head, ColumnParallelLinear):
            logits = lm_head(drafted, batch.signal_buffers)[0]
        else:
            assert isinstance(lm_head, Linear)
            (only,) = drafted
            logits = lm_head(only)
        logits = ops.cast(logits, DType.float32)
        if self.logits_scaling != 1.0:
            logits = logits / self.logits_scaling
        cap = self.final_logit_softcapping
        if cap is not None:
            logits = ops.tanh(logits / cap) * cap
        return sampler.sample_all(logits)
