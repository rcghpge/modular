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
"""Picks the proposer a unified spec-decode graph drafts through.

The speculative method picks the proposer, as vLLM's model runner picks its
drafter and SGLang's ``SpeculativeAlgorithm.create_worker`` its worker. A
proposer calls its draft module by the method's convention, so one proposer
serves every draft checkpoint its method runs, whichever architecture built
the draft.
"""

from __future__ import annotations

from typing import Any

from max.graph import TensorValue
from max.nn.layer import Module
from max.nn.transformer import ReturnHiddenStates

from .block_driver import BlockProposer
from .config import SpeculativeConfig, SpeculativeMethod
from .dflash import DFlashDraft, DFlashProposer
from .driver import SequentialProposer
from .dspark import (
    DSparkDraft,
    DSparkProposer,
    DSparkTargetHeadDraft,
    DSparkTargetHeadProposer,
)
from .eagle import EagleProposer

__all__ = [
    "BLOCK_METHODS",
    "SEQUENTIAL_METHODS",
    "block_proposer",
    "capture_layers",
    "sequential_proposer",
    "target_hidden_states",
]

SEQUENTIAL_METHODS: frozenset[SpeculativeMethod] = frozenset({"eagle"})
"""The one-token-per-step methods :func:`sequential_proposer` runs."""

BLOCK_METHODS: frozenset[SpeculativeMethod] = frozenset({"dflash", "dspark"})
"""The block-drafting methods :func:`block_proposer` runs."""


def target_hidden_states(speculative: SpeculativeConfig) -> ReturnHiddenStates:
    """Returns the target hidden states the configured method's draft reads.

    A block draft reads the outputs of the target layers it names. A
    one-token-per-step draft reads the target's last normalized layer.
    """
    if speculative.is_block_draft():
        return ReturnHiddenStates.SELECTED_LAYERS
    return ReturnHiddenStates.ALL_NORMALIZED


def capture_layers(
    speculative: SpeculativeConfig, draft_config: Any
) -> tuple[int, ...]:
    """Returns the target layers a block draft reads, as MAX indices.

    Every block draft's config names them in ``target_layer_ids``.
    """
    if not speculative.is_block_draft():
        return ()
    return tuple(draft_config.target_layer_ids)


def sequential_proposer(
    speculative: SpeculativeConfig,
    draft: Module,
    draft_config: Any,
    target_huggingface_config: Any,
) -> SequentialProposer[TensorValue]:
    """Returns the proposer for a one-token-per-step method.

    Raises:
        ValueError: If no proposer runs the configured method.
    """
    del target_huggingface_config
    if speculative.is_eagle():
        return EagleProposer(draft, draft_config.hidden_size)
    raise ValueError(
        f"No proposer runs {speculative.speculative_method!r} one token at a"
        " time."
    )


def block_proposer(
    speculative: SpeculativeConfig,
    draft: Module,
    draft_config: Any,
    target_huggingface_config: Any,
) -> BlockProposer[list[TensorValue], Any]:
    """Returns the proposer for a block-drafting method.

    The draft's config names the mask token and the draft's width. A draft
    that projects through its target's head applies the target's scales.

    Raises:
        ValueError: If no proposer runs the configured method.
    """
    text_config = target_huggingface_config.get_text_config()
    if speculative.speculative_method == "dflash":
        assert isinstance(draft, DFlashDraft)
        return DFlashProposer(
            draft,
            # The anchor slot plus the drafted tokens.
            block_size=speculative.draft_width + 1,
            mask_token_id=draft_config.mask_token_id,
            hidden_size=draft_config.hidden_size,
            embedding_multiplier=getattr(
                text_config, "embedding_multiplier", 1.0
            ),
            logits_scaling=getattr(text_config, "logits_scaling", 1.0),
            final_logit_softcapping=getattr(
                text_config, "final_logit_softcapping", None
            ),
        )
    if speculative.is_dspark():
        # A DSpark draft with its own head drops the anchor slot; one that
        # holds its target's drafts at every slot.
        if getattr(draft, "has_own_lm_head", False):
            assert isinstance(draft, DSparkDraft)
            return DSparkProposer(
                draft,
                block_size=speculative.draft_width + 1,
                mask_token_id=draft_config.mask_token_id,
                hidden_size=draft_config.hidden_size,
            )
        assert isinstance(draft, DSparkTargetHeadDraft)
        return DSparkTargetHeadProposer(
            draft,
            block_size=speculative.draft_width,
            mask_token_id=draft_config.mask_token_id,
            hidden_size=draft_config.hidden_size,
            final_logit_softcapping=draft_config.final_logit_softcapping,
        )
    raise ValueError(
        f"No proposer runs {speculative.speculative_method!r} a block at a"
        " time."
    )
