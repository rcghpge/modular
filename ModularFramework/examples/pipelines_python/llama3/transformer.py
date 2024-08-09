# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from max.graph import ValueLike, GraphValue
from .attention import Attention
from .mlp import MLP
from .norm import RMSNorm


@dataclass
class TransformerBlock:
    """
    Stacks Attention, FeedForward, and RMSNorm layers
    into single transformer block.
    """

    attention: Attention
    mlp: MLP
    attention_norm: RMSNorm
    mlp_norm: RMSNorm

    def __call__(
        self, x: ValueLike, k_cache: ValueLike, v_cache: ValueLike
    ) -> Tuple[GraphValue, GraphValue, GraphValue]:
        attention_out, k_cache_update, v_cache_update = self.attention(
            self.attention_norm(x), k_cache, v_cache
        )
        h = x + attention_out
        h = h + self.mlp(self.mlp_norm(h))
        return h, k_cache_update, v_cache_update
