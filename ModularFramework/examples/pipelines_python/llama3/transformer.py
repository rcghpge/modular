# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
import numpy as np
from max.graph import GraphValue, ops
from .attention import Attention
from .mlp import MLP, Linear
from .norm import RMSNorm
from typing import List


@dataclass
class TransformerBlock:
    attention: Attention
    mlp: MLP
    attention_norm: RMSNorm
    mlp_norm: RMSNorm

    def __call__(self, x, k_cache, v_cache):
        attention_out, k_cache_update, v_cache_update = self.attention(
            self.attention_norm(input), k_cache, v_cache
        )
        h = x + attention_out
        h = h + self.mlp(self.mlp_norm(h))
        return h, k_cache_update, v_cache_update
