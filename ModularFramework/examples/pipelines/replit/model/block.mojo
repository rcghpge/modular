# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The core Transformer block of the model."""

from collections import Optional

from max.graph import ops, Symbol, Graph
from pathlib import Path
from pipelines.weights.gguf import GGUFFile
from pipelines.weights.loadable_model import LoadableModel

from pipelines.nn import Linear
from ..model.norm import LPLayerNorm
from ..weights.hyperparams import HyperParams


@value
struct MPTMLP:
    """Multiplayer perceptron used in MPT."""

    var up_proj: Linear
    var down_proj: Linear

    def __call__(self, input: Symbol) -> Symbol:
        g = input.graph()
        with g.layer("MPTMLP"):
            return self.down_proj(ops.gelu(self.up_proj(input)))
