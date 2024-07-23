# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from collections import Optional, List
from pathlib import Path

from max.graph import ops, Dim, TensorType, Symbol


@value
struct Embedding:
    """A layer that converts tokens into dense vectors."""

    var weights: Symbol

    def __call__(self, input: Symbol) -> Symbol:
        g = input.graph()
        with g.layer("Embedding"):
            return ops.gather(self.weights, input, axis=0)


@value
struct SharedEmbedding:
    """An embedding layer that can both embed and unembed inputs."""

    var weights: Symbol

    def __call__(self, input: Symbol, unembed: Bool = False) -> Symbol:
        g = input.graph()
        with g.layer("SharedEmbedding"):
            if unembed:
                return input @ ops.transpose_matrix(self.weights)
            return ops.gather(self.weights, input, axis=0)
