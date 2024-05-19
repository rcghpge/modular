# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.graph import ops, TensorType, Symbol
from max.tensor import Tensor, TensorShape

from weights.hyperparams import HyperParams


@value
struct LPLayerNorm:
    """Low Precision Layer Normalization."""

    alias eps: Float32 = 1e-05
    var weight: Symbol
    var hyperparams: HyperParams

    def __call__(self, input: Symbol) -> Symbol:
        g = input.graph()
        beta = g.constant(
            Tensor[DType.float32](TensorShape(self.hyperparams.d_model), 0)
        )
        out = ops.layer_norm(input, self.weight, beta, self.eps)
        return out
