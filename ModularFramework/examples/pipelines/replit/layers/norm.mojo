# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from max.graph import Dim, TensorType, Symbol
from tensor import Tensor, TensorShape
from weights.hyperparams import HyperParams


@value
struct LPLayerNorm:
    """Low Precision Layer Normalization."""

    alias eps: Float32 = 1e-05
    var weight: Symbol
    var hyperparams: HyperParams

    fn __call__(self, input: Symbol) raises -> Symbol:
        var g = input.graph()
        var beta = g.constant(
            Tensor[DType.float32](TensorShape(self.hyperparams.d_model), 0)
        )
        var eps = g.constant(Tensor[DType.float32](TensorShape(1), self.eps))
        var out = g.op(
            "mo.layer_norm",
            List[Symbol](input, self.weight, beta, eps),
            TensorType(DType.float32, input.tensor_type().dims),
        )
        return out
