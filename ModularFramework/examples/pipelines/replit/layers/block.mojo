# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional

from max.graph import ops, Symbol, Graph

from layers.attention import GroupedQueryAttention
from layers.linear import Linear
from layers.norm import LPLayerNorm
from weights.replit_checkpoint import Checkpoint
from weights.hyperparams import HyperParams


@value
struct MPTMLP:
    """Multiplayer perceptron used in MPT."""

    var up_proj: Linear
    var down_proj: Linear

    fn __call__(self, input: Symbol) raises -> Symbol:
        return self.down_proj(ops.gelu(self.up_proj(input)))


struct MPTBlock[T: Checkpoint, weights_type: DType]:
    """A block in the MosaicML Pretrained Transformer model."""

    var norm_1: LPLayerNorm
    var attn: GroupedQueryAttention
    var norm_2: LPLayerNorm
    var ffn: MPTMLP

    def __init__(
        inout self,
        norm_1: LPLayerNorm,
        attn: GroupedQueryAttention,
        norm_2: LPLayerNorm,
        ffn: MPTMLP,
    ):
        self.norm_1 = norm_1
        self.attn = attn
        self.norm_2 = norm_2
        self.ffn = ffn

    fn __call__(
        self,
        input: Symbol,
        attn_bias: Optional[Symbol] = None,
        k_cache: Optional[Symbol] = None,
        v_cache: Optional[Symbol] = None,
    ) raises -> (Symbol, Symbol, Symbol):
        var a = self.norm_1(input)
        var k_cache_update: Symbol
        var v_cache_update: Symbol
        var b: Symbol
        b, k_cache_update, v_cache_update = self.attn(
            a, attn_bias, k_cache, v_cache
        )
        # Rebind the attention output to the shape of the input to allow the
        # `add` op to correctly set shapes.
        b = ops.rebind(b, input.tensor_type().dims)
        var output = input + b
        var m = self.norm_2(output)
        var n = self.ffn(m)
        return output + n, k_cache_update, v_cache_update

    @staticmethod
    fn create(
        params: T, prefix: String, g: Graph, hyperparams: HyperParams
    ) raises -> MPTBlock[T, weights_type]:
        """Build a MPT Block from the given params and string prefix.

        Args:
            params: Checkpoint class for getting the weights.
            prefix: String prefix to add to the weight key.
            g: Graph in which to create the weights.
            hyperparams: Model hyperparameters.
        Returns:
            A new MPTBlock object.
        """

        @parameter
        fn load_param(name: String) raises -> Symbol:
            return ops.cast(
                g.constant(params.get[weights_type](prefix + name)),
                DType.float32,
            )

        var norm_1 = LPLayerNorm(load_param("norm_1.weight"), hyperparams)
        var attn = GroupedQueryAttention(
            hyperparams,
            Linear(load_param("attn.Wqkv.weight")),
            Linear(load_param("attn.out_proj.weight")),
        )
        var norm_2 = LPLayerNorm(load_param("norm_2.weight"), hyperparams)
        var ffn = MPTMLP(
            Linear(load_param("ffn.up_proj.weight")),
            Linear(load_param("ffn.down_proj.weight")),
        )
        return MPTBlock[T, weights_type](norm_1, attn, norm_2, ffn)
