# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Build a Llama3 model via Graph API from GGUF weights."""

from max.graph import Graph, Weight
from max.graph.utils.load_gguf import Weights

from .model.attention import Attention
from .model.embedding import Embedding
from .model.hyperparameters import Hyperparameters
from .model.mlp import Linear, MLP
from .model.norm import RMSNorm
from .model.rotary_embedding import RotaryEmbedding
from .model.transformer import Transformer, TransformerBlock


def feed_forward(weights: Weights):
    return MLP(
        linear(weights.ffn_gate),
        linear(weights.ffn_down),
        linear(weights.ffn_up),
    )


def linear(weights: Weights) -> Linear:
    weight: Weight = weights.weight(Graph.current)
    if weight.quantization_encoding is None:
        return Linear(weight.value.T)
    else:
        return Linear(weight.value, weight.quantization_encoding)


def rms_norm(weights: Weights, eps: float):
    return RMSNorm(weights.weight(Graph.current).value, eps)


def embedding(weights: Weights):
    weight: Weight = weights.weight(Graph.current)
    return Embedding(weight.value, weight.quantization_encoding)


def attention(params: Hyperparameters, weights: Weights, rope: RotaryEmbedding):
    return Attention(
        n_heads=params.n_heads,
        n_kv_heads=params.n_kv_heads,
        head_dim=params.head_dim,
        dim=params.hidden_dim,
        wk=linear(weights.attn_k),
        wv=linear(weights.attn_v),
        wq=linear(weights.attn_q),
        wo=linear(weights.attn_output),
        rope=rope,
    )


def transformer(graph: Graph, params: Hyperparameters, weights: Weights):
    with graph:
        rope = RotaryEmbedding(
            dim=params.hidden_dim,
            n_heads=params.n_heads,
            theta=params.rope_theta,
            max_seq_len=params.seq_len,
        )

        layers = [
            TransformerBlock(
                attention=attention(params, weights.blk[i], rope),
                mlp=feed_forward(weights.blk[i]),
                attention_norm=rms_norm(
                    weights.blk[i].attn_norm, params.layer_norm_rms_epsilon
                ),
                mlp_norm=rms_norm(
                    weights.blk[i].ffn_norm, params.layer_norm_rms_epsilon
                ),
            )
            for i in range(params.n_layers)
        ]

        try:
            rope_scaling = weights.rope_freqs.weight(graph).value.T
        except:
            rope_scaling = 1.0

        return Transformer(
            dim=params.hidden_dim,
            n_heads=params.n_heads,
            layers=layers,
            norm=rms_norm(weights.output_norm, params.layer_norm_rms_epsilon),
            output=linear(weights.output),
            theta=params.rope_theta,
            embedding=embedding(weights.token_embd),
            rope_scaling=rope_scaling,
        )
