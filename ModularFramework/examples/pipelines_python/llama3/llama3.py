# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import gguf
import numpy as np
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import Graph, TensorType
from max.graph.utils.load_gguf import Weights
from max.graph.weight import Weight

from utils import gguf_utils, tokenizer_from_gguf

from .attention import Attention
from .config import Llama3Config
from .embedding import Embedding
from .kv_cache import KVCache
from .mlp import MLP, Linear
from .norm import RMSNorm
from .rotary_embedding import RotaryEmbedding
from .transformer import Transformer, TransformerBlock


@dataclass
class Llama3Context:
    """The context for text generation using a Llama 3 model."""

    next_token: np.ndarray
    prompt_size: int
    max_tokens: int
    prompt: str


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


def rms_norm(weights: Weights):
    return RMSNorm(weights.weight(Graph.current).value)


def embedding(weights: Weights):
    weight: Weight = weights.weight(Graph.current)
    return Embedding(weight.value, weight.quantization_encoding)


def attention(config: Llama3Config, weights: Weights, rope: RotaryEmbedding):
    return Attention(
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        dim=config.hidden_dim,
        wk=linear(weights.attn_k),
        wv=linear(weights.attn_v),
        wq=linear(weights.attn_q),
        wo=linear(weights.attn_output),
        rope=rope,
    )


def _llama_graph(config: Llama3Config, gguf_reader: gguf.GGUFReader) -> Graph:
    cache_type = TensorType(
        DType.float32,
        shape=[
            "start_pos",
            config.n_layers,
            config.batch_size,
            config.n_kv_heads,
            config.head_dim,
        ],
    )
    tokens_type = TensorType(DType.int64, shape=[config.batch_size, "seq_len"])

    assert config.weight_path
    weights = Weights(gguf.GGUFReader(config.weight_path))

    with Graph(
        "llama3", input_types=[tokens_type, cache_type, cache_type]
    ) as graph:
        rope = RotaryEmbedding(
            dim=config.hidden_dim,
            n_heads=config.n_heads,
            theta=config.rope_theta,
            max_seq_len=config.seq_len,
        )

        layers = [
            TransformerBlock(
                attention=attention(config, weights.blk[i], rope),
                mlp=feed_forward(weights.blk[i]),
                attention_norm=rms_norm(weights.blk[i].attn_norm),
                mlp_norm=rms_norm(weights.blk[i].ffn_norm),
            )
            for i in range(config.n_layers)
        ]

        rope_scaling = weights.rope_freqs.weight(Graph.current).value.T

        transformer = Transformer(
            dim=config.hidden_dim,
            n_heads=config.n_heads,
            layers=layers,
            norm=rms_norm(weights.output_norm),
            output=linear(weights.output),
            theta=config.rope_theta,
            embedding=embedding(weights.token_embd),
            rope_scaling=rope_scaling,
        )
        logits, k_update, v_update = transformer(*graph.inputs)
        # TODO (MSDK-813): This logic should be included here instead of with logits in next_token.
        # graph.output(logits[:, -1], k_update, v_update)
        graph.output(logits, k_update, v_update)
        return graph


class Llama3:
    """The overall interface to the Llama 3 model."""

    config: Llama3Config
    _model: Model
    _kv_cache: KVCache

    def __init__(self, config: Llama3Config):
        self.config = config

        assert config.weight_path is not None
        gguf_reader = gguf.GGUFReader(config.weight_path)
        _override_config(config, gguf_reader)

        self._model = self._load_model(config, gguf_reader)
        self._tokenizer = tokenizer_from_gguf(gguf_reader)

        # Work around for older Llama 1/2 GGUFs, where the vocab size may be -1.
        # See https://github.com/ggerganov/llama.cpp/pull/4258.
        if config.vocab_size < 0:
            config.vocab_size = self._tokenizer.vocab_size

        self._kv_cache = KVCache(
            self.config.seq_len,
            self.config.batch_size,
            self.config.n_layers,
            self.config.n_kv_heads,
            self.config.head_dim,
        )

    def _load_model(
        self, config: Llama3Config, reader: gguf.GGUFReader
    ) -> Model:
        session = InferenceSession()
        if serialized_path := config.serialized_model_path:
            print("Loading serialized model from", serialized_path, "...")
            return session.load(serialized_path)
        else:
            graph = _llama_graph(config, reader)
            print("Compiling...")
            return session.load(graph)

    async def new_context(self, prompt: str) -> Llama3Context:
        encoded_prompt = self._tokenizer.encode(prompt)
        prompt_size = len(encoded_prompt)
        return Llama3Context(
            next_token=np.array(encoded_prompt).reshape(1, -1),
            prompt_size=prompt_size,
            max_tokens=_max_tokens_to_generate(prompt_size, self.config),
            prompt=prompt,
        )

    async def next_token(
        self, contexts: dict[str, Llama3Context]
    ) -> dict[str, str]:
        # Note: assuming a single request.
        request_id, context = next(iter(contexts.items()))

        cache = self._kv_cache
        input_names = [t.name for t in self._model.input_metadata]
        output_names = [t.name for t in self._model.output_metadata]
        inputs = [context.next_token, cache.keys_view(), cache.values_view()]

        result = self._model.execute(**dict(zip(input_names, inputs)))
        logits, k_cache, v_cache = (result[o] for o in output_names)
        self._kv_cache.update(k_cache, v_cache)

        # TODO (MSDK-813): This slicing should move into graph output above when working.
        # TODO: Add a weighted sampler here.
        # Get argmax of the logits of the last token.
        next_token = logits[:, -1].argmax(axis=-1)[-1]
        context.next_token = next_token.reshape(1, -1)
        print(f"\nDEBUG: {next_token=}")
        decoded_token = self._tokenizer.decode(next_token)
        if decoded_token == self._tokenizer.eos_token:
            return {}
        return {request_id: decoded_token}


def _max_tokens_to_generate(prompt_size: int, config: Llama3Config) -> int:
    """Returns the max number of tokens to generate (including the prompt)."""
    if config.max_new_tokens < 0:
        return config.max_length
    return min(config.max_new_tokens + prompt_size, config.max_length)


def _override_config(config: Llama3Config, reader: gguf.GGUFReader):
    """Overrides parameters from config if they're in the GGUF checkpoint."""

    def _maybe_override(attribute, gguf_name):
        value = gguf_utils.read_number(reader, gguf_name)
        if value is not None:
            setattr(config, attribute, value)

    _maybe_override("seq_len", "llama.context_length")
    _maybe_override("n_layers", "llama.block_count")
    _maybe_override("n_heads", "llama.attention.head_count")
    _maybe_override("n_kv_heads", "llama.attention.head_count_kv")
    _maybe_override("vocab_size", "llama.vocab_size")
    _maybe_override("hidden_dim", "llama.embedding_length")
    _maybe_override("rope_theta", "llama.rope.freq_base")
    _maybe_override(
        "layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon"
    )
