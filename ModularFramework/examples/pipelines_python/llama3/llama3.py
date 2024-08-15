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
from max.engine import InferenceSession, Model
from max.graph import DType, Graph, TensorType
from max.graph.utils import load_gguf  # TODO Use this

from utils import gguf_utils, tokenizer_from_gguf

from .config import Llama3Config
from .embedding import Embedding
from .kv_cache import KVCache


@dataclass
class Llama3Context:
    """The context for text generation using a Llama 3 model."""

    next_token: np.ndarray
    prompt_size: int
    max_tokens: int
    prompt: str


class Llama3:
    """The overall interface to the Llama 3 model."""

    config: Llama3Config
    _session: InferenceSession
    _compiled_model: Model
    _kv_cache: KVCache

    def __init__(self, config: Llama3Config):
        self.config = config
        assert self.config.weight_path is not None
        gguf_reader = gguf.GGUFReader(self.config.weight_path)
        _override_config(self.config, gguf_reader)
        self._tokenizer = tokenizer_from_gguf(gguf_reader)

        # Work around for older Llama 1/2 GGUFs, where the vocab size may be -1.
        # See https://github.com/ggerganov/llama.cpp/pull/4258.
        if self.config.vocab_size < 0:
            self.config.vocab_size = self._tokenizer.vocab_size

        self._session = InferenceSession()
        if self.config.serialized_model_path:
            self._load_graph(self.config.serialized_model_path)
        else:
            self._build_graph("llama3", gguf_reader)

    def _build_graph(self, name: str, gguf_reader: gguf.GGUFReader):
        self._session = InferenceSession()
        cache_type = TensorType(
            DType.float32,
            shape=[
                "prev_seq_len",
                self.config.n_layers,
                self.config.batch_size,
                self.config.n_kv_heads,
                self.config.head_dim,
            ],
        )
        tokens_type = TensorType(
            DType.int64, shape=[self.config.batch_size, "seq_len"]
        )

        print("Compiling...")
        with Graph(
            name, input_types=[tokens_type, cache_type, cache_type]
        ) as graph:
            embedding = Embedding(
                np.random.uniform(
                    size=(self.config.vocab_size, self.config.hidden_dim)
                )
            )
            out = embedding(graph.inputs[0])
            graph.output(out, graph.inputs[1], graph.inputs[2])
            self._compiled_model = self._session.load(graph)

    def _load_graph(self, file: Union[str, Path]):
        print("Loading serialized model from", file, "...")
        self._compiled_model = self._session.load(file)

    async def new_context(self, prompt: str) -> Llama3Context:
        encoded_prompt = self._tokenizer.encode(prompt)
        prompt_size = len(encoded_prompt)
        max_tokens = _get_max_tokens_to_generate(
            prompt_size, self.config.max_length, self.config.max_new_tokens
        )
        self._kv_cache = KVCache(
            max_tokens,
            self.config.batch_size,
            self.config.n_layers,
            self.config.n_kv_heads,
            self.config.head_dim,
        )
        return Llama3Context(
            np.array(encoded_prompt).reshape(1, -1),
            prompt_size,
            max_tokens,
            prompt,
        )

    async def next_token(
        self, contexts: dict[str, Llama3Context]
    ) -> dict[str, str]:
        # Note: assuming a single request.
        request_id, context = next(iter(contexts.items()))

        tokens = context.next_token
        input_names = [t.name for t in self._compiled_model.input_metadata]
        result = self._compiled_model.execute(
            **dict(
                zip(
                    input_names,
                    [
                        tokens,
                        self._kv_cache.keys_view(),
                        self._kv_cache.values_view(),
                    ],
                )
            )
        )
        output_names = [t.name for t in self._compiled_model.output_metadata]
        logits = result[output_names[0]]
        self._kv_cache.update(result[output_names[1]], result[output_names[2]])

        # TODO: Add a weighted sampler here.
        # Get argmax of the logits of the last token.
        next_token = logits.argmax(axis=-1)[-1]
        context.next_token = next_token.reshape(1, -1)
        print("\nnext", next_token)
        decoded_token = self._tokenizer.decode(next_token)

        if decoded_token == self._tokenizer.eos_token:
            return {}
        return {request_id: decoded_token}


def _get_max_tokens_to_generate(
    prompt_len: int, max_length: int, max_new_tokens: int
) -> int:
    """Returns the max number of tokens to generate (including the prompt)."""
    if max_new_tokens < 0:
        return max_length

    return min(max_new_tokens + prompt_len, max_length)


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
