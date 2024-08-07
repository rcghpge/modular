# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass
from max.engine import InferenceSession, Model
from max.graph import DType, Graph, TensorType, ops
from pathlib import Path
from typing import Generic, Protocol, TypeAlias, TypeVar
import numpy as np

from .config import Llama3Config
from .kv_cache import KVCache

Context = TypeVar("Context")


# TODO: Remove this re-implementation when max.serve is in monorepo.
class TokenGenerator(Generic[Context], Protocol):
    """Interface for LLM token-generator models."""

    async def new_context(self, prompt: str) -> Context:
        ...

    async def next_token(self, contexts: dict[str, Context]) -> dict[str, str]:
        ...


def get_max_tokens_to_generate(
    prompt_len: int, max_length: int, max_new_tokens: int
) -> int:
    """Returns the max number of tokens to generate (including the prompt)."""
    if max_new_tokens < 0:
        return max_length

    return min(max_new_tokens + prompt_len, max_length)


@dataclass
class Llama3Context:
    """The context for text generation using a Llama 3 model."""

    prompt_size: int
    max_tokens: int


class Llama3:
    """The overall interface to the Llama 3 model."""

    config: Llama3Config
    _session: InferenceSession
    _compiled_model: Model
    _kv_cache: KVCache

    def __init__(self, config: Llama3Config):
        self.config = config
        print(self.config)
        # TODO: Load hyperparameters from the GGUF weights.
        self._build_graph("llama3")

    def _build_graph(self, name: str):
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

        with Graph(
            name, input_types=[tokens_type, cache_type, cache_type]
        ) as graph:
            graph.output(
                ops.constant(np.array([0.0, 0.5, 1.0, 1.5])),
                graph.inputs[1],
                graph.inputs[2],
            )
            self._compiled_model = self._session.load(graph)

    async def new_context(self, prompt: str) -> Llama3Context:
        # TODO: Encode prompt
        prompt_size = len(prompt)
        max_tokens = get_max_tokens_to_generate(
            prompt_size, self.config.max_length, self.config.max_new_tokens
        )
        self._kv_cache = KVCache(
            max_tokens,
            self.config.batch_size,
            self.config.n_layers,
            self.config.n_kv_heads,
            self.config.head_dim,
        )
        # TODO: Get prompt tensor
        print("Initializing context for Llama 3...")
        return Llama3Context(prompt_size, max_tokens)

    async def next_token(
        self, contexts: dict[str, Llama3Context]
    ) -> dict[str, str]:
        # Note: assuming a single request.
        request_id, context = next(iter(contexts.items()))

        # TODO: Provide previous tokens to compiled model

        tokens = np.zeros(shape=[1, 1]).astype(np.int64)
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
        next_token = logits.argmax()
        tokens = next_token

        # TODO: Decode token using tokenizer.
        decoded_token = str(next_token)

        # TODO: Check against proper end-of-text token here.
        if decoded_token == "end":
            return {}
        return {request_id: decoded_token}
