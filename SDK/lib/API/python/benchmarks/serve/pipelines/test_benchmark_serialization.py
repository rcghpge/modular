#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle

import numpy as np
from max.interfaces import msgpack_numpy_decoder, msgpack_numpy_encoder
from max.pipelines.core import TextContext

msgpack_encoder = msgpack_numpy_encoder()
msgpack_decoder = msgpack_numpy_decoder(TextContext)


def test_text_context_pickle_serialization(benchmark) -> None:  # noqa: ANN001
    context = TextContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    benchmark(pickle.dumps, context)


def test_text_context_pickle_deserialization(benchmark) -> None:  # noqa: ANN001
    context = TextContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    benchmark(pickle.loads, pickle.dumps(context))


def test_text_context_msgpack_serialization(benchmark) -> None:  # noqa: ANN001
    context = TextContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    benchmark(msgpack_encoder, context)


def test_text_context_msgpack_deserialization(benchmark) -> None:  # noqa: ANN001
    context = TextContext(
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    benchmark(msgpack_decoder, msgpack_encoder(context))


def test_long_text_context_pickle_serialization(benchmark) -> None:  # noqa: ANN001
    # Create a longer token array
    long_tokens = np.array(range(300))  # 300 tokens

    context = TextContext(
        max_length=500,  # Increased max length
        tokens=long_tokens,
    )

    benchmark(pickle.dumps, context)


def test_long_text_context_pickle_deserialization(benchmark) -> None:  # noqa: ANN001
    long_tokens = np.array(range(300))

    context = TextContext(
        max_length=500,
        tokens=long_tokens,
    )

    benchmark(pickle.loads, pickle.dumps(context))


def test_long_text_context_msgpack_serialization(benchmark) -> None:  # noqa: ANN001
    long_tokens = np.array(range(300))

    context = TextContext(
        max_length=500,
        tokens=long_tokens,
    )

    benchmark(msgpack_encoder, context)


def test_long_text_context_msgpack_deserialization(benchmark) -> None:  # noqa: ANN001
    long_tokens = np.array(range(300))

    context = TextContext(
        max_length=500,
        tokens=long_tokens,
    )

    benchmark(msgpack_decoder, msgpack_encoder(context))


def test_very_long_text_context_pickle_serialization(benchmark) -> None:  # noqa: ANN001
    # Create a very long token array
    very_long_tokens = np.array(range(12000))  # 12k tokens

    context = TextContext(
        max_length=15000,  # Increased max length for very long context
        tokens=very_long_tokens,
    )

    benchmark(pickle.dumps, context)


def test_very_long_text_context_pickle_deserialization(benchmark) -> None:  # noqa: ANN001
    very_long_tokens = np.array(range(12000))

    context = TextContext(
        max_length=15000,
        tokens=very_long_tokens,
    )

    benchmark(pickle.loads, pickle.dumps(context))


def test_very_long_text_context_msgpack_serialization(benchmark) -> None:  # noqa: ANN001
    very_long_tokens = np.array(range(12000))

    context = TextContext(
        max_length=15000,
        tokens=very_long_tokens,
    )

    benchmark(msgpack_encoder, context)


def test_very_long_text_context_msgpack_deserialization(benchmark) -> None:  # noqa: ANN001
    very_long_tokens = np.array(range(12000))

    context = TextContext(
        max_length=15000,
        tokens=very_long_tokens,
    )

    benchmark(msgpack_decoder, msgpack_encoder(context))
