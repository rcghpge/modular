#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pickle

import numpy as np
from max.pipelines.core import (
    TextContext,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)

msgpack_encoder = msgpack_numpy_encoder()
msgpack_decoder = msgpack_numpy_decoder(TextContext)


def test_text_context_pickle_serialization(benchmark):
    context = TextContext(
        prompt="sampling params test prompt",
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    benchmark(pickle.dumps, context)


def test_text_context_pickle_deserialization(benchmark):
    context = TextContext(
        prompt="sampling params test prompt",
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    benchmark(pickle.loads, pickle.dumps(context))


def test_text_context_msgpack_serialization(benchmark):
    context = TextContext(
        prompt="sampling params test prompt",
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    benchmark(msgpack_encoder, context)


def test_text_context_msgpack_deserialization(benchmark):
    context = TextContext(
        prompt="sampling params test prompt",
        max_length=50,
        tokens=np.array([0, 1, 2, 3, 4]),
    )

    benchmark(
        msgpack_decoder,
        msgpack_encoder(context),
    )


def test_long_text_context_pickle_serialization(benchmark):
    # Create a longer prompt with more than 240 tokens
    long_prompt = (
        "This is a much longer test prompt that contains more than 240 tokens. "
        * 50
    )
    # Create a longer token array
    long_tokens = np.array(range(300))  # 300 tokens

    context = TextContext(
        prompt=long_prompt,
        max_length=500,  # Increased max length
        tokens=long_tokens,
    )

    benchmark(pickle.dumps, context)


def test_long_text_context_pickle_deserialization(benchmark):
    long_prompt = (
        "This is a much longer test prompt that contains more than 240 tokens. "
        * 50
    )
    long_tokens = np.array(range(300))

    context = TextContext(
        prompt=long_prompt,
        max_length=500,
        tokens=long_tokens,
    )

    benchmark(pickle.loads, pickle.dumps(context))


def test_long_text_context_msgpack_serialization(benchmark):
    long_prompt = (
        "This is a much longer test prompt that contains more than 240 tokens. "
        * 50
    )
    long_tokens = np.array(range(300))

    context = TextContext(
        prompt=long_prompt,
        max_length=500,
        tokens=long_tokens,
    )

    benchmark(msgpack_encoder, context)


def test_long_text_context_msgpack_deserialization(benchmark):
    long_prompt = (
        "This is a much longer test prompt that contains more than 240 tokens. "
        * 50
    )
    long_tokens = np.array(range(300))

    context = TextContext(
        prompt=long_prompt,
        max_length=500,
        tokens=long_tokens,
    )

    benchmark(
        msgpack_decoder,
        msgpack_encoder(context),
    )


def test_very_long_text_context_pickle_serialization(benchmark):
    # Create a very long prompt with more than 10k tokens
    very_long_prompt = (
        "This is a very long test prompt that contains more than 10k tokens. "
        * 1000
    )
    # Create a very long token array
    very_long_tokens = np.array(range(12000))  # 12k tokens

    context = TextContext(
        prompt=very_long_prompt,
        max_length=15000,  # Increased max length for very long context
        tokens=very_long_tokens,
    )

    benchmark(pickle.dumps, context)


def test_very_long_text_context_pickle_deserialization(benchmark):
    very_long_prompt = (
        "This is a very long test prompt that contains more than 10k tokens. "
        * 1000
    )
    very_long_tokens = np.array(range(12000))

    context = TextContext(
        prompt=very_long_prompt,
        max_length=15000,
        tokens=very_long_tokens,
    )

    benchmark(pickle.loads, pickle.dumps(context))


def test_very_long_text_context_msgpack_serialization(benchmark):
    very_long_prompt = (
        "This is a very long test prompt that contains more than 10k tokens. "
        * 1000
    )
    very_long_tokens = np.array(range(12000))

    context = TextContext(
        prompt=very_long_prompt,
        max_length=15000,
        tokens=very_long_tokens,
    )

    benchmark(msgpack_encoder, context)


def test_very_long_text_context_msgpack_deserialization(benchmark):
    very_long_prompt = (
        "This is a very long test prompt that contains more than 10k tokens. "
        * 1000
    )
    very_long_tokens = np.array(range(12000))

    context = TextContext(
        prompt=very_long_prompt,
        max_length=15000,
        tokens=very_long_tokens,
    )

    benchmark(
        msgpack_decoder,
        msgpack_encoder(context),
    )
