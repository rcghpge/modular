#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import multiprocessing

import pytest
import zmq
from max.pipelines.core import (
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
)
from max.serve.process_control import ProcessControl
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
    TokenGenerationSchedulerConfig,
)


@pytest.fixture
def zmq_ctx():
    return zmq.Context(io_threads=2)


def test_stream_responses_to_frontend(benchmark, zmq_ctx):
    """Benchmark stream_responses_to_frontend performance."""
    batch_size_to_test = 512

    # Create a minimal scheduler instance for testing
    ctx = multiprocessing.get_context()
    # the values below are arbitrary, we are not using them in this test
    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=1,
        target_tokens_per_batch_tg=None,
        max_batch_size_ce=4,
        max_forward_steps_ce=1,
        target_tokens_per_batch_ce=1,
    )

    scheduler = TokenGenerationScheduler(
        process_control=ProcessControl(ctx=ctx, name="test"),
        scheduler_config=config,
        pipeline=None,  # type: ignore
    )

    # Create test data - simulate multiple responses with tokens
    batch_responses = {}
    for i in range(batch_size_to_test):
        tokens = [TextResponse(next_token=i) for i in range(10)]
        batch_responses[f"req{i}"] = TextGenerationResponse(
            tokens=tokens, final_status=TextGenerationStatus.ACTIVE
        )

    # Run the benchmark
    result = benchmark.pedantic(
        scheduler._stream_responses_to_frontend,
        args=(batch_responses,),
        warmup_rounds=1,
        rounds=3,
        iterations=10,
    )
