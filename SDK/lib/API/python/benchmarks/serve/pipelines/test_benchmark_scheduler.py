#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Any

from max.interfaces import GenerationStatus, TextGenerationOutput
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
    TokenGenerationSchedulerConfig,
)


def test_stream_responses_to_frontend(benchmark: Any) -> None:
    """Benchmark stream_responses_to_frontend performance."""
    batch_size_to_test = 512

    # Create a minimal scheduler instance for testing
    # the values below are arbitrary, we are not using them in this test
    config = TokenGenerationSchedulerConfig(
        max_batch_size_tg=4,
        max_forward_steps_tg=1,
        max_batch_size_ce=4,
        target_tokens_per_batch_ce=1,
    )

    scheduler = TokenGenerationScheduler(
        scheduler_config=config,
        pipeline=None,  # type: ignore
        request_queue=None,  # type: ignore
        response_queue=None,  # type: ignore
        cancel_queue=None,  # type: ignore
    )

    # Create test data - simulate multiple responses with tokens
    batch_responses = {}
    for i in range(batch_size_to_test):
        tokens = [i for i in range(10)]
        batch_responses[f"req{i}"] = TextGenerationOutput(
            request_id=f"req{i}",
            tokens=tokens,
            final_status=GenerationStatus.ACTIVE,
            log_probabilities=None,
        )

    # Run the benchmark
    _ = benchmark.pedantic(
        scheduler._stream_responses_to_frontend,
        args=(batch_responses,),
        warmup_rounds=1,
        rounds=3,
        iterations=10,
    )
