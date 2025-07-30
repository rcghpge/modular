# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import AsyncGenerator, Coroutine
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Generic, Optional, TypeVar

import numpy as np
from max.interfaces import (
    AudioGenerationMetadata,
    AudioGenerationRequest,
    AudioGeneratorOutput,
    PipelineTokenizer,
    TextGenerationRequest,
)
from max.profiler import Tracer
from max.serve.pipelines.stop_detection import StopDetector
from max.serve.scheduler.queues import EngineQueue
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import StopWatch, record_ms

logger = logging.getLogger("max.serve")


@dataclass(frozen=True)
class TokenGeneratorOutput:
    decoded_token: str
    token_log_probabilities: Optional[list[float]] = None
    top_log_probabilities: Optional[list[dict[str, float]]] = None
    prompt_token_count: Optional[int] = None
    stop_sequence: Optional[str] = None


@dataclass(frozen=True)
class EmbeddingsGeneratorOutput:
    embeddings: np.ndarray


@dataclass
class TokenGeneratorStats:
    token_gen_batch_size: int = 0
    token_gen_batch_calls: int = 0


TokenGeneratorContext = TypeVar("TokenGeneratorContext")


class TokenGeneratorPipeline(Generic[TokenGeneratorContext]):
    """Base class for LLM text generation pipelines."""

    def __init__(
        self,
        model_name: str,
        tokenizer: PipelineTokenizer,
        engine_queue: EngineQueue,
    ) -> None:
        self.logger = logging.getLogger(
            "max.serve.pipelines.TokenGeneratorPipeline"
        )
        # This logger is too verbose to expose to end users. Disable propagation to the root logger by default.
        self.logger.info("%s: Constructed", model_name)
        self.debug_logging = self.logger.isEnabledFor(logging.DEBUG)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.engine_queue = engine_queue
        self.stats = TokenGeneratorStats()

        self._background_tasks: set[asyncio.Task] = set()

    async def _collect_log_probs(self, log_prob, context, skip_special_tokens):  # noqa: ANN001
        token_log_probabilities = log_prob.token_log_probabilities
        top_log_probabilities = []
        for top_log_probs in log_prob.top_log_probabilities:
            decoded_log_probs = {}
            for token_id, value in top_log_probs.items():
                decoded_log_probs[
                    await self.tokenizer.decode(
                        token_id,
                        skip_special_tokens=skip_special_tokens,
                    )
                ] = value
            top_log_probabilities.append(decoded_log_probs)

        return (token_log_probabilities, top_log_probabilities)

    async def next_token(
        self, request: TextGenerationRequest
    ) -> AsyncGenerator[TokenGeneratorOutput, None]:
        """Generates and streams tokens for the provided request."""
        itl = StopWatch()
        total_sw = StopWatch()
        self.logger.debug(
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.request_id,
            request.index,
            total_sw.elapsed_ms,
        )

        # Skip special tokens if tool use is enabled
        tool_use = request.tools is not None
        skip_special_tokens = tool_use

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            # TODO(AITLIB-319): Remove hashattr check
            if hasattr(context, "active_length"):
                METRICS.input_tokens(context.active_length)

            with record_ms(METRICS.output_time):
                # stop detector is stateful, so new it up here for
                # use in the response stream
                stop_detector = StopDetector(stop=request.sampling_params.stop)

                async for response in self.engine_queue.stream(
                    request.request_id, context
                ):
                    for i, token in enumerate(response.tokens):
                        # We intentionally do not use `with Trace(...)` to minimize
                        # nesting in code.
                        # Additionally, using a parent span and pushing/popping causes
                        # the nsys trace to be overly noisy since this is an async loop.
                        tracer = Tracer("tokenizer.decode")
                        decoded_token = await self.tokenizer.decode(
                            token,
                            skip_special_tokens=skip_special_tokens,
                        )
                        del tracer  # tokenizer.decode

                        # Detect custom stop phrases
                        stop_sequence_match = None
                        if len(stop_detector.stop) > 0:
                            tracer = Tracer("stop_detector.step")
                            if stop_sequence_match := stop_detector.step(
                                decoded_token
                            ):
                                # Tell the scheduler to stop generating this request
                                self.engine_queue.cancel_push_socket.put(
                                    [request.request_id]
                                )

                                logger.debug(
                                    f"Cancelling {request.request_id} because stop sequence ({stop_sequence_match}) detected in {stop_detector.continuation_tail}"
                                )
                            del tracer  # stop_detector.step

                        token_log_probabilities = None
                        top_log_probabilities = None
                        if response.log_probabilities:
                            log_prob = response.log_probabilities[i]
                            tracer = Tracer("collect_log_probs")
                            (
                                token_log_probabilities,
                                top_log_probabilities,
                            ) = await self._collect_log_probs(
                                log_prob, context, skip_special_tokens
                            )
                            del tracer  # collect_log_probs

                        output = TokenGeneratorOutput(
                            decoded_token=decoded_token,
                            token_log_probabilities=token_log_probabilities,
                            top_log_probabilities=top_log_probabilities,
                            prompt_token_count=context.current_length,
                            stop_sequence=stop_sequence_match,
                        )

                        tracer = Tracer("metrics_report_ttft_or_itl")
                        if i == 0:
                            METRICS.ttft(itl.elapsed_ms)
                        else:
                            METRICS.itl(itl.elapsed_ms)
                        itl.reset()
                        del tracer  # metrics_report_ttft_or_itl

                        yield output
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s [%d]: Completed: Elapsed: %0.2f ms",
                    request.request_id,
                    request.index,
                    total_sw.elapsed_ms,
                )

    async def all_tokens(
        self, request: TextGenerationRequest
    ) -> list[TokenGeneratorOutput]:
        """Generates all tokens for the provided request."""
        return [token async for token in self.next_token(request)]

    async def encode(
        self, request: TextGenerationRequest
    ) -> Optional[EmbeddingsGeneratorOutput]:
        """Generates embedded outputs for the provided request."""
        total_sw = StopWatch()
        self.logger.debug(
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.request_id,
            request.index,
            total_sw.elapsed_ms,
        )

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            with record_ms(METRICS.output_time):
                async for response in self.engine_queue.stream(
                    request.request_id, context
                ):
                    return EmbeddingsGeneratorOutput(
                        embeddings=response.embeddings
                    )
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s [%d]: Completed: Elapsed: %0.2f ms",
                    request.request_id,
                    request.index,
                    total_sw.elapsed_ms,
                )
        return None

    async def __aenter__(self) -> TokenGeneratorPipeline:
        self.logger.info("%s: Starting workers:", self.model_name)
        assert not self._background_tasks
        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError("Worker process not healthy not starting worker")

        # Add global fanout worker.
        self.create_background_task(self.engine_queue.response_worker)

        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError(
                "Worker process not healthy after running background task"
            )

        self.logger.info(
            "%s: Started workers: %d tasks",
            self.model_name,
            len(self._background_tasks),
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        self.logger.info("%s: Stopping workers", self.model_name)
        for task in self._background_tasks:
            task.cancel()
        # await asyncio.sleep(0.1)
        # TODO: also cancel any `queue.get()` tasks

    def create_background_task(
        self, fn: Callable[[], Coroutine[Any, Any, None]]
    ) -> None:
        task_name = fn.__name__
        task = asyncio.create_task(fn())
        task.add_done_callback(partial(self.log_task_done, task_name=task_name))
        self._background_tasks.add(task)
        self.logger.info(
            "%s: Task Added: %s, %s, %d total",
            self.model_name,
            task_name,
            type(fn),
            len(self._background_tasks),
        )

    def log_task_done(self, task: asyncio.Task, task_name: str) -> None:
        # TODO - should gracefully shut down here.
        self._background_tasks.remove(task)
        self.logger.info(
            "%s: Task completed: %s, %d remaining",
            self.model_name,
            task_name,
            len(self._background_tasks),
        )
        # Cancel remaining tasks.
        for t in self._background_tasks:
            if not t.done():
                t.cancel("Terminating task")
        if task.cancelled():
            return
        e = task.exception()
        if e:
            self.logger.error("Task completed with error. Stopping", exc_info=e)
            # Shut server down.
            # Sending SIGTERM is ugly, but simplifies the internal plumbing.
            os.kill(os.getpid(), signal.SIGTERM)


AudioGeneratorContext = TypeVar("AudioGeneratorContext")


class AudioGeneratorPipeline(Generic[AudioGeneratorContext]):
    """Base class for LLM audio generation pipelines."""

    def __init__(
        self,
        model_name: str,
        tokenizer: PipelineTokenizer,
        engine_queue: EngineQueue,
    ) -> None:
        self.logger = logging.getLogger(
            "max.serve.pipelines.AudioGeneratorPipeline"
        )
        self.logger.info("%s: Constructed", model_name)
        self.debug_logging = self.logger.isEnabledFor(logging.DEBUG)

        self.model_name = model_name
        self.tokenizer = tokenizer
        self.engine_queue = engine_queue
        self.stats = TokenGeneratorStats()

        self._background_tasks: set[asyncio.Task] = set()

    async def _collect_audio_metadata(self, response, context):  # noqa: ANN001
        # Collect metadata about generated audio like duration, sample rate etc.
        sample_rate = getattr(response, "sample_rate", None)
        duration = getattr(response, "duration", None)
        return AudioGenerationMetadata(
            sample_rate=sample_rate, duration=duration
        )

    async def next_chunk(
        self, request: AudioGenerationRequest
    ) -> AsyncGenerator[AudioGeneratorOutput, None]:
        """Generates and streams audio for the provided request."""
        total_sw = StopWatch()
        self.logger.debug(
            "%s [%d]: Started: Elapsed: %0.2f ms",
            request.request_id,
            request.index,
            total_sw.elapsed_ms,
        )

        try:
            with record_ms(METRICS.input_time):
                context = await self.tokenizer.new_context(request)

            with record_ms(METRICS.output_time):
                async for response in self.engine_queue.stream(
                    request.request_id, context
                ):
                    audio_metadata = await self._collect_audio_metadata(
                        response, context
                    )

                    output = AudioGeneratorOutput(
                        audio_data=response.audio_data,
                        metadata=audio_metadata,
                        is_done=response.is_done,
                    )

                    yield output
        finally:
            if self.debug_logging:
                self.logger.debug(
                    "%s [%d]: Completed: Elapsed: %0.2f ms",
                    request.request_id,
                    request.index,
                    total_sw.elapsed_ms,
                )

    async def generate_full_audio(
        self, request: AudioGenerationRequest
    ) -> AudioGeneratorOutput:
        """Generates complete audio for the provided request."""
        audio_chunks: list[AudioGeneratorOutput] = []
        async for chunk in self.next_chunk(request):
            if not chunk.audio_data.size:
                continue
            audio_chunks.append(chunk)

        # We import torch here so that only folks that use the
        # AudioGeneratorPipeline will need to have it installed.
        import numpy as np

        if len(audio_chunks) == 0:
            return AudioGeneratorOutput(
                audio_data=np.array([], dtype=np.float32),
                metadata=AudioGenerationMetadata(),
                is_done=True,
            )

        # Combine audio chunks and metadata.
        # Convert numpy arrays to torch tensors for concatenation, then back to numpy
        np_chunks = [chunk.audio_data for chunk in audio_chunks]
        combined_audio = np.concatenate(np_chunks, axis=-1)

        # We should only return from the next_chunk loop when the last chunk
        # is done.
        last_chunk = audio_chunks[-1]
        assert last_chunk.is_done

        return AudioGeneratorOutput(
            audio_data=combined_audio,
            metadata=last_chunk.metadata,
            is_done=True,
        )

    async def __aenter__(self):
        self.logger.info("%s: Starting workers:", self.model_name)
        assert not self._background_tasks
        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError("Worker process not healthy not starting worker")

        # Add global fanout worker.
        self.create_background_task(self.engine_queue.response_worker)

        if not self.engine_queue.is_worker_healthy():
            raise RuntimeError(
                "Worker process not healthy after running background task"
            )

        self.logger.info(
            "%s: Started workers: %d tasks",
            self.model_name,
            len(self._background_tasks),
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        self.logger.info("%s: Stopping workers", self.model_name)
        for task in self._background_tasks:
            task.cancel()
        # await asyncio.sleep(0.1)
        # TODO: also cancel any `queue.get()` tasks

    def create_background_task(
        self, fn: Callable[[], Coroutine[Any, Any, None]]
    ) -> None:
        task_name = fn.__name__
        task = asyncio.create_task(fn())
        task.add_done_callback(partial(self.log_task_done, task_name=task_name))
        self._background_tasks.add(task)
        self.logger.info(
            "%s: Task Added: %s, %s, %d total",
            self.model_name,
            task_name,
            type(fn),
            len(self._background_tasks),
        )

    def log_task_done(self, task: asyncio.Task, task_name: str) -> None:
        # TODO - should gracefully shut down here.
        self._background_tasks.remove(task)
        self.logger.info(
            "%s: Task completed: %s, %d remaining",
            self.model_name,
            task_name,
            len(self._background_tasks),
        )
        # Cancel remaining tasks.
        for t in self._background_tasks:
            if not t.done():
                t.cancel("Terminating task")
        if task.cancelled():
            return
        e = task.exception()
        if e:
            self.logger.error("Task completed with error. Stopping", exc_info=e)
            # Shut server down.
            # Sending SIGTERM is ugly, but simplifies the internal plumbing.
            os.kill(os.getpid(), signal.SIGTERM)
