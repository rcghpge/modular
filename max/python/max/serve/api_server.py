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

"""MAX serving in Python prototype. Main API server thing."""

from __future__ import annotations

import logging
import os
import signal
import socket
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from max.interfaces import PipelinesFactory, PipelineTask, PipelineTokenizer
from max.nn.kv_cache.paged_cache import ResetPrefixCacheFrontend
from max.pipelines.lib import PipelineConfig
from max.serve.config import APIType, MetricRecordingMethod, Settings
from max.serve.pipelines.llm import (
    AudioGeneratorPipeline,
    TokenGeneratorPipeline,
)
from max.serve.pipelines.model_worker import start_model_worker
from max.serve.pipelines.telemetry_worker import start_telemetry_consumer
from max.serve.queue.lora_queue import LoRAQueue
from max.serve.recordreplay.jsonl import JSONLFileRecorder
from max.serve.recordreplay.middleware import RecorderMiddleware
from max.serve.request import register_request
from max.serve.router import kserve_routes, openai_routes, sagemaker_routes
from max.serve.scheduler.queues import SchedulerZmqConfigs
from max.serve.telemetry.common import send_telemetry_log
from max.serve.telemetry.metrics import METRICS
from uvicorn import Config

ROUTES = {
    APIType.KSERVE: kserve_routes,
    APIType.OPENAI: openai_routes,
    APIType.SAGEMAKER: sagemaker_routes,
}

logger = logging.getLogger("max.serve")


def validate_port_is_free(port: int) -> int:
    # check if port is already in use
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            return port
        except OSError as e:
            raise ValueError(
                f"The network port {port} is already in use"
            ) from e


@dataclass(frozen=True)
class ServingTokenGeneratorSettings:
    # Pipeline config
    model_factory: PipelinesFactory  # type: ignore
    pipeline_config: PipelineConfig
    tokenizer: PipelineTokenizer[Any, Any, Any]
    pipeline_task: PipelineTask = PipelineTask.TEXT_GENERATION


@asynccontextmanager
async def lifespan(
    app: FastAPI,
    settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
) -> AsyncGenerator[None]:
    try:
        if not settings.disable_telemetry:
            send_telemetry_log(
                serving_settings.pipeline_config.model_config.model_name
            )
    except Exception as e:
        logger.warning("Failed to send telemetry log: %s", e)

    if settings.offline_inference:
        raise ValueError(
            "It is not valid to start the API Server if the server is in offline inference mode"
        )

    logger.info("Starting server...")

    async with AsyncExitStack() as exit_stack:
        # start telemetry worker and configure Metrics to use it

        metric_client = await exit_stack.enter_async_context(
            start_telemetry_consumer(settings)
        )
        METRICS.configure(client=metric_client)

        # start model worker
        scheduler_zmq_configs = SchedulerZmqConfigs(
            serving_settings.pipeline_task
        )
        worker_monitor = await exit_stack.enter_async_context(
            start_model_worker(
                serving_settings.model_factory,
                serving_settings.pipeline_config,
                settings,
                metric_client,
                scheduler_zmq_configs=scheduler_zmq_configs,
            )
        )

        lora_queue: LoRAQueue | None = (
            LoRAQueue(serving_settings.pipeline_config.zmq_endpoint_base)
            if serving_settings.pipeline_config.lora_config
            else None
        )

        METRICS.pipeline_load(
            serving_settings.pipeline_config.model_config.model_name
        )

        pipeline: TokenGeneratorPipeline | AudioGeneratorPipeline
        if serving_settings.pipeline_task in (
            PipelineTask.TEXT_GENERATION,
            PipelineTask.EMBEDDINGS_GENERATION,
        ):
            pipeline = TokenGeneratorPipeline(
                model_name=serving_settings.pipeline_config.model_config.model_name,
                tokenizer=serving_settings.tokenizer,
                lora_queue=lora_queue,
                scheduler_zmq_configs=scheduler_zmq_configs,
            )
        elif serving_settings.pipeline_task == PipelineTask.AUDIO_GENERATION:
            pipeline = AudioGeneratorPipeline(
                model_name=serving_settings.pipeline_config.model_config.model_name,
                tokenizer=serving_settings.tokenizer,
                lora_queue=lora_queue,
                scheduler_zmq_configs=scheduler_zmq_configs,
            )
        else:
            raise ValueError(
                f"Unsupported pipeline task: {serving_settings.pipeline_task}"
            )

        app.state.pipeline = pipeline
        app.state.pipeline_config = serving_settings.pipeline_config

        await exit_stack.enter_async_context(pipeline)
        logger.info(
            f"\n\n{'*' * 80}\n\n"
            f"{f'🚀 Server ready on http://{settings.host}:{settings.port} (Press CTRL+C to quit)'.center(80)}\n\n"
            f"{'*' * 80}\n"
        )

        yield

        logger.info("Shutting down workers...")


def version() -> JSONResponse:
    """Returns max-serve version information."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        package_version = version("max")
        return JSONResponse({"version": package_version})
    except PackageNotFoundError:
        logger.debug("Version could not be reported for max.")
        return JSONResponse({"version": "unknown"})


async def health() -> Response:
    """Health check, tools like lm-eval use this to check for readiness."""
    return Response(status_code=200)


def make_metrics_app() -> Callable[..., Any]:
    from prometheus_client import disable_created_metrics, make_asgi_app

    disable_created_metrics()
    return make_asgi_app()


def fastapi_app(
    settings: Settings,
    serving_settings: ServingTokenGeneratorSettings,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan_wrap(app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            async with lifespan(app, settings, serving_settings):
                yield
        except BaseException as e:
            logger.exception("Worker exception, Shutting down. %s", e)
            # Caught by uvicorn to shutdown the server
            os.kill(os.getpid(), signal.SIGINT)
            # After first SIGINT uvicorn waits for pending requests to complete
            # In our case, they would hang forever due to waiting on worker queues
            # Uvicorn listens for a second SIGINT to cancel this waiting phase and
            # close all remaining connections with "Internal Server Error" status
            os.kill(os.getpid(), signal.SIGINT)
            # Ideally we'd just rethrow here, which is caught by
            # starlette Router.lifespan() and converted into ASGI
            # lifespan.shutdown.failed event. However uvicorn only
            # listens for this event if it's already initiated the
            # shutdown sequence.
            # See https://github.com/Kludex/uvicorn/discussions/2298

    app = FastAPI(title="MAX Serve", lifespan=lifespan_wrap)

    if settings.transaction_recording_file is not None:
        transaction_recording_file = settings.transaction_recording_file
        app.add_middleware(
            RecorderMiddleware,  # type: ignore
            recorder_factory=(
                lambda: JSONLFileRecorder(transaction_recording_file)
            ),
            include_responses=settings.transaction_recording_include_responses,
        )

    if (
        not settings.disable_telemetry
        and settings.metric_recording == MetricRecordingMethod.ASYNCIO
    ):
        app.mount("/metrics", make_metrics_app())

    app.add_api_route("/version", version)
    app.add_api_route("/health", health)

    reset_prefix_cache_frontend = ResetPrefixCacheFrontend(
        serving_settings.pipeline_config.zmq_endpoint_base
    )

    async def reset_prefix_cache() -> Response:
        """Reset the prefix cache."""
        if not serving_settings.pipeline_config.model_config.kv_cache_config.enable_prefix_caching:
            return Response(
                status_code=400,
                content="Prefix caching is not enabled. Ignoring request",
            )

        reset_prefix_cache_frontend.enqueue_reset_prefix_cache()
        return Response(status_code=200, content="Success")

    app.add_api_route(
        "/reset_prefix_cache", reset_prefix_cache, methods=["POST"]
    )

    for api_type in settings.api_types:
        app.include_router(ROUTES[api_type].router)

    app.state.settings = settings
    register_request(app)

    return app


def fastapi_config(app: FastAPI, server_settings: Settings) -> Config:
    config = Config(
        app=app,
        log_config=None,
        loop="uvloop",
        host=server_settings.host,
        port=server_settings.port,
        timeout_graceful_shutdown=5,
    )

    for route in app.routes:
        logger.debug("Route enabled : %s", route)
    return config
