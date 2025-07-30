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

import logging
import logging.handlers
import os
import platform
import uuid
from time import time
from typing import Union

import requests
from max.serve.config import Settings
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from pythonjsonlogger import jsonlogger

otelBaseUrl = "https://telemetry.modular.com:443"


def _getCloudProvider() -> str:
    providers = ["amazon", "google", "microsoft", "oracle"]
    path = "/sys/class/dmi/id/"
    if os.path.isdir(path):
        for idFile in os.listdir(path):
            try:
                with open(idFile) as file:
                    contents = file.read().lower()
                    for provider in providers:
                        if provider in contents:
                            return provider
            except Exception:
                pass
    return ""


def _getWebUserId() -> str:
    try:
        idFile = os.path.expanduser("~") + "/.modular/webUserId"
        with open(idFile) as file:
            return file.readline().rstrip("\n")
    except Exception:
        return ""


logs_resource = Resource.create(
    {
        "event.domain": "serve",
        "telemetry.session": uuid.uuid4().hex,
        "web.user.id": _getWebUserId(),
        "enduser.id": os.environ.get("MODULAR_USER_ID", ""),
        "os.type": platform.system(),
        "os.version": platform.release(),
        "cpu.description": platform.processor(),
        "cpu.arch": platform.architecture()[0],
        "system.cloud": _getCloudProvider(),
        "deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", ""),
    }
)

metrics_resource = Resource.create(
    {
        "enduser.id": os.environ.get("MODULAR_USER_ID", ""),
        "deployment.id": os.environ.get("MAX_SERVE_DEPLOYMENT_ID", ""),
    }
)


def get_log_level(settings: Settings) -> Union[int, str, None]:
    otlp_level: Union[int, str, None] = (
        logging.getLevelName(settings.logs_otlp_level)
        if settings.logs_otlp_level
        else None
    )

    if settings.disable_telemetry:
        otlp_level = None

    return otlp_level


# Create a logger that buffers logs in memory and prints them to the console in batches.
# The returned logger is a no op if logging has not yet been configured.
def get_batch_logger(
    parent_logger: logging.Logger, capacity: int = 10
) -> logging.Logger:
    batch_logger = logging.getLogger(parent_logger.name)
    console_handlers = [
        h
        for h in logging.getLogger().handlers
        if type(h) is logging.StreamHandler
    ]
    memory_handler = logging.handlers.MemoryHandler(
        capacity=capacity,
        target=console_handlers[0] if len(console_handlers) > 0 else None,
    )
    batch_logger.addHandler(memory_handler)
    batch_logger.propagate = False
    return batch_logger


# Force a flush of the batch given logger.
def flush_batch_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if type(handler) is logging.handlers.MemoryHandler:
            handler.flush()


COLOR_MAP = {
    "green": "\033[92m",
    "blue": "\033[94m",
    "red": "\033[91m",
}


# Configure logging to console and OTEL.  This should be called before any
# 3rd party imports whose logging you wish to capture.
# Note that the color is not propagated to subprocesses. eg: ModelWorker
def configure_logging(settings: Settings, color: str | None = None) -> None:
    otlp_level = get_log_level(settings)
    egress_enabled = not settings.disable_telemetry

    logging_handlers: list[logging.Handler] = []

    # Set up log filtering
    components_to_log = [
        "root",
        "max.entrypoints",
        "max.pipelines",
        "max.serve",
    ]
    try:
        if settings.logs_enable_components is not None:
            components = settings.logs_enable_components.split(",")
            components_to_log.extend(components)
    except Exception:
        print(
            "ERROR: Failed to parse logging components setting!  Using default."
        )

    def LogFilter(record):  # noqa: ANN001
        return record.name in components_to_log

    # Create a console handler
    if settings.logs_console_level is not None:
        if color is not None:
            if color not in COLOR_MAP:
                raise ValueError(f"Invalid color: {color}")
            color_code = COLOR_MAP[color]
            color_terminator = "\033[0m"
        else:
            color_code = ""
            color_terminator = ""

        console_handler = logging.StreamHandler()
        console_formatter: logging.Formatter
        if settings.structured_logging:
            console_formatter = jsonlogger.JsonFormatter(
                f"{color_code}%(levelname)s %(process)d %(threadName)s %(name)s %(message)s %(request_id)s %(batch_id)s{color_terminator}",
                timestamp=True,
            )
        else:
            console_formatter = logging.Formatter(
                (
                    f"{color_code}%(asctime)s.%(msecs)03d %(levelname)s: %(process)d %(threadName)s:"
                    f" %(name)s:{color_terminator} %(message)s"
                ),
                datefmt="%H:%M:%S",
            )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(settings.logs_console_level)
        console_handler.addFilter(LogFilter)

        logging_handlers.append(console_handler)

    if (
        settings.logs_file_level is not None
        and settings.logs_file_path is not None
    ):
        # Create a file handler
        file_handler = logging.FileHandler(settings.logs_file_path)
        file_formatter: logging.Formatter
        if settings.structured_logging:
            file_formatter = jsonlogger.JsonFormatter(
                "%(levelname)s %(process)d %(threadName)s %(name)s %(message)s %(request_id)s %(batch_id)s",
                timestamp=True,
            )
        else:
            file_formatter = logging.Formatter(
                (
                    "%(asctime)s.%(msecs)03d %(levelname)s: %(process)d %(threadName)s:"
                    " %(name)s: %(message)s"
                ),
                datefmt="%y:%m:%d-%H:%M:%S",
            )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(settings.logs_file_level)
        file_handler.addFilter(LogFilter)
        logging_handlers.append(file_handler)

    if egress_enabled and otlp_level is not None:
        # Create an OTEL handler
        logger_provider = LoggerProvider(logs_resource)
        set_logger_provider(logger_provider)
        exporter = OTLPLogExporter(endpoint=otelBaseUrl + "/v1/logs")
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(exporter)
        )
        otlp_handler = LoggingHandler(
            level=logging.getLevelName(otlp_level),
            logger_provider=logger_provider,
        )
        otlp_handler.addFilter(LogFilter)
        logging_handlers.append(otlp_handler)

    # Configure root logger level
    logger = logging.getLogger()
    if len(logging_handlers) > 0:
        logger_level = min(h.level for h in logging_handlers)
        logger.setLevel(logger_level)
        for handler in logging_handlers:
            logger.addHandler(handler)

        # TODO use FastAPIInstrumentor once Motel supports traces.
        # For now, manually configure uvicorn.
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        # Explicit levels to reduce noise
        logging.getLogger("sse_starlette.sse").setLevel(
            max(logger_level, logging.INFO)
        )

    logger.info(
        "Logging initialized: Console: %s, File: %s, Telemetry: %s",
        settings.logs_console_level,
        settings.logs_file_level,
        otlp_level,
    )


def configure_metrics(settings: Settings) -> None:
    egress_enabled = not settings.disable_telemetry

    meterProviders: list[MetricReader] = [PrometheusMetricReader(True)]
    if egress_enabled:
        meterProviders.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=otelBaseUrl + "/v1/metrics")
            )
        )
    set_meter_provider(MeterProvider(meterProviders, metrics_resource))

    logger = logging.getLogger()
    if settings.disable_telemetry:
        logger.info("Metrics disabled.")
    else:
        logger.info("Metrics initialized.")


# Send a simple one-time structured log, avoiding the buggy OTEL SDK
# (see MAXSERV-904)
def send_telemetry_log(model_name: str) -> None:
    request_body = f"""{{
  "resourceLogs": [
    {{
      "resource": {{
        "attributes": [
          {{"key": "deployment.model", "value": {{"stringValue": "{model_name}"}}}},
          {{"key": "web.user.id", "value": {{"stringValue": "{logs_resource.attributes["web.user.id"]}"}}}},
          {{"key": "enduser.id", "value": {{"stringValue": "{logs_resource.attributes["enduser.id"]}"}}}},
          {{"key": "deployment.id", "value": {{"stringValue": "{logs_resource.attributes["deployment.id"]}"}}}},
          {{"key": "os.type", "value": {{"stringValue": "{logs_resource.attributes["os.type"]}"}}}},
          {{"key": "os.version", "value": {{"stringValue": "{logs_resource.attributes["os.version"]}"}}}},
          {{"key": "cpu.description", "value": {{"stringValue": "{logs_resource.attributes["cpu.description"]}"}}}},
          {{"key": "cpu.arch", "value": {{"stringValue": "{logs_resource.attributes["cpu.arch"]}"}}}},
          {{"key": "system.cloud", "value": {{"stringValue": "{logs_resource.attributes["system.cloud"]}"}}}},
          {{"key": "service.name", "value": {{"stringValue": "unknown_service"}}}},
          {{"key": "telemetry.sdk.language", "value": {{"stringValue": "python"}}}},
          {{"key": "telemetry.sdk.version", "value": {{"stringValue": "0.0.0"}}}},
          {{"key": "telemetry.sdk.name", "value": {{"stringValue": "opentelemetry"}}}}
        ]
      }},
      "scopeLogs": [
        {{
          "logRecords": [
            {{
              "attributes": [
                {{"key": "event.domain", "value": {{"stringValue": "modular"}}}},
                {{"key": "event.name", "value": {{"stringValue": "serve.telemetry.log"}}}}
              ],
              "body": {{"stringValue": ""}},
              "observedTimeUnixNano": "{int(time() * 1_000_000_000)}",
              "severityNumber": 9,
              "severityText": "INFO"
            }}
          ],
          "scope": {{"name": "modular_logger"}}
        }}
      ]
    }}
  ]
}}"""

    requests.post(
        otelBaseUrl + "/v1/logs",
        data=request_body,
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
