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

import asyncio
import functools
import logging
import queue
import sys
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Callable, NoReturn, Optional

from max.serve.config import MetricLevel
from max.serve.telemetry.metrics import MaxMeasurement, MetricClient

logger = logging.getLogger("max.serve")


if sys.version_info >= (3, 13):
    QueueShutDown = asyncio.QueueShutDown
else:

    class QueueShutDown(Exception):
        pass


class NotStarted(Exception):
    pass


class AsyncioMetricClient(MetricClient):
    def __init__(
        self, level: MetricLevel, q: asyncio.Queue[MaxMeasurement]
    ) -> None:
        self.q = q
        self.level = level

    def __getstate__(self) -> NoReturn:
        raise TypeError(
            "AsyncioMetricClient is not safe to serialize.  "
            "Use cross_process_factory to safely send across processes."
        )

    def send_measurement(self, m: MaxMeasurement, level: MetricLevel) -> None:
        if level > self.level:
            return
        try:
            self.q.put_nowait(m)
        except queue.Full:
            logger.warning(
                f"Telemetry Queue is full. Dropping data for {m.instrument_name}"
            )

    def cross_process_factory(
        self,
    ) -> Callable[[], AbstractAsyncContextManager[MetricClient]]:
        return functools.partial(start_asyncio_consumer, self.level)


class AsyncioTelemetryController:
    """Execute a sync function asynchronously

    Use an asyncio Queue & Task to asynchronously commit metric measurements
    """

    def __init__(self, maxsize=0) -> None:  # noqa: ANN001
        self.q: asyncio.Queue[MaxMeasurement] = asyncio.Queue(maxsize=maxsize)
        self.task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self.task is not None:
            raise Exception("task already started")
        self.task = asyncio.create_task(self._consume(self.q))

    async def shutdown(self, timeout_s: float = 2.0) -> None:
        if sys.version_info >= (3, 13):
            self.q.shutdown()

        # task was never started.  return early
        if self.task is None:
            return

        try:
            await asyncio.wait_for(self.task, timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "AsyncioTelemetryController timeout out while stopping"
            )
        finally:
            self.task = None

    def Client(self, level: MetricLevel) -> AsyncioMetricClient:
        if self.task is None:
            raise NotStarted(
                "AsyncioTelemetryController task not started. Cannot enqueue work."
            )
        return AsyncioMetricClient(level, self.q)

    @staticmethod
    async def _consume(q: asyncio.Queue[MaxMeasurement]) -> None:
        while True:
            try:
                m: MaxMeasurement = await q.get()
            except QueueShutDown:
                break
            except asyncio.CancelledError:
                logger.debug("AsyncioTelemetryController cancelled")
                break

            try:
                m.commit()
            except:
                logger.warning("Failed to record telemetry", exc_info=True)

        logger.debug(
            f"AsyncioTelemetryController consumer shut down. Residual queue size: {q.qsize()}"
        )

    async def __aenter__(self):
        self.start()
        return self

    async def __aexit__(self, type, value, traceback):  # noqa: ANN001
        await self.shutdown()


@asynccontextmanager
async def start_asyncio_consumer(
    level: MetricLevel,
) -> AsyncGenerator[AsyncioMetricClient, None]:
    async with AsyncioTelemetryController() as controller:
        yield controller.Client(level)
