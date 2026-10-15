# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Cascade worker framework: base classes for workers, proxies, and runtimes."""

from __future__ import annotations

import asyncio
import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
)
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
)

from max.experimental.cascade.core.pipeline_method import _get_pipeline_context
from max.experimental.cascade.core.result import Result, ResultIter
from max.experimental.cascade.core.type_hints import (
    return_type,
    stream_elem_type,
)

T = TypeVar("T")


@dataclass
class Worker:
    """Base class for cascade workers with deployment metadata."""

    # hints to guide deployment in heterogenous runtime pools
    deploy_hints: list[str] = field(default_factory=list)
    deploy_timeout: float | None = None

    @asynccontextmanager
    async def open(self) -> AsyncIterator[Worker]:
        """Lifecycle context manager. Override to add setup/teardown."""
        yield self


WorkerType = TypeVar("WorkerType", bound=Worker)


# Note having Runtime as a member on Result is actually a feature
# When Runtime is a thin client proxy Runtime, it's serializable as a URL
# We can explicitly hook the rebuild of runtime to memoize by URL
# to enable efficient grpc connection sharing


class Runtime(AsyncExitStack, ABC):
    """Transport-agnostic runtime for worker execution and result delivery.

    Concrete runtimes are async context managers, sharing the
    :py:class:`AsyncExitStack` lifecycle so any custom setup/teardown a
    subclass needs (connection pools, task groups, subprocess handles,
    ...) plugs in by pushing onto the stack in an overridden
    ``__aenter__``. Exit tears registered resources down in reverse
    order. There is no separate ``open()`` method -- runtimes follow the
    same shape as :py:mod:`asyncio` and standard-library context
    managers.
    """

    @abstractmethod
    async def deploy_worker(self, worker: Worker) -> str:
        """Wire primitive: register a worker and return its stable id.

        This is the primitive that differs between local and remote runtimes.
        Most callers want :py:meth:`deploy` instead, which wraps the returned
        id in a client-side :py:class:`Proxy`.
        """
        ...

    async def deploy(self, worker: WorkerType) -> WorkerType:
        """Register a worker and return a client-side :py:class:`Proxy`.

        The runtime returns a :py:class:`Proxy` that is observationally
        equivalent to ``worker`` (same public method surface, same awaited
        result types, same async-iterator streaming surface). The return type
        is annotated as ``WorkerType`` so callers can type pipeline attributes
        with their worker classes (or :py:class:`typing.Protocol` interfaces)
        and get end-to-end type checking; the runtime cast is safe because
        :py:class:`Proxy` exposes a structurally compatible interface.

        Raises:
            asyncio.TimeoutError: If the worker is not deployed within its
                ``deploy_timeout`` seconds.
        """
        worker_id = await asyncio.wait_for(
            self.deploy_worker(worker), timeout=worker.deploy_timeout
        )
        return cast(WorkerType, Proxy.of_worker(self, worker_id, worker))

    async def deploy_interface(
        self, interface: object, config: Mapping[str, object] | None = None
    ) -> Proxy[Any]:
        """Deploy the runtime's own implementation of a schema interface.

        The proxy's methods are read from the interface, so there is no
        worker class to write. ``config`` fills the worker's constructor
        config struct.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot deploy schema interfaces"
        )

    @abstractmethod
    def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> AbstractAsyncContextManager[str]:
        """Launch a worker method asynchronously.

        Returns an async context manager that yields the ``result_id`` bound
        to the call. The id stays valid for the lifetime of the ``async
        with`` block; exiting cancels the in-flight task and releases the
        result buffer. The context is normally entered via
        :py:meth:`AsyncExitStack.enter_async_context` on the pipeline-scope
        stack returned by :py:func:`_get_pipeline_context`, so the call
        lifetime tracks the surrounding request.
        """
        ...

    @abstractmethod
    def get_result(self, resid: str) -> Awaitable[object]:
        """Await a single result.

        Raises whatever exception the call raised.
        """
        ...

    @abstractmethod
    def stream_result(self, resid: str) -> AsyncIterator[object]:
        """Bind a streaming result and iterate inline.

        Single-consumer: the first call (across :py:meth:`stream_result` and
        :py:meth:`stream_next`) binds the stream; subsequent attempts on the
        same ``resid`` raise. Used for lightweight inline streams (such as token
        streams) where backpressure rides on the underlying transport.
        """
        ...

    def make_result(
        self, result_id: str, type_hint: object | None
    ) -> Result[T]:
        """Build the :py:class:`Result` a proxy returns for a call.

        The default resolves through :py:meth:`get_result`. A runtime can
        return a :py:class:`Result` subclass that waits on its own transport
        instead; callers use it through the same interface.
        """
        return Result(result_id, self, type_hint)

    def make_result_iter(
        self, result_id: str, type_hint: object | None
    ) -> ResultIter[T]:
        """Build the :py:class:`ResultIter` a proxy returns for a stream.

        The default iterates :py:meth:`stream_result`. A runtime can return
        a :py:class:`ResultIter` subclass that reads its own transport
        instead; callers use it through the same interface.
        """
        return ResultIter(result_id, self, type_hint)

    @abstractmethod
    async def get_metrics(self) -> str:
        """Return Prometheus metrics summary."""
        ...


@dataclass(frozen=True)
class ProxyMethod:
    """What a :py:class:`Proxy` needs to know to bind one worker method."""

    stream: bool
    # Expected type of the resolved value, or of each streamed item.
    type_hint: object | None = None
    # The worker method this stands in for, whose metadata the proxy copies.
    impl: Callable[..., object] | None = None


class Proxy(Generic[T]):
    """Client-side handle to a deployed worker.

    Attaches one bound method per entry of ``methods``. Each method picks a
    fresh ``result_id``, hands it to ``runtime.call_method`` for binding, and
    returns it wrapped in a :py:class:`Result` (or :py:class:`ResultIter` for
    a streaming method). The methods come from a :py:class:`Worker` class
    (:py:meth:`of_worker`) or, for a worker the runtime implements, from its
    schema interface.
    """

    runtime: Runtime
    worker_id: str

    def __init__(
        self,
        runtime: Runtime,
        worker_id: str,
        methods: Mapping[str, ProxyMethod],
    ) -> None:
        self.runtime = runtime
        self.worker_id = worker_id
        for name, method in methods.items():
            setattr(self, name, self._bind(name, method))

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
            """Methods are attached from the worker's schema or class."""
            ...

    @classmethod
    def of_worker(
        cls, runtime: Runtime, worker_id: str, worker: Worker
    ) -> Proxy[Any]:
        """A proxy with one method per public method of ``type(worker)``."""
        methods: dict[str, ProxyMethod] = {}
        worker_class = type(worker)
        for name in dir(worker_class):
            method = getattr(worker_class, name, None)
            if name.startswith("_") or not callable(method):
                continue
            if inspect.isasyncgenfunction(method):
                methods[name] = ProxyMethod(
                    True, stream_elem_type(method), method
                )
            else:
                methods[name] = ProxyMethod(False, return_type(method), method)
        return cls(runtime, worker_id, methods)

    def _bind(
        self, name: str, method: ProxyMethod
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        runtime = self.runtime
        worker_id = self.worker_id

        async def call(*args: object, **kwargs: object) -> object:
            context = _get_pipeline_context()
            result_id = await context.enter_async_context(
                runtime.call_method(worker_id, name, args, kwargs)
            )
            if method.stream:
                return runtime.make_result_iter(result_id, method.type_hint)
            return runtime.make_result(result_id, method.type_hint)

        return functools.wraps(method.impl)(call) if method.impl else call
