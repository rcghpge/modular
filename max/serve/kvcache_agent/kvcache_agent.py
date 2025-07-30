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

import concurrent.futures
import logging
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import grpc
import zmq
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2 import (  # type: ignore
    KVCacheStateUpdate,
    MemoryTier,
    SubscriptionRequest,
    UpdateType,
)
from max.serve.kvcache_agent.kvcache_agent_service_v1_pb2_grpc import (
    KVCacheAgentServiceServicer,
    add_KVCacheAgentServiceServicer_to_server,
)
from max.serve.queue.zmq_queue import ZmqPullSocket

logger = logging.getLogger("max.serve")


@dataclass
class KVCacheChangeMessage:
    """A message that MAX Serve uses to communicate the KV cache updates to the agent."""

    cache_id: str
    memory_tier: MemoryTier
    update_type: UpdateType


@dataclass
class KVCacheAgentServerConfig:
    """Configuration for the KVCacheAgentServer."""

    host: str = "0.0.0.0"
    port: int = 50051
    num_workers: int = 4

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class KVCacheAgentServicer(KVCacheAgentServiceServicer):
    """Implementation of the KVCacheAgentService service."""

    def __init__(self) -> None:
        """Initialize the KVCacheAgentServicer."""
        self._subscribers: set[Any] = set()
        self._cache_state: dict[MemoryTier, set[str]] = {}
        self._lock = threading.RLock()

    def SubscribeToUpdates(
        self, request: SubscriptionRequest, context: grpc.ServicerContext
    ) -> Iterator[KVCacheStateUpdate]:
        """
        Subscribe to cache state updates.

        Args:
            request: The subscription request.
            context: The gRPC ServicerContext.

        Yields:
            CacheStateUpdate: Stream of cache state updates.
        """
        logger.debug(f"New subscription request received from {context.peer()}")
        logger.debug(f"Current subscribers count: {len(self._subscribers)}")
        logger.debug(f"Current cache state: {self._cache_state}")
        subscription_event: threading.Event = threading.Event()
        subscription_queue: queue.Queue[KVCacheStateUpdate] = queue.Queue()

        # Add subscriber
        with self._lock:
            # Send initial state with all existing entries
            if self._cache_state:
                for memory_tier in self._cache_state:
                    if self._cache_state[memory_tier]:
                        initial_update = KVCacheStateUpdate(
                            update_type=UpdateType.UPDATE_TYPE_ADDED,
                            memory_tier=memory_tier,
                            cache_ids=list(self._cache_state[memory_tier]),
                        )
                        logger.debug(
                            f"preparing initial update: {initial_update}"
                        )
                        subscription_queue.put(initial_update)

            # Add this client to subscribers
            subscription_event.set()
            subscriber = (subscription_queue, subscription_event)
            self._subscribers.add(subscriber)
            logger.debug(f"added subscriber: {subscriber}")

        try:
            while context.is_active():
                # Wait for updates or cancellation
                if not subscription_event.wait(timeout=1.0):
                    continue

                # Process available updates
                with self._lock:
                    if subscription_queue.empty():
                        subscription_event.clear()
                        continue

                    # Get all queued updates
                    updates_to_send = []
                    while not subscription_queue.empty():
                        updates_to_send.append(subscription_queue.get())

                # Send all updates
                for update in updates_to_send:
                    logger.debug(f"sending update: {update}")
                    yield update

        finally:
            # Remove subscriber when done
            logger.debug(f"removing subscriber: {subscriber}")
            with self._lock:
                self._subscribers.discard(subscriber)

    def add_cache_entry(self, cache_id: str, memory_tier: MemoryTier) -> None:
        """
        Add a new cache entry.

        Args:
            cache_id: Unique identifier for the cache entry.
            memory_tier: Memory tier where the entry is stored.
        """

        with self._lock:
            # Add the new entry
            if memory_tier not in self._cache_state:
                self._cache_state[memory_tier] = set()

            self._cache_state[memory_tier].add(cache_id)
            self._notify_subscribers(
                UpdateType.UPDATE_TYPE_ADDED, cache_id, memory_tier
            )

    def remove_cache_entry(
        self, cache_id: str, memory_tier: MemoryTier
    ) -> None:
        """
        Remove a cache entry.

        Args:
            cache_id: Unique identifier for the cache entry to remove.
            memory_tier: Memory tier where the entry is stored.
        """
        with self._lock:
            if memory_tier not in self._cache_state:
                logger.warning(
                    f"Cache entry {cache_id} not found in memory tier {memory_tier}"
                )
                return

            self._cache_state[memory_tier].discard(cache_id)
            self._notify_subscribers(
                UpdateType.UPDATE_TYPE_REMOVED, cache_id, memory_tier
            )

    def _notify_subscribers(
        self, update_type: UpdateType, cache_id: str, memory_tier: MemoryTier
    ) -> None:
        """
        Notify all subscribers of a cache state update.

        Args:
            update_type: The type of update (added, removed).
            cache_id: The cache id that was updated.
            memory_tier: The memory tier where the cache entry is stored.
        """
        update = KVCacheStateUpdate(
            update_type=update_type,
            memory_tier=memory_tier,
            cache_ids=[cache_id],
        )

        for subscription_queue, subscription_event in self._subscribers:
            subscription_queue.put(update)
            subscription_event.set()


class KVCacheAgentServer:
    """A server that hosts the KVCacheAgentService."""

    def __init__(
        self,
        config: KVCacheAgentServerConfig,
        zmq_ctx: zmq.Context,
        kv_cache_events_zmq_endpoint: str,
    ) -> None:
        """
        Initialize the KVCacheAgentServer.

        Args:
            config: Configuration for the server.
        """
        self.config = config
        self._kv_cache_events_pull_socket = ZmqPullSocket[KVCacheChangeMessage](
            zmq_ctx, kv_cache_events_zmq_endpoint
        )
        self.server = grpc.server(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.num_workers
            )
        )
        self.servicer = KVCacheAgentServicer()

        # Register the servicer
        add_KVCacheAgentServiceServicer_to_server(self.servicer, self.server)

        # Add a listening port
        self.server.add_insecure_port(self.config.address)

        # Start a background thread to process kv cache events
        self._kv_cache_events_stop_event = threading.Event()
        self._kv_cache_events_thread = threading.Thread(
            target=self._process_kv_cache_events,
            daemon=True,
            name="KVCacheAgentEventsProcessor",
        )
        self._started = False

    def _pull_from_kv_cache_events_socket(self) -> KVCacheChangeMessage:
        """
        Pull a message from the kv cache events socket.
        """
        return self._kv_cache_events_pull_socket.get_nowait()

    def _process_kv_cache_events(self) -> None:
        """
        Process messages from the kv cache events socket and update the cache state.

        Runs in a background thread, continuously polling the kv cache events socket for new messages.
        Based on the message type, it will call the appropriate method to update the cache.
        """
        logger.info("KV Cache Events processor thread started")

        while not self._kv_cache_events_stop_event.is_set():
            try:
                message = self._pull_from_kv_cache_events_socket()

                logger.debug(f"Received message: {message}")

                if message.update_type == UpdateType.UPDATE_TYPE_ADDED:
                    self.servicer.add_cache_entry(
                        message.cache_id, message.memory_tier
                    )
                elif message.update_type == UpdateType.UPDATE_TYPE_REMOVED:
                    self.servicer.remove_cache_entry(
                        message.cache_id, message.memory_tier
                    )
                else:
                    logger.warning(f"Unknown operation: {message.update_type}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing queue message: {e}")
                # Continue processing other messages even if one fails

        logger.info("KV Cache Events processor thread stopped")

    def start(self) -> None:
        """Start the KVCacheAgentServer."""
        if self._started:
            return

        self.server.start()
        self._kv_cache_events_thread.start()
        self._started = True
        logger.info(f"KVCacheAgentServer started on {self.config.address}")

    def stop(self, grace: int = 5) -> None:
        """
        Stop the KVCacheAgentServer.

        Args:
            grace: Grace period in seconds for clean shutdown.
        """
        if not self._started:
            return

        self._kv_cache_events_stop_event.set()
        self.server.stop(grace)
        self._kv_cache_events_thread.join()
        self._started = False
        logger.info("KVCacheAgentServer stopped")

    def wait_for_termination(self) -> None:
        """Block until the server terminates."""
        if self._started:
            self.server.wait_for_termination()
            self._kv_cache_events_thread.join()


def start_kvcache_agent_service(
    kv_cache_events_zmq_endpoint: str,
    zmq_ctx: zmq.Context,
    host: str = "0.0.0.0",
    port: int = 50051,
    num_workers: int = 10,
) -> KVCacheAgentServer:
    """
    Start the KVCacheAgentService on the specified address and port.

    Args:
        kv_cache_events_zmq_endpoint: The ZMQ endpoint for the agent to listen for kv cache events.
        zmq_ctx: An optional ZMQ context. One will be created if not provided.
        host: The server address to bind to.
        port: The port to listen on.
        num_workers: Number of worker threads for handling requests.

    Returns:
        KVCacheAgentServer: The running server instance.
    """

    config = KVCacheAgentServerConfig(
        host=host, port=port, num_workers=num_workers
    )
    server = KVCacheAgentServer(config, zmq_ctx, kv_cache_events_zmq_endpoint)
    server.start()
    return server
