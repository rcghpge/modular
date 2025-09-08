#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Benchmark test for dispatcher service communication.

This module sets up a dispatcher service running in a separate thread using
dynamic ZMQ transport. The service runs for the entire test duration and is
not rebuilt or restarted for each benchmark invocation.

Key features:
- Global port configuration (default: 5555, configurable via DISPATCHER_PORT env var)
- Global host configuration (default: 0.0.0.0, configurable via MAX_SERVE_DISPATCHER_HOST env var)
- Shared transport configuration and factory to eliminate duplication
- Thread-based service management with proper lifecycle control
- Automatic cleanup on exit
- Reusable service across multiple benchmark runs
- Configurable benchmark execution (RUN_BENCHMARK env var: when set to false,
  keeps client and server alive indefinitely instead of running benchmark)

Environment Variables:
- MAX_SERVE_DISPATCHER_PORT: Port for the dispatcher service (default: 5555)
- MAX_SERVE_DISPATCHER_HOST: Host for the dispatcher service (default: 127.0.0.1)
- MAX_SERVE_TARGET_PORT: Port for the target service (default: 5556)
- MAX_SERVE_TARGET_HOST: Host for the target service (default: 127.0.0.1)
- RUN_BENCHMARK: Whether to run the benchmark or keep services alive (default: true)
  Accepted values for false: "false", "0", "no", "off" (case insensitive)
  All other values are treated as true
"""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
import time
from typing import Any

import numpy as np
import pytest
from max.nn.kv_cache import XferReqData
from max.pipelines.core import TextContext
from max.serve.kvcache_agent.dispatcher_base import MessageType, ReplyContext
from max.serve.kvcache_agent.dispatcher_factory import (
    DispatcherConfig,
    DispatcherFactory,
    TransportFactory,
    TransportType,
)
from max.serve.kvcache_agent.dispatcher_service import DispatcherService
from max.serve.kvcache_agent.dispatcher_transport import (
    DynamicZmqTransport,
    TransportMessage,
)
from max.serve.scheduler.base import (
    PayloadType,
    PrefillRequest,
    PrefillResponse,
)

# Global configuration
DISPATCHER_PORT = int(os.environ.get("DISPATCHER_PORT", "5555"))
DISPATCHER_HOST = str(
    os.environ.get("DISPATCHER_HOST", "127.0.0.1")
)  # Bind to all interfaces for accessibility
DISPATCHER_BIND_ADDRESS = f"tcp://{DISPATCHER_HOST}:{DISPATCHER_PORT}"

# Target configuration for sending requests (defaults to local if not specified)
TARGET_PORT = int(os.environ.get("TARGET_PORT", "5556"))
TARGET_HOST = str(
    os.environ.get("TARGET_HOST", "127.0.0.1")
)  # Default to localhost
TARGET_ADDRESS = f"tcp://{TARGET_HOST}:{TARGET_PORT}"
TRANSPORT_INIT_TIMEOUT = 30.0

# Benchmark control configuration
RUN_BENCHMARK = os.environ.get("RUN_BENCHMARK", "true").lower() in (
    "true",
    "1",
    "yes",
    "on",
)

# Shared transport configuration and factory - created once and reused
# This eliminates duplication between service and client creation
_transport_config = TransportFactory.DynamicZmqTransportConfig(
    bind_address=DISPATCHER_BIND_ADDRESS,
    default_destination_address=TARGET_ADDRESS,
    instance_id="benchmark_dispatcher",
    socket_init_timeout=None,
)

_dispatcher_config = DispatcherConfig(
    transport=TransportType.DYNAMIC_ZMQ, transport_config=_transport_config
)

_dispatcher_factory = DispatcherFactory[PayloadType](
    config=_dispatcher_config,
    transport_payload_type=TransportMessage[PayloadType],
    local_socket_init_timeout=5.0,
)

# Global service state - these maintain the service across benchmark runs
_dispatcher_service: DispatcherService[PayloadType] | None = None
_dispatcher_thread: threading.Thread | None = None
_service_running = False
_service_lock = threading.RLock()


def _run_dispatcher_service_thread():
    """Run the dispatcher service in a separate thread."""
    global _dispatcher_service, _service_running

    print("Starting dispatcher service thread")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print("Created new event loop in service thread")

    try:
        print("Creating dispatcher service using factory")
        # Use the shared dispatcher factory
        _dispatcher_service = _dispatcher_factory.create_service()
        print("Dispatcher service created successfully")

        # Start the service
        async def run_service():
            print("Starting async service function")
            if _dispatcher_service is not None:
                print(
                    f"Starting dispatcher service on {DISPATCHER_BIND_ADDRESS}"
                )
                await _dispatcher_service.start()
                print(
                    f"Dispatcher service started successfully on {DISPATCHER_BIND_ADDRESS}"
                )
                assert isinstance(
                    _dispatcher_service.transport, DynamicZmqTransport
                ), "Transport is not a DynamicZmqTransport"
                await _dispatcher_service.transport.connect_to_default(
                    TRANSPORT_INIT_TIMEOUT
                )

                # Keep the service running
                try:
                    loop_count = 0
                    print("Entering service main loop")
                    while _service_running:
                        await asyncio.sleep(0.1)
                        loop_count += 1
                        if loop_count % 50 == 0:  # Log every 5 seconds
                            print(f"Service running, loop count: {loop_count}")
                    print("Service loop exiting, _service_running is False")
                finally:
                    print("Stopping dispatcher service")
                    await _dispatcher_service.stop()
                    print("Dispatcher service stopped successfully")
            else:
                print("_dispatcher_service is None, cannot start")

        print("Running service coroutine")
        loop.run_until_complete(run_service())
        print("Service coroutine completed")

    except Exception as e:
        print(f"Error in dispatcher service thread: {e}")
        raise
    finally:
        print("Closing event loop")
        loop.close()
        print("Dispatcher service thread finished")


def _ensure_dispatcher_service_running():
    """Ensure the dispatcher service is running in a separate thread."""
    global _dispatcher_service, _dispatcher_thread, _service_running

    print("Ensuring dispatcher service is running")
    with _service_lock:
        if (
            _service_running
            and _dispatcher_thread
            and _dispatcher_thread.is_alive()
        ):
            print("Service is already running, returning")
            return  # Service is already running

        print("Starting new dispatcher service")
        # Start the service
        _service_running = True
        _dispatcher_thread = threading.Thread(
            target=_run_dispatcher_service_thread, daemon=True
        )
        _dispatcher_thread.start()
        print("Dispatcher service thread started, waiting for initialization")

        # Wait a bit for the service to start
        time.sleep(0.5)
        print("Service initialization wait completed")


def _stop_dispatcher_service():
    """Stop the dispatcher service."""
    global _service_running, _dispatcher_thread

    print("Stopping dispatcher service")
    with _service_lock:
        if _service_running:
            print("Setting _service_running to False")
            _service_running = False
            if _dispatcher_thread and _dispatcher_thread.is_alive():
                print("Waiting for dispatcher thread to join (timeout=5.0s)")
                _dispatcher_thread.join(timeout=5.0)
                if _dispatcher_thread.is_alive():
                    print(
                        "Dispatcher service thread did not stop within timeout"
                    )
                else:
                    print("Dispatcher service thread stopped successfully")
        else:
            print("Service was not running, nothing to stop")


# Register cleanup function to stop the service on exit
atexit.register(_stop_dispatcher_service)


def create_variable_prefill_requests(
    n: int, min_tokens: int, max_tokens: int
) -> list[PrefillRequest]:
    """
    Create a list of PrefillRequest objects with TextContext containing variable token counts.

    Args:
        n: Number of PrefillRequest objects to create
        min_tokens: Minimum number of tokens in generated contexts
        max_tokens: Maximum number of tokens in generated contexts

    Returns:
        List of PrefillRequest objects with variable TextContext payloads
    """
    # Reset numpy random state to ensure variable token counts across runs
    # Use current time as seed for true randomness
    import time

    np.random.seed(int(time.time() * 1000000) % 2**32)

    print(
        f"Creating {n} PrefillRequest objects with token counts between {min_tokens} and {max_tokens}"
    )

    prefill_requests = []

    for i in range(n):
        # Generate variable token count within specified range
        token_count = np.random.randint(min_tokens, max_tokens + 1)

        # Create different token patterns to simulate realistic workload variance
        pattern_type = i % 3

        if pattern_type == 0:
            # Pattern 1: Random tokens (diverse vocabulary)
            tokens = np.random.randint(
                1, 32000, size=token_count, dtype=np.int32
            )
        elif pattern_type == 1:
            # Pattern 2: Repeating sequences (common in real text patterns)
            base_sequence_length = min(50, token_count)
            base_tokens = np.random.randint(
                1, 1000, size=base_sequence_length, dtype=np.int32
            )
            repetitions = (token_count // base_sequence_length) + 1
            tokens = np.tile(base_tokens, repetitions)[:token_count]
        else:
            # Pattern 3: Sequential tokens with random starting point
            start_token = np.random.randint(1, 10000)
            tokens = np.arange(
                start_token, start_token + token_count, dtype=np.int32
            )

        # Set max_length to accommodate tokens plus some buffer
        max_length = max(100, token_count + np.random.randint(10, 100))

        # Create TextContext with variable tokens
        text_context = TextContext(
            max_length=max_length,
            tokens=tokens,
        )

        # Create PrefillRequest with the TextContext
        prefill_request = PrefillRequest(
            id=text_context.request_id,
            context=text_context,
            transfer_engine_name="benchmark_transfer_engine",
            block_ids=list(range(len(tokens))),  # Match actual token count
        )

        prefill_requests.append(prefill_request)

    token_counts = [len(req.context.tokens) for req in prefill_requests]
    print(f"Created {len(prefill_requests)} PrefillRequest objects:")
    print(f"  Token count range: {min(token_counts)} - {max(token_counts)}")
    print(f"  Average token count: {np.mean(token_counts):.1f}")
    print(f"  Token count std dev: {np.std(token_counts):.1f}")

    return prefill_requests


@pytest.mark.parametrize(
    "min_tokens,max_tokens",
    [
        (10, 100),
        (100, 1000),
        (1000, 8000),
        (8000, 40000),
    ],
    ids=[
        "01_small_010-100_tokens",
        "02_medium_100-1000_tokens",
        "03_large_1000-8000_tokens",
        "04_xlarge_8000-40000_tokens",
    ],
)
def test_benchmark_dispatch_service(
    benchmark,  # noqa: ANN001
    min_tokens: int,
    max_tokens: int,
    n_iterations: int = 100,
):
    """
    Benchmark test for dispatch service communication with support for multi-machine testing.

    This test can be run in two modes:
    1. Single-machine benchmarking: Runs performance benchmarks measuring request/response latency
    2. Multi-machine service mode: Keeps services alive for cross-machine communication testing

    Multi-Machine Testing Setup:
    ---------------------------
    To test communication between two machines, run these commands simultaneously:

    Machine 1 (Benchmark Runner):
    ```bash
    bazel --output_base /tmp/worker_1 test SDK/lib/API/python/benchmarks/serve/pipelines:benchmark_dispatch_service \\
        --test_output=all --test_arg=-svv --cache_test_results=no \\
        --test_env=DISPATCHER_HOST="127.0.0.1" --test_env=DISPATCHER_PORT=5555 \\
        --test_env=TARGET_HOST="127.0.0.1" --test_env=TARGET_PORT=5556
    ```

    Machine 2 (Service Provider):
    ```bash
    bazel --output_base /tmp/worker_2 test SDK/lib/API/python/benchmarks/serve/pipelines:benchmark_dispatch_service \\
        --test_output=all --test_arg=-svv --cache_test_results=no \\
        --test_env=DISPATCHER_HOST="127.0.0.1" --test_env=DISPATCHER_PORT=5556 \\
        --test_env=TARGET_HOST="127.0.0.1" --test_env=TARGET_PORT=5555 \\
        --test_env=RUN_BENCHMARK=0
    ```

    Configuration Details:
    - Machine 1: Runs dispatcher on port 5555, sends requests to Machine 2 on port 5556
    - Machine 2: Runs dispatcher on port 5556, receives requests from Machine 1 on port 5555
    - Machine 2 has RUN_BENCHMARK=0, so it acts as a service provider without running benchmarks
    - Both machines should use actual network addresses (not 127.0.0.1) for cross-machine communication

    Environment Variables:
    - DISPATCHER_HOST: IP address where this machine's dispatcher service binds (default: 127.0.0.1)
    - DISPATCHER_PORT: Port where this machine's dispatcher service listens (default: 5555)
    - TARGET_HOST: IP address of the target machine to send requests to (default: 127.0.0.1)
    - TARGET_PORT: Port of the target machine's dispatcher service (default: 5556)
    - RUN_BENCHMARK: If "0"/"false"/"no"/"off", keeps services alive without benchmarking (default: true)

    Test Behavior:
    - Creates PrefillRequest objects with variable token counts within the specified range
    - Sends requests via ZMQ transport to the target dispatcher service
    - Measures round-trip latency for request-response communication
    - Uses different output base directories to avoid bazel cache conflicts between machines

    Args:
        benchmark: pytest-benchmark fixture for running performance measurements
        min_tokens: Minimum number of tokens in generated test contexts (from pytest parametrize)
        max_tokens: Maximum number of tokens in generated test contexts (from pytest parametrize)
        n_iterations: Number of benchmark iterations to run per round (default: 100)

    Raises:
        RuntimeError: If dispatcher service stops unexpectedly during testing
        TimeoutError: If no response is received within the 15-second timeout period
        ValueError: If received payload types don't match expected PrefillRequest/PrefillResponse
    """

    # Start dispatcher service manually (without ensure_dispatcher_service_running)
    global _dispatcher_service, _dispatcher_thread, _service_running

    with _service_lock:
        if (
            not _service_running
            or not _dispatcher_thread
            or not _dispatcher_thread.is_alive()
        ):
            _service_running = True
            _dispatcher_thread = threading.Thread(
                target=_run_dispatcher_service_thread, daemon=True
            )
            _dispatcher_thread.start()

            # Wait for the service to initialize
            time.sleep(1.0)  # Give it more time to start up properly

    # Create and start dispatcher client
    client = _dispatcher_factory.create_client()
    client.start()

    # Verify that the dispatcher service is still running
    with _service_lock:
        if not (
            _service_running
            and _dispatcher_thread
            and _dispatcher_thread.is_alive()
        ):
            raise RuntimeError(
                "Dispatcher service stopped after client start"
            ) from None

    try:
        # Set up request handler to process PrefillRequests and send responses
        @client.request_handler(MessageType.PREFILL_REQUEST)
        def handle_prefill_request(
            payload: PayloadType, reply_context: ReplyContext
        ) -> None:
            # Ensure we received a PrefillRequest
            if not isinstance(payload, PrefillRequest):
                raise ValueError(
                    f"Expected PrefillRequest, got {type(payload)}"
                )

            # Create a mock PrefillResponse with the same request ID and context
            # In a real implementation, this would process the request and generate the response
            mock_transfer_metadata = XferReqData(
                dst_name="benchmark_dst_engine",
                src_name="benchmark_src_engine",
                dst_agent_names=["benchmark_dst_agent"],
                src_agent_names=["benchmark_src_agent"],
                xfer_name="benchmark_xfer",
                xfer_ids=[1],
                src_idxs=[0, 1, 2, 3, 4],
                dst_idxs=[0, 1, 2, 3, 4],
            )
            response_payload = PrefillResponse(
                id=payload.id,
                context=payload.context,  # Echo back the same context
                transfer_metadata=mock_transfer_metadata,
            )
            client.send_reply(
                MessageType.PREFILL_RESPONSE, response_payload, reply_context
            )

        # Set up synchronization for request-response tracking - outside of benchmarked code
        response_received = threading.Event()
        received_response: dict[str, Any] = {}
        current_request_id = None

        # Set up reply handler to receive PrefillResponse messages - outside of benchmarked code
        @client.reply_handler(MessageType.PREFILL_RESPONSE)
        def handle_prefill_response(payload: PayloadType) -> None:
            nonlocal current_request_id
            # Ensure we received a PrefillResponse
            if not isinstance(payload, PrefillResponse):
                raise ValueError(
                    f"Expected PrefillResponse, got {type(payload)}"
                )

            # Verify this is the response for our current request
            if payload.id == current_request_id:
                received_response["payload"] = payload
                received_response["token_count"] = len(payload.context.tokens)
                response_received.set()

        time.sleep(10)

        # Create variable PrefillRequest objects - outside of benchmarked code
        prefill_requests = create_variable_prefill_requests(
            n_iterations, min_tokens, max_tokens
        )

        def send_and_receive(prefill_request: PrefillRequest):
            nonlocal current_request_id
            # Clear the event and response for this benchmark run
            response_received.clear()
            received_response.clear()

            # Set the current request ID for reply handler tracking
            current_request_id = prefill_request.id

            # Send the PrefillRequest via the dispatcher client
            client.send(
                MessageType.PREFILL_REQUEST,
                prefill_request,
                destination_address=TARGET_ADDRESS,
            )

            # Wait for the response with a timeout
            if response_received.wait(timeout=15.0):
                # Verify we received a response for the correct request ID
                if (
                    "payload" in received_response
                    and received_response["payload"].id == current_request_id
                ):
                    return
                else:
                    raise RuntimeError(
                        f"Received response for wrong request ID: expected {current_request_id}"
                    ) from None
            else:
                raise TimeoutError(
                    "No response received within timeout period"
                ) from None

        if RUN_BENCHMARK:
            print(
                f"Running benchmark with {n_iterations} iterations using {len(prefill_requests)} variable PrefillRequest objects (token range: {min_tokens}-{max_tokens}) (RUN_BENCHMARK=true)"
            )

            # Create an iterator to cycle through prefill_requests
            request_index = 0

            def setup():
                nonlocal request_index
                # Get the next prefill_request, cycling through the list
                current_request = prefill_requests[
                    request_index % len(prefill_requests)
                ]
                request_index += 1
                # Return tuple of (args, kwargs) for the benchmark function
                return (current_request,), {}

            # Use setup function to provide different prefill_request for each iteration
            benchmark.pedantic(
                send_and_receive,
                setup=setup,
                rounds=n_iterations,  # Number of benchmark rounds (parameterized)
                iterations=1,  # Number of iterations per round
            )
        else:
            print(
                "Skipping benchmark, keeping client and server alive indefinitely (RUN_BENCHMARK=false)"
            )
            print(f"Dispatcher service running on: {DISPATCHER_BIND_ADDRESS}")
            print(f"Target address configured as: {TARGET_ADDRESS}")
            print("Press Ctrl+C to stop the services")

            try:
                # Keep the client and server alive indefinitely
                while _service_running:
                    time.sleep(1)
                    # Log status periodically to show it's alive
                    if int(time.time()) % 30 == 0:  # Log every 30 seconds
                        print(
                            "Services are still running - press Ctrl+C to stop"
                        )
            except KeyboardInterrupt:
                print("Received KeyboardInterrupt, shutting down services")

    finally:
        # Always stop the client first
        client.stop()
        _stop_dispatcher_service()
