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

import logging
import queue
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Union, cast

import zmq
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
)
from max.pipelines.core import (
    TextAndVisionContext,
    TextContext,
    TextGenerationResponse,
    TextResponse,
    TokenGenerator,
    msgpack_numpy_decoder,
)
from max.pipelines.lib.pipeline import get_paged_manager
from max.profiler import traced
from max.serve.config import Settings
from max.serve.kvcache_agent.dispatcher_base import MessageType
from max.serve.kvcache_agent.dispatcher_client import DispatcherClient
from max.serve.queue.zmq_queue import ZmqPullSocket, ZmqPushSocket
from max.serve.scheduler.base import PrefillRequest, PrefillResponse

from .base import Scheduler
from .queues import STOP_STREAM

logger = logging.getLogger("max.serve")


@dataclass
class DecodeSchedulerConfig:
    """Decode Specific Scheduler Config."""

    max_batch_size_tg: int
    """The maximum number of requests that can be in the token generation batch."""

    max_forward_steps_tg: int
    """The number of tokens to generate for each request in the token generation iteration."""


class DecodeScheduler(Scheduler):
    def __init__(
        self,
        pipeline: TokenGenerator,
        scheduler_config: DecodeSchedulerConfig,
        paged_manager: PagedKVCacheManager,
        *,
        request_zmq_endpoint: str,
        response_zmq_endpoint: str,
        cancel_zmq_endpoint: str,
        zmq_ctx: zmq.Context,
        dispatcher_client: DispatcherClient,
    ) -> None:
        # Initialize Pipeline and Config
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.paged_manager = paged_manager
        self.zmq_ctx = zmq_ctx
        self.dispatcher_client = dispatcher_client

        # Initialize Queues
        self.request_pull_socket = ZmqPullSocket[
            tuple[str, Union[TextContext, TextAndVisionContext]]
        ](
            zmq_ctx,
            zmq_endpoint=request_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(
                tuple[str, Union[TextContext, TextAndVisionContext]]
            ),
        )
        self.response_push_socket = ZmqPushSocket[tuple[str, TextResponse]](
            zmq_ctx=zmq_ctx, zmq_endpoint=response_zmq_endpoint
        )
        self.cancel_pull_socket = ZmqPullSocket[
            tuple[str, Union[TextContext, TextAndVisionContext]]
        ](
            zmq_ctx=zmq_ctx,
            zmq_endpoint=cancel_zmq_endpoint,
            deserialize=msgpack_numpy_decoder(
                tuple[str, Union[TextContext, TextAndVisionContext]]
            ),
        )

        self.dispatcher_client = dispatcher_client
        self.dispatcher_client.register_reply_handler(
            MessageType.PREFILL_RESPONSE, self.handle_prefill_response
        )
        self.dispatcher_client.register_reply_handler(
            MessageType.TRANSFER_ENGINE_RESPONSE,
            self.handle_transfer_engine_response,
        )

        self.preempted_request: queue.Queue[
            tuple[str, Union[TextContext, TextAndVisionContext]]
        ] = queue.Queue()
        self.prefill_responses: dict[str, PrefillResponse] = {}
        self.completed_transfers: set[str] = set()

        # Initialize Scheduler state.
        self.active_batch: OrderedDict[
            str, Union[TextContext, TextAndVisionContext]
        ] = OrderedDict()
        self.available_cache_indices = set(
            range(self.scheduler_config.max_batch_size_tg)
        )
        self.reserved_cache_indices: dict[str, int] = {}

        # Create Transfer Engine
        self.transfer_engine = KVTransferEngine(
            name=f"decode_agent_{uuid.uuid4()}",
            listen_port=8057,
            tensor=self.paged_manager.device_tensors[0],
            total_num_pages=self.paged_manager.total_num_pages,
        )

        # Ensure that prefix caching is enabled.
        if not self.paged_manager.enable_prefix_caching:
            raise ValueError(
                "Prefix Caching must be enabled on the Paged Manager for Decode Scheduling."
            )

    def pull_from_request_socket(
        self,
    ) -> tuple[str, Union[TextContext, TextAndVisionContext]]:
        """Pulls a request from the request socket.

        Returns:
            tuple[str, Union[TextContext, TextAndVisionContext]]: A tuple containing the request ID and input context.

        Raises:
            queue.Empty: If no requests are available.
            zmq.ZMQError: If there is an error receiving from the socket.
        """

        if not self.preempted_request.empty():
            return self.preempted_request.get()

        return self.request_pull_socket.get_nowait()

    @traced
    def handle_transfer_engine_response(
        self, message: KVTransferEngineMetadata
    ) -> None:
        logger.info(f"connecting to remote transfer engine: {message.name}")
        self.transfer_engine.connect(message)

    def handle_prefill_response(self, message: PrefillResponse) -> None:
        """Handles a prefill response from the dispatcher."""
        self.prefill_responses[message.transfer_metadata.xfer_name] = message

    def push_to_response_socket(
        self, responses: list[dict[str, TextResponse]] = [{}]
    ) -> None:
        """Pushes response messages to the response socket.

        Args:
            responses: List of response dictionaries to send, defaults to empty dict.

        Raises:
            zmq.ZMQError: If there is an error sending on the socket.
        """
        self.response_push_socket.put_nowait(responses)

    @traced
    def send_prefill_request(
        self,
        request_id: str,
        data: Union[TextContext, TextAndVisionContext],
        dst_idx: list[int],
    ) -> None:
        """Pushes a request to the prefill socket.

        Args:
            request_id: The ID of the request to send
            data: The Union[TextContext, TextAndVisionContext] containing the request data

        Raises:
            zmq.ZMQError: If there is an error sending on the socket
        """
        # TODO: Handle this dynamically.
        if len(self.transfer_engine.remote_connections) == 0:
            self.dispatcher_client.send(
                MessageType.TRANSFER_ENGINE_REQUEST,
                self.transfer_engine.metadata,
            )

        self.dispatcher_client.send(
            MessageType.PREFILL_REQUEST,
            PrefillRequest(
                id=request_id,
                context=data,
                transfer_engine_name=self.transfer_engine.name,
                block_ids=dst_idx,
            ),
        )

    def reserve_memory_and_send_to_prefill(self) -> None:
        """Continuously pulls requests from the request queue and forwards them to the prefill node.

        Breaks when the request queue is empty. Memory reservation is pending implementation.
        """
        while self.available_cache_indices:
            try:
                # Pop off request queue
                request_id, request_context = self.pull_from_request_socket()
                logger.info("request received from api worker.")

                # Try and Prefetch Memory Eagerly
                #

                # If we pop off a request successfully.
                # Grab new cache index, claim the slot with the paged manager
                # and add it to the reserved_cache_indices.
                cache_seq_id = self.available_cache_indices.pop()
                self.paged_manager.external_claim([cache_seq_id])

                # Ensure request_context uses the appropriate cache seq id
                request_context.unassign_from_cache()
                request_context.assign_to_cache(cache_seq_id)

                # TODO: E2EOPT-269

                # Prefetch memory for Context Encoding eagerly, this only needs to be
                # for one step.
                if not self.paged_manager.prefetch(request_context, 1):
                    # If we don't have enough space in the paged manager
                    # return this to the request queue.
                    self.preempted_request.put((request_id, request_context))
                    self.available_cache_indices.add(cache_seq_id)
                    self.paged_manager.release(cache_seq_id)

                    # Error out here, if we cant prefetch and have no outstanding reserved_cache_indices.
                    # This means it will not be possible to fulfill this request.
                    if not self.reserved_cache_indices:
                        raise RuntimeError(
                            "no cache space reserved, and prefetch is failing. This indicates that the cache does not have enough pages to hold a single request."
                        )

                    # Break out of the loop, we cant add this to our reserved cache indices
                    # or send for prefilling.
                    break

                # If successful, mark as reserved and send to prefill socket.
                self.reserved_cache_indices[request_id] = cache_seq_id

                dst_idx = self.paged_manager.block_manager.get_req_blocks(
                    cache_seq_id
                )

                # Send to the Prefill Node
                self.send_prefill_request(request_id, request_context, dst_idx)

            except queue.Empty:
                # Break loop when no items in queue
                break

            except Exception as e:
                logger.error(e)
                raise e

    def update_batch(self) -> None:
        """Updates the active batch by adding new requests from the decode queue and managing memory prefetching.

        Adds new requests to the batch while cache indices are available. For each request, attempts to prefetch
        required memory. If prefetch fails, handles preemption by returning newer requests to the decode queue.
        """

        # Walk the active batch, and prefetch for all existing items.
        candidate_request_ids = list(self.active_batch.keys())
        for candidate_request_id in candidate_request_ids:
            # If we have already removed the candidate_request, move on
            if candidate_request_id not in self.active_batch:
                break

            # If the request_id is in the active batch, try and prefetch.
            request_context = self.active_batch[candidate_request_id]

            # TODO: Shrink num_steps appropriately.
            num_steps = self.scheduler_config.max_forward_steps_tg
            # If prefetch fails, pre-empt the request and continue evaluating
            # the batch
            if not self.paged_manager.prefetch(request_context, num_steps):
                raise RuntimeError("""
                    Prefetching memory failed for new decode request.
                    This is likely due to memory contention concerns among the batch.
                    Please decrease the batch size and try again.""")

        # Walk all outstanding prefill responses
        # Notifications provides a list of completed XferReqData.xfer_name
        # keyed on remote named (XferReqData.src_name)
        notifications = self.transfer_engine.agent.get_notifs()
        new_completed = {
            completed_transfer_name.decode()
            for remote in notifications
            for completed_transfer_name in notifications[remote]
        }
        self.completed_transfers.update(new_completed)

        # Process ready transfers: intersection of completed transfers
        # and prefill responses received.
        for completed_transfer_name in (
            self.completed_transfers & self.prefill_responses.keys()
        ):
            # Retrieve Prefill Response
            prefill_response = self.prefill_responses.pop(
                completed_transfer_name
            )

            prefill_response.context.unassign_from_cache()
            prefill_response.context.assign_to_cache(
                self.reserved_cache_indices[prefill_response.id]
            )

            # Calculate num_steps
            num_available_steps = (
                prefill_response.context.compute_num_available_steps(
                    prefill_response.context.max_length
                )
            )
            num_steps = (
                self.scheduler_config.max_forward_steps_tg
                if self.scheduler_config.max_forward_steps_tg
                < num_available_steps
                else num_available_steps
            )

            # Prefetch data early, only add to batch if we can prefetch successfully.
            if not self.paged_manager.prefetch(
                prefill_response.context, num_steps
            ):
                self.prefill_responses[completed_transfer_name] = (
                    prefill_response
                )
                continue

            # Add to active batch.
            self.active_batch[prefill_response.id] = prefill_response.context

            # Remove from completed transfers.
            self.completed_transfers.remove(completed_transfer_name)

    @traced
    def calculate_batch_num_steps(self) -> int:
        """Calculate the number of steps to process in the current batch.

        Returns:
            int: Number of steps to process, either max_forward_steps_tg or a smaller value
                based on request max_lengths.

        Raises:
            RuntimeError: If active_batch is empty.
        """
        if not self.active_batch:
            raise RuntimeError(
                "active_batch must contain at least one context to calculate num_steps"
            )

        # Calculate the maximum number of steps for an individual context.
        batch_available_steps = -1
        for data in self.active_batch.values():
            # If any request has no max_length, we should not change num_steps.
            if data.max_length is None:
                return self.scheduler_config.max_forward_steps_tg

            request_available_steps = data.compute_num_available_steps(
                data.max_length
            )
            if request_available_steps > batch_available_steps:
                batch_available_steps = request_available_steps

        if (
            batch_available_steps > 0
            and batch_available_steps
            < self.scheduler_config.max_forward_steps_tg
        ):
            return batch_available_steps

        return self.scheduler_config.max_forward_steps_tg

    @traced
    def stream_responses_to_frontend(
        self, responses: dict[str, TextGenerationResponse]
    ) -> None:
        """Streams text generation responses to the frontend by converting them into a format suitable for streaming.

        Args:
            responses: Dictionary mapping request IDs to their text generation responses.
        """
        if not responses:
            return

        # Convert this to list[dict[str, Any]]
        stream_responses: list[dict[str, TextResponse]] = [{}]
        for request_id, response in responses.items():
            # This will just ensure that there is always a response for each token
            # We add one here, as we need to send a stop sentinel
            while (len(response.tokens) + (1 if response.is_done else 0)) > len(
                stream_responses
            ):
                stream_responses.append({})

            for token_idx, text_response in enumerate(response.tokens):
                stream_responses[token_idx][request_id] = text_response

            if response.is_done:
                stream_responses[len(response.tokens)][request_id] = cast(
                    TextResponse, STOP_STREAM
                )

        self.push_to_response_socket(stream_responses)

    def _handle_terminated_responses(
        self, responses: dict[str, TextGenerationResponse]
    ) -> None:
        """Handles cleanup for completed text generation responses by releasing cache and removing from active batch.

        Args:
            responses: Dictionary mapping request IDs to their text generation responses.
        """
        if responses is None:
            return

        for request_id, response in responses.items():
            if response.is_done:
                # Release from cache, and active batch.
                cache_id = self.active_batch[request_id].cache_seq_id
                self.pipeline.release(self.active_batch[request_id])
                self.available_cache_indices.add(cache_id)
                del self.reserved_cache_indices[request_id]
                del self.active_batch[request_id]

    @traced
    def schedule(self, num_steps: int) -> None:
        """Schedules a batch of requests for token generation and handles the responses.

        Args:
            num_steps: Number of tokens to generate for this batch.
        """
        responses = self.pipeline.next_token(
            self.active_batch, num_steps=num_steps
        )

        self._handle_terminated_responses(responses)
        self.stream_responses_to_frontend(responses)

    def run_iteration(self) -> None:
        """Main scheduling loop that processes decode requests.

        Receives requests, updates batches, and schedules them for processing
        while handling memory management.
        """
        # Eagerly reserve memory and send to prefill worker
        self.reserve_memory_and_send_to_prefill()

        # Update the active decode batch
        self.update_batch()

        # If empty, skip
        if not self.active_batch:
            return

        # Calculate num_steps
        num_steps = self.calculate_batch_num_steps()

        # Schedule Batch
        self.schedule(num_steps)

    def needs_dispatcher_client(self) -> bool:
        return True


def load_decode_scheduler(
    zmq_ctx: zmq.Context,
    settings: Settings,
    pipeline: TokenGenerator,
    max_batch_size_tg: int,
    max_forward_steps_tg: int,
    dispatcher_client: DispatcherClient,
) -> DecodeScheduler:
    # Create Scheduler Config
    scheduler_config = DecodeSchedulerConfig(
        max_batch_size_tg=max_batch_size_tg,
        max_forward_steps_tg=max_forward_steps_tg,
    )

    # Retrieve Paged Manager
    paged_manager = get_paged_manager(pipeline)

    if paged_manager is None:
        raise RuntimeError(
            "A paged KV cache manager must be present to use the DecodeScheduler"
        )

    return DecodeScheduler(
        pipeline=pipeline,
        scheduler_config=scheduler_config,
        paged_manager=paged_manager,
        request_zmq_endpoint=settings.request_zmq_endpoint,
        response_zmq_endpoint=settings.response_zmq_endpoint,
        cancel_zmq_endpoint=settings.cancel_zmq_endpoint,
        zmq_ctx=zmq_ctx,
        dispatcher_client=dispatcher_client,
    )
