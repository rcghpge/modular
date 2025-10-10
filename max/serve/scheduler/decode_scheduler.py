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
import queue
import time
import uuid
from collections import OrderedDict

from max.interfaces import (
    MAXPullQueue,
    MAXPushQueue,
    RequestID,
    Scheduler,
    SchedulerResult,
    TextGenerationInputs,
    TextGenerationOutput,
)
from max.interfaces.queue import BackgroundQueueDrainer, drain_queue
from max.nn.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManager,
    TransferReqData,
)
from max.pipelines.core import TextAndVisionContext, TextContext
from max.pipelines.lib import PipelineConfig, TextGenerationPipelineType
from max.pipelines.lib.pipeline import get_paged_manager
from max.profiler import Tracer, traced
from max.serve.config import Settings
from max.serve.scheduler.base import (
    CancelRequest,
    PrefillRequest,
    PrefillResponse,
)
from max.serve.scheduler.di_dispatchers import DecodeDispatcherClientV2

from .base import SchedulerProgress
from .text_batch_constructor import (
    TextBatchConstructor,
    TokenGenerationSchedulerConfig,
)
from .utils import (
    SchedulerLogger,
    add_newly_encoded_reqs_to_tg_batch,
    release_terminated_requests,
)

logger = logging.getLogger("max.serve")


class DecodeScheduler(Scheduler):
    def __init__(
        self,
        pipeline: TextGenerationPipelineType[TextContext],
        scheduler_config: TokenGenerationSchedulerConfig,
        paged_manager: PagedKVCacheManager,
        *,
        request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[TextGenerationOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        dispatcher: DecodeDispatcherClientV2,
        offload_queue_draining: bool = False,
    ) -> None:
        # Initialize Pipeline and Config
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.paged_manager = paged_manager

        # Initialize Queues
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        self.dispatcher = dispatcher

        # Initialize Scheduler state.
        self.pending_reqs: OrderedDict[RequestID, TextContext] = OrderedDict()
        self.prefill_reqs: dict[RequestID, TextContext] = {}
        self.inflight_transfers: dict[RequestID, TransferReqData] = {}

        # Create Transfer Engine
        self.transfer_engine = KVTransferEngine(
            name=f"decode_agent_{uuid.uuid4()}",
            tensors=self.paged_manager.device_tensors,
            total_num_pages=self.paged_manager.total_num_pages,
        )

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            paged_cache=paged_manager,
        )
        self.scheduler_logger = SchedulerLogger()
        # None corresponds to the default destination address.
        # TODO: delete the default destination address.
        self.remote_endpoints: set[str | None] = set()

        # We are parameterizing the offload of queue draining to allow for
        # the use case where we want to drain the queue in the main thread.
        # This is useful for debugging and testing purposes.
        self._queue_drainer: (
            BackgroundQueueDrainer[TextContext | TextAndVisionContext] | None
        ) = None
        if offload_queue_draining:
            # Initialize the background queue drainer
            self._queue_drainer = BackgroundQueueDrainer[
                TextContext | TextAndVisionContext
            ](
                self.request_queue,
                max_items_per_drain=self.scheduler_config.max_batch_size_tg * 2,
            )

    @traced
    def handle_transfer_engine_response(
        self, message: KVTransferEngineMetadata
    ) -> None:
        logger.debug(f"connecting to remote transfer engine: {message.name}")
        self.transfer_engine.connect(message)

    def handle_prefill_response(self, message: PrefillResponse) -> None:
        """Handles a prefill response from the dispatcher."""
        # Update the context with the generated token
        request_id = message.id
        context = self.prefill_reqs[request_id]
        context.update(message.generated_token_id)

        # Send singular token to the API process
        output = context.to_generation_output()
        self.response_queue.put_nowait(
            {request_id: SchedulerResult.create(output)}
        )

        self.inflight_transfers[request_id] = message.transfer_metadata

    @traced
    def send_prefill_request(
        self,
        request_id: RequestID,
        data: TextContext,
        dst_idxs: list[int],
    ) -> None:
        """Pushes a request to the prefill socket.

        Args:
            request_id: The ID of the request to send
            data: The context containing the request data

        Raises:
            zmq.ZMQError: If there is an error sending on the socket
        """

        if data.target_endpoint not in self.remote_endpoints:
            self.dispatcher.send_request_nowait(
                self.transfer_engine.metadata,
                data.target_endpoint,
            )
            self.remote_endpoints.add(data.target_endpoint)

        assert data.needs_ce, (
            f"Invalid Context: Expected needs_ce to be True. Found: {data}"
        )

        # Set dst_idx to -1 to denote pages which the decode already has due to
        # prefix caching.
        for i in range(data.start_idx // self.paged_manager.page_size):
            dst_idxs[i] = -1

        self.dispatcher.send_request_nowait(
            PrefillRequest(
                id=request_id,
                context=data,
                transfer_engine_name=self.transfer_engine.name,
                block_ids=dst_idxs,
            ),
            data.target_endpoint,
        )

    def reserve_memory_and_send_to_prefill(self) -> None:
        """Continuously pulls requests from the request queue and forwards them to the prefill node."""
        if self._queue_drainer is not None:
            self._queue_drainer.start_draining()
            items = self._queue_drainer.retrieve_items()
        else:
            items = drain_queue(
                self.request_queue,
                max_items=self.scheduler_config.max_batch_size_tg * 2,
            )

        for context in items:
            self.pending_reqs[context.request_id] = context

        while (
            self.pending_reqs
            and (len(self.batch_constructor.tg_reqs) + len(self.prefill_reqs))
            < self.scheduler_config.max_batch_size_tg
            and (
                self.paged_manager is None
                or self.paged_manager.free_blocks_pct > 0.1
            )
        ):
            # Pop off request queue
            context = next(iter(self.pending_reqs.values()))
            req_id = context.request_id
            del self.pending_reqs[req_id]

            # Claim the slot with the paged manager
            if not self.paged_manager.contains(req_id):
                self.paged_manager.external_claim(req_id)

            # Prefetch memory for Context Encoding eagerly, this only needs to be
            # for one step.
            if not self.paged_manager.maybe_reserve(context, 1):
                # If we don't have enough space in the paged manager
                # return this to the request queue.
                self.pending_reqs[req_id] = context
                self.pending_reqs.move_to_end(req_id, last=False)
                self.paged_manager.release(req_id)
                break

            # Send to the Prefill Node
            dst_idxs = self.paged_manager.block_manager.get_req_blocks(req_id)
            self.prefill_reqs[req_id] = context
            self.send_prefill_request(req_id, context, dst_idxs)

    def _handle_cancelled_requests(self) -> None:
        while True:
            try:
                for request_id in self.cancel_queue.get_nowait():
                    # Remove it from the active batch.
                    if request_id in self.batch_constructor.tg_reqs:
                        del self.batch_constructor.tg_reqs[request_id]

                        # Send the cancelled result back to the response q
                        self.response_queue.put_nowait(
                            {request_id: SchedulerResult.cancelled()}
                        )

                    # If it is pending prefill, remove the pending request.
                    elif request_id in self.prefill_reqs:
                        # Remove from pending requests.
                        del self.prefill_reqs[request_id]

                        # Send a cancel request to the prefill node
                        self.dispatcher.send_request_nowait(
                            CancelRequest(id=request_id)
                        )

                        # Send the cancelled result back to the response q
                        self.response_queue.put_nowait(
                            {request_id: SchedulerResult.cancelled()}
                        )

                    else:
                        logger.debug(
                            f"cancel request received on decode node for {request_id} not in pending or active batch."
                        )

            except queue.Empty:
                break

    def check_for_completed_transfers(self) -> None:
        """Updates the active batch by adding new requests from the decode queue and managing memory prefetching.

        Adds new requests to the batch up to the maximum batch size. For each request, attempts to prefetch
        required memory. If prefetch fails, handles preemption by returning newer requests to the decode queue.
        """

        request_ids = list(self.inflight_transfers.keys())
        for request_id in request_ids:
            transfer_metadata = self.inflight_transfers[request_id]

            # Transfer is not complete, skip.
            if not self.transfer_engine.is_complete(transfer_metadata):
                continue

            # Cleanup the transfer.
            del self.inflight_transfers[request_id]
            self.transfer_engine.cleanup_transfer(transfer_metadata)

            # When cancelled, the request is removed from prefill_reqs
            # therefore the request should only be added to the active_batch
            # if it is still in prefill_reqs.
            if request_id not in self.prefill_reqs:
                continue

            # Remove from pending prefill requests and add to TG requests.
            context = self.prefill_reqs.pop(request_id)
            self.batch_constructor.tg_reqs[request_id] = context

        # Manage for cancelled requests
        self._handle_cancelled_requests()

    @traced
    def schedule(self, inputs: TextGenerationInputs[TextContext]) -> int:
        """Schedules a batch of requests for token generation and handles the responses.

        Args:
            inputs: The inputs containing the batch of requests to schedule.
        """
        assert len(inputs.batches) > 0
        responses = self.pipeline.execute(inputs)

        add_newly_encoded_reqs_to_tg_batch(
            inputs.batch,
            responses,
            self.batch_constructor,
        )

        # remove terminated requests from the batch
        num_terminated_reqs = release_terminated_requests(
            responses,
            self.pipeline,
            self.batch_constructor.tg_reqs,
        )

        # send the responses to the API process
        self.response_queue.put_nowait(
            {
                req_id: SchedulerResult.create(response)
                for req_id, response in responses.items()
            }
        )

        return num_terminated_reqs

    def run_iteration(self) -> SchedulerProgress:
        """Main scheduling loop that processes decode requests.

        Receives requests, updates batches, and schedules them for processing
        while handling memory management.

        Returns:
            SchedulerProgress: Indicates whether work was performed in this iteration.
        """
        while True:
            try:
                reply = self.dispatcher.recv_reply_nowait()
            except queue.Empty:
                break
            if isinstance(reply, KVTransferEngineMetadata):
                self.handle_transfer_engine_response(reply)
            elif isinstance(reply, PrefillResponse):
                self.handle_prefill_response(reply)
            else:
                raise ValueError(f"Invalid reply type: {reply}")

        # Eagerly reserve memory and send to prefill worker
        self.reserve_memory_and_send_to_prefill()

        # Update the active decode batch
        self.check_for_completed_transfers()

        # Construct the batch to execute
        t0 = time.monotonic()
        inputs = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # If the batch is empty, skip
        if len(inputs.batch) == 0:
            return SchedulerProgress.NO_PROGRESS

        # Schedule the batch
        t0 = time.monotonic()
        with Tracer(f"_schedule({inputs})"):
            num_terminated_reqs = self.schedule(inputs)
        t1 = time.monotonic()
        batch_execution_time_s = t1 - t0

        # Log batch metrics
        self.scheduler_logger.log_metrics(
            sch_config=self.scheduler_config,
            inputs=inputs,
            paged_cache=self.paged_manager,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=len(self.pending_reqs) + len(self.prefill_reqs),
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=self.batch_constructor.total_preemption_count,
        )

        return SchedulerProgress.MADE_PROGRESS


def load_decode_scheduler(
    pipeline: TextGenerationPipelineType[TextContext],
    pipeline_config: PipelineConfig,
    request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
    response_queue: MAXPushQueue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ],
    cancel_queue: MAXPullQueue[list[RequestID]],
    settings: Settings,
) -> DecodeScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config
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
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        dispatcher=DecodeDispatcherClientV2(
            bind_addr=settings.dispatcher_config.transport_config.bind_address,
            default_dest_addr=settings.dispatcher_config.transport_config.default_destination_address,
        ),
        offload_queue_draining=pipeline_config.experimental_background_queue,
    )
