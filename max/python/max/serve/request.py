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
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from max.serve.telemetry.stopwatch import StopWatch

logger = logging.getLogger("max.serve")


def register_request(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_session(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.request_timer = StopWatch()
        try:
            response: Response = await call_next(request)
        except HTTPException:
            raise  # already wrapped
        except Exception as e:
            logger.exception("Exception in request session : %s", request_id)
            raise HTTPException(
                status_code=500, headers={"X-Request-ID": request_id}
            ) from e
        response.headers["X-Request-ID"] = request_id
        return response
