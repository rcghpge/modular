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

"""Provides some canonical client data"""

from typing import Any


def simple_openai_request(
    model_name: str = "gpt-3.5-turbo",
    content: str = "Say this is a test!",
    stream: bool = False,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.7,
        "stream": stream,
    }


def simple_openai_stream_request() -> dict[str, Any]:
    """
    A simple streaming request.
    Verify via:
    curl https://api.openai.com/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" -d '{ "model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Say this is a test!"}], "stream": true}'
    """
    return {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say This is a test!"}],
        "stream": "true",
    }


def simple_kserve_request() -> dict[str, Any]:
    return {
        "inputs": [
            {
                "name": "args_0",
                "shape": [1, 1],
                "datatype": "FP32",
                "data": [[1]],
            },
            {
                "name": "args_1",
                "shape": [1, 1],
                "datatype": "FP32",
                "data": [[1]],
            },
        ],
        "outputs": ["output_0"],
    }


def simple_kserve_response() -> dict[str, Any]:
    return {
        "id": "infer-add",
        "model_name": "Add",
        "model_version": "v1.0.0",
        "outputs": [
            {
                "data": [[1.0]],
                "datatype": "FP32",
                "name": "output_0",
                "shape": [1, 1],
            }
        ],
    }
