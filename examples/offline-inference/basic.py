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

# DOC: max/serve/offline-inference.mdx

from __future__ import annotations

import os

from max.entrypoints.llm import LLM
from max.pipelines import PipelineConfig


def main() -> None:
    model_path = "modularai/Llama-3.1-8B-Instruct-GGUF"
    print(f"Loading model: {model_path}")
    pipeline_config = PipelineConfig(model_path=model_path)
    llm = LLM(pipeline_config)

    prompts = [
        "In the beginning, there was",
        "I believe the meaning of life is",
        "The fastest way to learn python is",
    ]

    print("Generating responses...")
    responses = llm.generate(prompts, max_new_tokens=50)

    for i, (prompt, response) in enumerate(
        zip(prompts, responses, strict=True)
    ):
        print(f"========== Response {i} ==========")
        print(prompt + response)
        print()


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
