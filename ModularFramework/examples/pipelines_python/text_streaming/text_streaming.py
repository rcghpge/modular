# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.serve.pipelines.llm_pipeline_serving import TokenGenerator
from typing import Optional
from utils import TextGenerationMetrics


async def stream_text_to_console(
    model: TokenGenerator, prompt: str, metrics: Optional[TextGenerationMetrics]
):
    context = await model.new_context(prompt)
    prompt_size = context.prompt_size
    max_tokens = context.max_tokens

    # Start with the initial prompt.
    print(context.prompt, end="", flush=True)
    if metrics:
        metrics.prompt_size = prompt_size
        metrics.signpost("begin_generation")

    # Note: assume a single request for now.
    request_id = 0
    for i in range(prompt_size, max_tokens + 1):
        response = await model.next_token({request_id: context})
        if request_id not in response:
            break
        if metrics:
            if i == prompt_size:
                metrics.signpost("first_token")
            metrics.new_token()
        print(response[request_id], end="", flush=True)
    if metrics:
        metrics.signpost("end_generation")
    print()
