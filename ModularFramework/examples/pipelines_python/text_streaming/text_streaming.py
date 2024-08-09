# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from max.serve.pipelines.llm_pipeline_serving import TokenGenerator


async def stream_text_to_console(model: TokenGenerator, prompt: str):
    context = await model.new_context(prompt)
    prompt_size = context.prompt_size
    max_tokens = context.max_tokens

    # Start with the initial prompt.
    print(context.prompt, end="", flush=True)

    # Note: assume a single request for now.
    request_id = 0
    for _ in range(prompt_size, max_tokens + 1):
        response = await model.next_token({request_id: context})
        if request_id not in response:
            break
        print(response[request_id], end="", flush=True)
    print()
