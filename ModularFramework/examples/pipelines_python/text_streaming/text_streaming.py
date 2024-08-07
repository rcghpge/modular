# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from llama3.llama3 import TokenGenerator


async def stream_text_to_console(model: TokenGenerator, prompt: str):
    context = await model.new_context(prompt)
    prompt_size = context.prompt_size
    max_tokens = context.max_tokens
    print("Beginning text generation...")
    for i in range(prompt_size, max_tokens + 1):
        if i == 10:
            # Artificially cut the sequence short at 10.
            request_id = "end"
        else:
            request_id = str(i)

        response = await model.next_token({request_id: context})
        if request_id not in response:
            break
        print(response[request_id], end="")
    print()
