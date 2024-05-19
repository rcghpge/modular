# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from pathlib import Path
import sys

from model.replit import Replit
from weights.replit_checkpoint import ReplitCheckpoint
from weights.hyperparams import get_default
from max.engine import InferenceSession, Model, TensorMap
from max.tensor import Tensor, TensorShape

from tokenizer import AutoTokenizer

alias MAX_SEQ_LEN = 512


def main():
    args = sys.argv()
    checkpoint_file = Path(args[1])

    # Generate a graph that does a single forward pass of the replit model.
    print("Building model...")
    replit = Replit[ReplitCheckpoint, DType.float32](get_default())
    g = replit.build_graph(
        "replit",
        ReplitCheckpoint(checkpoint_file),
        with_attention_mask=True,
        use_cache=True,
    )

    # Load the graph into the session, which generates the MLIR and runs
    # optimization passes on it.
    print("Compiling...")
    session = InferenceSession()
    compiled_model = session.load(g)

    # Set up input and caches, and preprocess the input.
    input_string = 'def hello():\n  print("hello world")'
    print("Running on input:", input_string)
    alias hf_model_name = "replit/replit-code-v1_5-3b"
    bpe_tokenizer = AutoTokenizer(hf_model_name)
    prompt = List[String]()
    prompt.append(input_string)
    tokens = bpe_tokenizer.encode(prompt)
    prompt_length = tokens.shape()[1]

    var k_cache: Tensor[DType.float32]
    var v_cache: Tensor[DType.float32]
    k_cache, v_cache = replit.create_empty_cache()

    # Greedily generate tokens one at a time until the end token is reached or
    # the token length has reached the max.
    print("Output:")
    for n in range(prompt_length, MAX_SEQ_LEN + 1):
        attention_mask = Tensor[DType.bool](TensorShape(1, n), True)
        results = execute(
            compiled_model, session, tokens, attention_mask, k_cache, v_cache
        )
        output = results.get[DType.float32]("output0")
        k_cache = results.get[DType.float32]("output1")
        v_cache = results.get[DType.float32]("output2")
        argmax = output.argmax(axis=-1).astype[DType.int64]()

        argmax_length = argmax.dim(1)
        next_token = argmax[0, argmax_length - 1]
        if bpe_tokenizer.is_end_of_text(next_token):
            break
        next_token_tensor = Tensor[DType.int64](TensorShape(1, 1), next_token)
        tokens = next_token_tensor
        tokens_str = bpe_tokenizer.decode(next_token_tensor)
        print(tokens_str[0], end="")
    print()


def execute(
    model: Model,
    session: InferenceSession,
    tokens: Tensor[DType.int64],
    attention_mask: Tensor[DType.bool],
    k_cache: Tensor[DType.float32],
    v_cache: Tensor[DType.float32],
) -> TensorMap:
    input_map = session.new_tensor_map()
    input_map.borrow("input0", tokens)
    input_map.borrow("input1", attention_mask)
    input_map.borrow("input2", k_cache)
    input_map.borrow("input3", v_cache)
    result_map = model.execute(input_map)
    return result_map^
