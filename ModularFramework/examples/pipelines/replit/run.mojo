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


fn main() raises:
    var args = sys.argv()
    var checkpoint_file = Path(args[1])

    # Generate a graph that does a single forward pass of the replit model.
    print("Building model...")
    var replit = Replit[ReplitCheckpoint, DType.bfloat16](get_default())
    var g = replit.build_graph(
        "replit",
        ReplitCheckpoint(checkpoint_file),
        with_attention_mask=True,
        use_cache=True,
    )

    # Load the graph into the session, which generates the MLIR and runs
    # optimization passes on it.
    print("Compiling...")
    var session = InferenceSession()
    var compiled_model = session.load(g)

    # Set up input and caches, and preprocess the input.
    var input_string = 'def hello():\n  print("hello world")'
    print("Running on input:", input_string)
    alias hf_model_name = "replit/replit-code-v1_5-3b"
    var tokenizer = AutoTokenizer(hf_model_name)
    var prompt = List[String]()
    prompt.append(input_string)
    var tokens = tokenizer.encode(prompt)
    var prompt_length = tokens.shape()[1]

    var k_cache: Tensor[DType.float32]
    var v_cache: Tensor[DType.float32]
    k_cache, v_cache = replit.create_empty_cache()

    # Greedily generate tokens one at a time until the end token is reached or
    # the token length has reached the max.
    print("Output:")
    for n in range(prompt_length, MAX_SEQ_LEN + 1):
        var attention_mask = Tensor[DType.bool](TensorShape(1, n), True)
        var results = execute(
            compiled_model, session, tokens, attention_mask, k_cache, v_cache
        )
        var output = results.get[DType.float32]("output0")
        k_cache = results.get[DType.float32]("output1")
        v_cache = results.get[DType.float32]("output2")
        var argmax = output.argmax(axis=-1).astype[DType.int64]()

        var argmax_length = argmax.dim(1)
        var next_token = argmax[0, argmax_length - 1]
        if tokenizer.is_end_of_text(next_token):
            break
        var next_token_tensor = Tensor[DType.int64](
            TensorShape(1, 1), next_token
        )
        tokens = next_token_tensor
        var tokens_str = tokenizer.decode(next_token_tensor)
        print(tokens_str[0], end="")
    print()


fn execute(
    model: Model,
    session: InferenceSession,
    tokens: Tensor[DType.int64],
    attention_mask: Tensor[DType.bool],
    k_cache: Tensor[DType.float32],
    v_cache: Tensor[DType.float32],
) raises -> TensorMap:
    var input_map = session.new_tensor_map()
    input_map.borrow("input0", tokens)
    input_map.borrow("input1", attention_mask)
    input_map.borrow("input2", k_cache)
    input_map.borrow("input3", v_cache)
    var result_map = model.execute(input_map)
    return result_map^
