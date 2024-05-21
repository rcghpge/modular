# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo "%s"

import sys
from memory import memset_zero

from max.graph import Graph, TensorType, Type
from max.graph.quantization import BFloat16Encoding
from max.graph._testing import assert_tensors_almost_equal, execute_unary
from tensor import Tensor, TensorShape


def test_quantize_bfloat16():
    @parameter
    if sys.has_neon():
        # TODO(KERN-228): Enable once LLVM bfloat16 emulation matures on ARM.
        return

    # Define hyperparameters.
    alias num_tokens = 32
    alias channels = 16

    # Initialize graph.
    g = Graph(
        "bfloat16_quant",
        in_types=List[Type](TensorType(DType.uint8, num_tokens, 2 * channels)),
        out_types=List[Type](TensorType(DType.uint8, num_tokens, 2 * channels)),
    )

    # Load token embeddings as float32.
    f32_tensor = Tensor[DType.float32].randn(TensorShape(num_tokens, channels))

    # Quantize the float32 token embeddings to bfloat16.
    bfloat16_symbol = g.quantize[BFloat16Encoding](f32_tensor)

    # Add zeros with the weights as a "pseudo identity" op.
    # Otherwise the API/runtime returns an invalid stack address as output.
    # TODO(GRA-498): Fix issue returning mgp.buffer.constant directly.
    g.output(bfloat16_symbol + g[0])

    zeros = Tensor[DType.uint8](TensorShape(num_tokens, 2 * channels))
    memset_zero(zeros.unsafe_ptr(), zeros.num_elements())

    output = execute_unary(g, zeros)

    # Bitcast output to bfloat16 then cast to float32 to verify.
    output = Tensor(
        TensorShape(num_tokens, channels),
        output._steal_ptr().bitcast[DType.bfloat16](),
    ).astype[DType.float32]()

    # Set rtol based on bfloat16 resolution: 0.01.
    assert_tensors_almost_equal(f32_tensor, output, rtol=1e-2)


def main():
    test_quantize_bfloat16()
