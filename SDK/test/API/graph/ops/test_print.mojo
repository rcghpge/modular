# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s" | FileCheck %s

from max.graph import _testing, Dim, Graph, TensorType, Symbol
from tensor import Tensor, TensorShape


fn test_print() raises:
    var g = Graph(TensorType(DType.float32, Dim.dynamic(), Dim.dynamic(), 2))
    g[0].print()
    g.output(List[Symbol]())
    g.verify()

    var x = Tensor[DType.float32](TensorShape(2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8)

    # CHECK{LITERAL}: debug_tensor_cpu = tensor([[[1.0000, 2.0000],
    # CHECK{LITERAL}: [3.0000, 4.0000]],
    # CHECK{LITERAL}: [[5.0000, 6.0000],
    # CHECK{LITERAL}: [7.0000, 8.0000]]], dtype=f32, shape=[2,2,2])
    _ = _testing.execute_base(g, x)


def main():
    test_print()
