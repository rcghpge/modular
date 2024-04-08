# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from tensor import Tensor, TensorShape

from max.graph import _testing, Graph, MOTensor, TypeTuple


fn test_add_basic() raises:
    var g = Graph(
        TypeTuple(
            MOTensor(DType.float32, 1, 2, 3), MOTensor(DType.float32, 2, 3)
        ),
        MOTensor(DType.float32, 1, 2, 3),
    )
    g.output(g[0] + g[1])
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](
        TensorShape(2, 3), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 3), -1.0, 0.0, 0.0, 2.0, 2.0, 4.0
    )

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_add_literal() raises:
    var g = Graph(
        MOTensor(DType.float32, 1, 2, 3), MOTensor(DType.float32, 1, 2, 3)
    )
    g.output(g[0] + 1.0)
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 3), -1.0, 0.0, 1.0, 2.0, 3.0, 4.0
    )

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_add_basic()
    test_add_literal()
