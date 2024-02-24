# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from tensor import Tensor, TensorShape

from max.graph import _testing, Module, MOTensor, TypeTuple


fn test_matmul() raises:
    var m = Module()
    var g = m.graph(
        TypeTuple(
            MOTensor(DType.float32, 1, 2, 3), MOTensor(DType.float32, 3, 2)
        ),
        MOTensor(DType.float32, 1, 2, 2),
    )
    g.output(g[0] @ g[1])
    m.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](
        TensorShape(3, 2), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](TensorShape(1, 2, 2), -2, -3, 1, 6)

    var actual = _testing.execute_binary(m, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_batch_matmul() raises:
    var m = Module()
    var g = m.graph(
        TypeTuple(
            MOTensor(DType.float32, 1, 2, 3), MOTensor(DType.float32, 3, 2)
        ),
        MOTensor(DType.float32, 1, 2, 2),
    )
    g.output(g[0] @ g[1])
    m.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](
        TensorShape(3, 2), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](TensorShape(1, 2, 2), -2, -3, 1, 6)

    var actual = _testing.execute_binary(m, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_matmul()
    test_batch_matmul()
