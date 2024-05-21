# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from testing import assert_raises

from max.graph import ops, _testing, Type, Graph, TensorType
from tensor import Tensor, TensorShape


fn test_matmul() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, 1, 2, 3), TensorType(DType.float32, 3, 2)
        ),
    )
    g.output(g[0] @ g[1])
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](
        TensorShape(3, 2), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](TensorShape(1, 2, 2), -2, -3, 1, 6)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_batch_matmul() raises:
    var g = Graph(
        List[Type](
            TensorType(DType.float32, 1, 2, 3), TensorType(DType.float32, 3, 2)
        ),
    )
    g.output(g[0] @ g[1])
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](
        TensorShape(3, 2), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](TensorShape(1, 2, 2), -2, -3, 1, 6)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def test_matmul_dimension_mismatch():
    var graph = Graph(TensorType(DType.float32, 2, 6))

    var matmul_constant_value = Tensor[DType.float32](2, 6)
    matmul_constant_value._to_buffer().fill(0.15)
    var matmul_constant = graph.constant(matmul_constant_value)

    with assert_raises(contains="matrix multiply K dimensions don't match"):
        var matmul = ops.matmul(graph[0], matmul_constant)


def test_matmul_rhs_too_small():
    var graph = Graph(TensorType(DType.float32, 2, 6))

    var matmul_constant_value = Tensor[DType.float32](6)
    matmul_constant_value._to_buffer().fill(0.15)
    var matmul_constant = graph.constant(matmul_constant_value)

    with assert_raises(contains="must have rank at least 2"):
        var matmul = ops.matmul(graph[0], matmul_constant)


def main():
    test_matmul()
    test_batch_matmul()
    test_matmul_dimension_mismatch()
    test_matmul_rhs_too_small()
