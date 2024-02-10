# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from tensor import Tensor, TensorShape

from max.graph import testing, Module, MOTensor, TypeTuple


fn test_add_basic() raises:
    alias name = "test_add_basic"

    var m = Module()
    var g = m.graph(
        name,
        TypeTuple(
            MOTensor(DType.float32, 1, 2, 3), MOTensor(DType.float32, 2, 3)
        ),
        MOTensor(DType.float32, 1, 2, 3),
    )
    g.output(g[0] + g[1])
    m.verify()

    let x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    let y = Tensor[DType.float32](
        TensorShape(2, 3), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    let expected = Tensor[DType.float32](
        TensorShape(1, 2, 3), -1.0, 0.0, 0.0, 2.0, 2.0, 4.0
    )

    let actual = testing.execute_binary(m, name, x, y)
    testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_add_literal() raises:
    alias name = "test_add_literal"

    var m = Module()
    var g = m.graph(
        name, MOTensor(DType.float32, 1, 2, 3), MOTensor(DType.float32, 1, 2, 3)
    )
    g.output(g[0] + 1.0)
    m.verify()

    let x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    let expected = Tensor[DType.float32](
        TensorShape(1, 2, 3), -1.0, 0.0, 1.0, 2.0, 3.0, 4.0
    )

    let actual = testing.execute_unary(m, name, x)
    testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_add_basic()
    test_add_literal()
