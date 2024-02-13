# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from tensor import Tensor, TensorShape

from max.graph import ops, _testing as testing, Module, MOTensor


fn test_arg_max() raises:
    alias name = "test_arg_max"

    var m = Module()
    var g = m.graph(
        name, MOTensor(DType.float32, 2, 3), MOTensor(DType.int64, 2, 1)
    )
    g.output(ops.arg_max(g[0], 1))
    m.verify()

    let x = Tensor[DType.float32](
        TensorShape(2, 3), -2.0, -1.0, 0.0, 3.0, 2.0, 1.0
    )
    let expected = Tensor[DType.int64](TensorShape(2, 1), 2, 0)

    let actual = testing.execute_unary[outtype = DType.int64](m, name, x)
    testing.assert_tensors_equal(expected, actual)


fn test_arg_max_neg_axis() raises:
    alias name = "test_arg_max_neg_axis"

    var m = Module()
    var g = m.graph(
        name, MOTensor(DType.float32, 2, 3), MOTensor(DType.int64, 2, 1)
    )
    g.output(ops.arg_max(g[0], -1))
    m.verify()

    let x = Tensor[DType.float32](
        TensorShape(2, 3), -2.0, -1.0, 0.0, 3.0, 2.0, 1.0
    )
    let expected = Tensor[DType.int64](TensorShape(2, 1), 2, 0)

    let actual = testing.execute_unary[outtype = DType.int64](m, name, x)
    testing.assert_tensors_equal(expected, actual)


def main():
    test_arg_max()
    test_arg_max_neg_axis()
