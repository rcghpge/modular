# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -D TEMP_DIR=%T "%s"

from tensor import Tensor, TensorShape

from max.graph import ops, testing, Dim, Module, MOTensor


fn test_squeeze() raises:
    alias name = "test_squeeze"

    var m = Module()
    var g = m.graph(
        name,
        MOTensor(DType.float32, Dim.dynamic(), 1, 3),
        MOTensor(DType.float32, Dim.dynamic(), 3),
    )
    g.output(ops.squeeze(g[0], 1))
    m.verify()

    let x = Tensor[DType.float32](TensorShape(1, 1, 3), 1, 2, 3)
    let expected = Tensor[DType.float32](TensorShape(1, 3), 1, 2, 3)

    let actual = testing.execute_unary(m, name, x)
    testing.assert_tensors_almost_equal(expected, actual)


def main():
    test_squeeze()
