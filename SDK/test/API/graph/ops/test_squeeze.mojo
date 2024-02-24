# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from tensor import Tensor, TensorShape

from max.graph import ops, _testing, Dim, Module, MOTensor


fn test_squeeze() raises:
    var m = Module()
    var g = m.graph(
        MOTensor(DType.float32, Dim.dynamic(), 1, 3),
        MOTensor(DType.float32, Dim.dynamic(), 3),
    )
    g.output(ops.squeeze(g[0], 1))
    m.verify()

    var x = Tensor[DType.float32](TensorShape(1, 1, 3), 1, 2, 3)
    var expected = Tensor[DType.float32](TensorShape(1, 3), 1, 2, 3)

    var actual = _testing.execute_unary(m, x)
    _testing.assert_tensors_almost_equal(expected, actual)


def main():
    test_squeeze()
