# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -I %mobi_pkg_dir -I %test_utils_pkg_dir -D TEMP_DIR=%T "%s"

from tensor import Tensor, TensorShape

from max.graph import *


fn test_add_basic() raises:
    alias name = "test_add_basic"

    var m = Module()
    var g = m.graph(
        name,
        Arity(m.f32(1, 2, 3), m.f32(2, 3)),
        Arity(m.f32(1, 2, 3)),
    )
    g.output(g[0] + g[1])
    m.verify()

    let x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    let y = Tensor[DType.float32](
        TensorShape(1, 2, 3), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
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
        name,
        Arity(m.f32(1, 2, 3)),
        Arity(m.f32(1, 2, 3)),
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
