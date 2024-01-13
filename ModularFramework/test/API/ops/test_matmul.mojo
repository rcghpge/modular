# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -I %mobi_pkg_dir -I %test_utils_pkg_dir -D TEMP_DIR=%T "%s"

from tensor import Tensor, TensorShape

from max.graph import *


fn test_matmul() raises:
    # TODO: test with non-equal dimensions so we don't just accidentally
    # get the broadcasting right :P
    alias name = "test_matmul"

    var m = Module()
    var g = m.graph(
        name,
        Arity(m.f32(1, 2, 3), m.f32(3, 2)),
        Arity(m.f32(1, 2, 2)),
    )
    g.output(g[0] @ g[1])
    m.verify()

    # TODO: We really need tabular initializers, impossible to read a matrix
    # this way.
    let x = Tensor[DType.float32](
        TensorShape(1, 2, 3),
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
    )
    let y = Tensor[DType.float32](
        TensorShape(3, 2),
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
    )
    let expected = Tensor[DType.float32](
        TensorShape(1, 2, 2),
        -2,
        -3,
        1,
        6,
    )

    let actual = testing.execute_binary(m, name, x, y)
    testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_batch_matmul() raises:
    alias name = "test_batch_matmul"

    var m = Module()
    var g = m.graph(
        name,
        Arity(m.f32(1, 2, 3), m.f32(3, 2)),
        Arity(m.f32(1, 2, 2)),
    )
    g.output(g[0] @ g[1])
    m.verify()

    # TODO: We really need tabular initializers, impossible to read a matrix
    # this way.
    let x = Tensor[DType.float32](
        TensorShape(1, 2, 3),
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
    )
    let y = Tensor[DType.float32](
        TensorShape(1, 3, 2),
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
    )
    let expected = Tensor[DType.float32](
        TensorShape(1, 2, 2),
        -2,
        -3,
        1,
        6,
    )

    let actual = testing.execute_binary(m, name, x, y)
    testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_matmul()
    test_batch_matmul()
