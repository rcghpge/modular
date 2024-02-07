# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo -D TEMP_DIR=%T %s

from tensor import Tensor, TensorShape

from max.graph import *


fn test_vector_index_int() raises:
    alias name = "test_vector_index_int"

    var m = Module()
    var g = m.graph(name, MOTensor(DType.int64, 3), MOTensor(DType.int64))
    let arg = g[0]
    g.output(arg[1])

    let x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    let expected = Tensor[DType.int64](TensorShape(), 2)

    let actual = testing.execute_unary[DType.int64, DType.int64](m, name, x)
    testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_index_sym() raises:
    alias name = "test_vector_index_sym"

    var m = Module()
    var g = m.graph(
        name,
        TypeTuple(MOTensor(DType.int64, 3), MOTensor(DType.int64)),
        MOTensor(DType.int64),
    )
    let arg = g[0]
    let idx = g[1]
    g.output(arg[idx])

    let x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    let i = Tensor[DType.int64](TensorShape(), 1)
    let expected = Tensor[DType.int64](TensorShape(), 2)

    let actual = testing.execute_binary[DType.int64, DType.int64](m, name, x, i)
    testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_slice_int() raises:
    alias name = "test_vector_slice_int"

    var m = Module()
    var g = m.graph(name, MOTensor(DType.int64, 3), MOTensor(DType.int64, 2))
    let arg = g[0]
    g.output(arg[1:3])

    let x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    let expected = Tensor[DType.int64](TensorShape(2), 2, 3)

    let actual = testing.execute_unary[DType.int64, DType.int64](m, name, x)
    testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_slice_sym() raises:
    alias name = "test_vector_slice_sym"

    var m = Module()
    var g = m.graph(
        name,
        TypeTuple(MOTensor(DType.int64, 3), MOTensor(DType.int64)),
        MOTensor(DType.int64, Dim.dynamic()),
    )
    let arg = g[0]
    let idx = g[1]
    g.output(arg[idx : idx + 2])

    let x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    let i = Tensor[DType.int64](TensorShape(), 1)
    let expected = Tensor[DType.int64](TensorShape(2), 2, 3)

    let actual = testing.execute_binary[DType.int64, DType.int64](m, name, x, i)
    testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_index_int() raises:
    alias name = "test_matrix_index_int"

    var m = Module()
    var g = m.graph(name, MOTensor(DType.int64, 2, 3), MOTensor(DType.int64, 3))
    let arg = g[0]
    g.output(arg[1])

    let x = Tensor[DType.int64](TensorShape(2, 3), 1, 2, 3, 4, 5, 6)
    let expected = Tensor[DType.int64](TensorShape(3), 4, 5, 6)

    let actual = testing.execute_unary[DType.int64, DType.int64](m, name, x)
    testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_index_sym() raises:
    alias name = "test_matrix_index_sym"

    var m = Module()
    var g = m.graph(
        name,
        TypeTuple(MOTensor(DType.int64, 2, 3), MOTensor(DType.int64)),
        MOTensor(DType.int64, 3),
    )
    let arg = g[0]
    let idx = g[1]
    g.output(arg[idx])

    let x = Tensor[DType.int64](TensorShape(2, 3), 1, 2, 3, 4, 5, 6)
    let i = Tensor[DType.int64](TensorShape(), 1)
    let expected = Tensor[DType.int64](TensorShape(3), 4, 5, 6)

    let actual = testing.execute_binary[DType.int64, DType.int64](m, name, x, i)
    testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_slice_int() raises:
    alias name = "test_matrix_slice_int"

    var m = Module()
    var g = m.graph(
        name,
        MOTensor(DType.int64, 3, 3),
        MOTensor(DType.int64, 2, 3),
    )
    let arg = g[0]
    g.output(arg[1:3])

    let x = Tensor[DType.int64](TensorShape(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9)
    let expected = Tensor[DType.int64](TensorShape(2, 3), 4, 5, 6, 7, 8, 9)

    let actual = testing.execute_unary[DType.int64, DType.int64](m, name, x)
    testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_slice_sym() raises:
    alias name = "test_matrix_slice_sym"

    var m = Module()
    var g = m.graph(
        name,
        TypeTuple(MOTensor(DType.int64, 3, 3), MOTensor(DType.int64)),
        MOTensor(DType.int64, Dim.dynamic(), 3),
    )
    let arg = g[0]
    let idx = g[1]
    g.output(arg[idx : idx + 2])

    let x = Tensor[DType.int64](TensorShape(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9)
    let i = Tensor[DType.int64](TensorShape(), 1)
    let expected = Tensor[DType.int64](TensorShape(2, 3), 4, 5, 6, 7, 8, 9)

    let actual = testing.execute_binary[DType.int64, DType.int64](m, name, x, i)
    testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_slice_double() raises:
    alias name = "test_matrix_slice_sym"

    var m = Module()
    var g = m.graph(
        name,
        TypeTuple(MOTensor(DType.int64, 3, 3), MOTensor(DType.int64)),
        MOTensor(DType.int64, Dim.dynamic(), Dim.dynamic()),
    )
    let arg = g[0]
    let idx = g[1]
    g.output(arg[idx : idx + 2, idx - 1 : idx + 1])

    let x = Tensor[DType.int64](TensorShape(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9)
    let i = Tensor[DType.int64](TensorShape(), 1)
    let expected = Tensor[DType.int64](TensorShape(2, 2), 4, 5, 7, 8)

    let actual = testing.execute_binary[DType.int64, DType.int64](m, name, x, i)
    testing.assert_tensors_equal[DType.int64](expected, actual)


def main():
    test_vector_index_int()
    test_vector_index_sym()
    test_vector_slice_int()
    test_vector_slice_sym()

    test_matrix_index_int()
    test_matrix_index_sym()
    test_matrix_slice_int()
    test_matrix_slice_sym()
    test_matrix_slice_double()
