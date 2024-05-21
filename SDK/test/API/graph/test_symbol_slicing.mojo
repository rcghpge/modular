# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo %s

from max.graph import _testing, ops, Dim, Graph, TensorType, Type
from tensor import Tensor, TensorShape


fn test_vector_index_int() raises:
    var g = Graph(TensorType(DType.int64, 3))
    var arg = g[0]
    g.output(arg[1])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var expected = Tensor[DType.int64](TensorShape(), 2)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_index_sym() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(), 2)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_index_neg_one() raises:
    var g = Graph(TensorType(DType.int64, 1, Dim.dynamic()))
    var arg = g[0]
    g.output(ops.shape_of(arg)[-1])

    var x = Tensor[DType.int64](TensorShape(1, 3), 1, 2, 3)
    var expected = Tensor[DType.int64](TensorShape(), 3)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_slice_int() raises:
    var g = Graph(TensorType(DType.int64, 3))
    var arg = g[0]
    g.output(arg[1:3])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var expected = Tensor[DType.int64](TensorShape(2), 2, 3)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_vector_slice_sym() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx : idx + 2])

    var x = Tensor[DType.int64](TensorShape(3), 1, 2, 3)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(2), 2, 3)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_index_int() raises:
    var g = Graph(TensorType(DType.int64, 2, 3))
    var arg = g[0]
    g.output(arg[1])

    var x = Tensor[DType.int64](TensorShape(2, 3), 1, 2, 3, 4, 5, 6)
    var expected = Tensor[DType.int64](TensorShape(3), 4, 5, 6)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_index_sym() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 2, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx])

    var x = Tensor[DType.int64](TensorShape(2, 3), 1, 2, 3, 4, 5, 6)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(3), 4, 5, 6)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_slice_int() raises:
    var g = Graph(TensorType(DType.int64, 3, 3))
    var arg = g[0]
    g.output(arg[1:3])

    var x = Tensor[DType.int64](TensorShape(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var expected = Tensor[DType.int64](TensorShape(2, 3), 4, 5, 6, 7, 8, 9)

    var actual = _testing.execute_unary[DType.int64, DType.int64](g, x)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_slice_sym() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx : idx + 2])

    var x = Tensor[DType.int64](TensorShape(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(2, 3), 4, 5, 6, 7, 8, 9)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


fn test_matrix_slice_double() raises:
    var g = Graph(
        List[Type](TensorType(DType.int64, 3, 3), TensorType(DType.int64)),
    )
    var arg = g[0]
    var idx = g[1]
    g.output(arg[idx : idx + 2, idx - 1 : idx + 1])

    var x = Tensor[DType.int64](TensorShape(3, 3), 1, 2, 3, 4, 5, 6, 7, 8, 9)
    var i = Tensor[DType.int64](TensorShape(), 1)
    var expected = Tensor[DType.int64](TensorShape(2, 2), 4, 5, 7, 8)

    var actual = _testing.execute_binary[DType.int64](g, x, i)
    _testing.assert_tensors_equal[DType.int64](expected, actual)


def main():
    test_vector_index_int()
    test_vector_index_neg_one()
    test_vector_index_sym()
    test_vector_slice_int()
    test_vector_slice_sym()

    test_matrix_index_int()
    test_matrix_index_sym()
    test_matrix_slice_int()
    test_matrix_slice_sym()
    test_matrix_slice_double()
