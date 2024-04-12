# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"

from tensor import Tensor, TensorShape

from max.graph import _testing, Graph, MOTensor, TypeTuple, Dim
from max.graph.ops.elementwise import equal
from testing import assert_raises


fn test_basic() raises:
    var g = Graph(
        TypeTuple(
            MOTensor(DType.float32, 1, 2, 3), MOTensor(DType.float32, 2, 3)
        ),
        MOTensor(DType.float32, 1, 2, 3),
    )
    g.output(g[0] + g[1])
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var y = Tensor[DType.float32](
        TensorShape(2, 3), 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 3), -1.0, 0.0, 0.0, 2.0, 2.0, 4.0
    )

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_literal() raises:
    var g = Graph(
        MOTensor(DType.float32, 1, 2, 3), MOTensor(DType.float32, 1, 2, 3)
    )
    g.output(g[0] - 1.0)
    g.verify()

    var x = Tensor[DType.float32](
        TensorShape(1, 2, 3), -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    var expected = Tensor[DType.float32](
        TensorShape(1, 2, 3), -3.0, -2.0, -1.0, 0.0, 1.0, 2.0
    )

    var actual = _testing.execute_unary(g, x)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_implicit_promotion() raises:
    var g = Graph(
        TypeTuple(MOTensor(DType.int16, 3), MOTensor(DType.float32, 3)),
        MOTensor(DType.float32, 3),
    )
    g.output(g[0] * g[1])
    g.verify()

    var x = Tensor[DType.int16](TensorShape(3), -2, -1, 0)
    var y = Tensor[DType.float32](TensorShape(3), 1.0, 1.0, 3.0)
    var expected = Tensor[DType.float32](TensorShape(3), -2.0, -1.0, 0.0)

    var actual = _testing.execute_binary[
        DType.int16, DType.float32, DType.float32
    ](g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_comparison_results_in_bool() raises:
    var g = Graph(
        TypeTuple(MOTensor(DType.int16, 3), MOTensor(DType.float32, 2, 3)),
        MOTensor(DType.bool, 2, 3),
    )
    g.output(equal(g[0], g[1]))
    g.verify()

    var x = Tensor[DType.int16](TensorShape(3), -2, -1, 0)
    var y = Tensor[DType.float32](TensorShape(2, 3), 1.0, -1.0, 3.0, 7, 2, 0)
    var expected = Tensor[DType.bool](
        TensorShape(2, 3), False, True, False, False, False, True
    )

    var actual = _testing.execute_binary[outtype = DType.bool](g, x, y)
    _testing.assert_tensors_equal(expected, actual)


fn test_bad_implicit_promotion() raises:
    var g = Graph(
        TypeTuple(MOTensor(DType.int16, 3), MOTensor(DType.float16, 3)),
        MOTensor(DType.float32, 3),
    )
    g.output(g[0] + g[1])
    g.verify()

    var x = Tensor[DType.int16](TensorShape(3), -2, -1, 0)
    var y = Tensor[DType.float16](TensorShape(3), 1.0, 1.0, 0.0)
    var expected = Tensor[DType.float32](TensorShape(3), -1.0, 0.0, 0.0)

    var actual = _testing.execute_binary[
        DType.int16, DType.float16, DType.float32
    ](g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_bad_broadcast_dynamic_one() raises:
    var g = Graph(
        TypeTuple(
            MOTensor(DType.float32, Dim.dynamic()),
            MOTensor(DType.float32, Dim.dynamic()),
        ),
        MOTensor(DType.float32, Dim.dynamic()),
    )
    g.output(g[0] + g[1])
    g.verify()

    var x = Tensor[DType.float32](TensorShape(3), -2, -1, 0)
    var y = Tensor[DType.float32](TensorShape(1), 1.0)
    var expected = Tensor[DType.float32](TensorShape(3), -1.0, 0.0, 1.0)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


fn test_matching_dynamic() raises:
    var g = Graph(
        TypeTuple(
            MOTensor(DType.float32, Dim.dynamic()),
            MOTensor(DType.float32, Dim.dynamic()),
        ),
        MOTensor(DType.float32, Dim.dynamic()),
    )
    g.output(g[0] + g[1])
    g.verify()

    var x = Tensor[DType.float32](TensorShape(3), -2, -1, 0)
    var y = Tensor[DType.float32](TensorShape(3), 1.0, 1.0, 1.0)
    var expected = Tensor[DType.float32](TensorShape(3), -1.0, 0.0, 1.0)

    var actual = _testing.execute_binary(g, x, y)
    _testing.assert_tensors_almost_equal(expected, actual, 1e-4)


def main():
    test_basic()
    test_literal()
    test_implicit_promotion()
    test_matching_dynamic()
    test_comparison_results_in_bool()

    # TODO: get the raised exception to capture the message.
    # with assert_raises(contains="unsafe cast from si16 to f16. Insert an explicit cast op if this conversion is wanted."):
    with assert_raises():
        test_bad_implicit_promotion()

    # TODO: Update rmo.rebind_tensor_shape to enable a nice message here.
    with assert_raises(contains="RMO rebind shape check failed"):
        test_bad_broadcast_dynamic_one()
