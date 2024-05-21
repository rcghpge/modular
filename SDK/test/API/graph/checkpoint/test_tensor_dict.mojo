# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo "%s"


import testing
from max.graph import _testing
from max.graph.checkpoint import TensorDict
from tensor import Tensor, TensorShape


fn test_simple() raises:
    var tensors = TensorDict()
    tensors.set("x", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
    tensors.set("y", Tensor[DType.float32](TensorShape(10, 5), -1.23))
    _testing.assert_equal(2, len(tensors))
    var x = tensors["x"]
    _testing.assert_tensors_equal(
        x.to_tensor[DType.int32](),
        Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4),
    )

    var y = tensors["y"]
    _testing.assert_tensors_equal(
        y.to_tensor[DType.float32](),
        Tensor[DType.float32](TensorShape(10, 5), -1.23),
    )

    with testing.assert_raises():
        _ = tensors["z"]


fn test_overwrite() raises:
    var tensors = TensorDict()
    tensors.set("x", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
    tensors.set("x", Tensor[DType.float32](TensorShape(10, 5), -1.23))
    _testing.assert_equal(1, len(tensors))

    var x = tensors["x"]
    _testing.assert_tensors_equal(
        x.to_tensor[DType.float32](),
        Tensor[DType.float32](TensorShape(10, 5), -1.23),
    )

    with testing.assert_raises():
        _ = tensors["y"]


fn test_empty_key() raises:
    var tensors = TensorDict()
    tensors.set("", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))

    var t = tensors[""]
    _testing.assert_tensors_equal(
        t.to_tensor[DType.int32](),
        Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4),
    )


def main():
    test_simple()
    test_overwrite()
    test_empty_key()
