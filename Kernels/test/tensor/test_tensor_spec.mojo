# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from collections import List

from tensor import TensorShape, TensorSpec
from tensor.tensor import _serialize_as_tensor


# CHECK: test_tensor_shape
fn test_tensor_shape():
    print("== test_tensor_shape")

    # CHECK: 1
    print(TensorShape(1))

    # CHECK: 1
    print(TensorShape((1,)))

    # CHECK: 1x2
    print(TensorShape(1, 2))

    # CHECK: 1x2x3
    print(TensorShape(1, 2, 3))

    # CHECK: 1x2x3x4
    print(TensorShape(1, 2, 3, 4))

    # CHECK: 1x2x3x4x5x6
    print(TensorShape(1, 2, 3, 4, 5, 6))

    # CHECK: 1x2x3x4x5x6x7
    print(TensorShape(1, 2, 3, 4, 5, 6, 7))

    # CHECK: 1x2x3x4x5x6x7x8x9x10
    print(TensorShape(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    # CHECK: 1048576x2147483648x3298534883328
    print(TensorShape(1048576, 2147483648, 3298534883328))

    # CHECK: True
    print(TensorShape(1, 2, 3, 4) == TensorShape(1, 2, 3, 4))

    # CHECK: False
    print(TensorShape(1, 2, 3, 4) == TensorShape(1, 4, 5, 1))

    var shape_vec = List[Int]()
    for i in range(1, 5):
        shape_vec.append(i)

    # CHECK: True
    print(TensorShape(shapes=shape_vec) == TensorShape(1, 2, 3, 4))


# CHECK: test_tensor_spec
fn test_tensor_spec():
    print("== test_tensor_spec")

    # CHECK: 1x2x3xfloat32
    print(str(TensorSpec(DType.float32, 1, 2, 3)))

    # CHECK: 1x2x3x4x5x6xfloat32
    print(str(TensorSpec(DType.float32, 1, 2, 3, 4, 5, 6)))

    # CHECK: True
    print(
        TensorSpec(DType.float32, 1, 2, 3, 4)
        == TensorSpec(DType.float32, 1, 2, 3, 4)
    )

    # CHECK: False
    print(
        TensorSpec(DType.int32, 1, 2, 3, 4)
        == TensorSpec(DType.float32, 1, 2, 3, 4)
    )

    # CHECK: False
    print(
        TensorSpec(DType.float32, 1, 2, 3, 4)
        == TensorSpec(DType.float32, 1, 4, 5, 1)
    )

    var shape_vec = List[Int]()
    for i in range(1, 5):
        shape_vec.append(i)

    # CHECK: 1x2x3x4xfloat32
    print(str(TensorSpec(DType.float32, shape_vec)))

    # Check that dynamic dims work.
    # CHECK: True
    var shape = TensorShape(-9223372036854775808)
    var spec = TensorSpec(DType.float32, shape)
    print(shape == spec.shape)


# CHECK: test_serialize
fn test_serialize():
    print("== test_serialize")

    var spec = TensorSpec(DType.float32, 1)
    var bytes = _serialize_as_tensor(spec)
    # CHECK: True
    print(bytes.num_elements() == sizeof[TensorSpec]())

    # CHECK: True
    print(TensorSpec.from_bytes(bytes.unsafe_ptr()) == spec)

    _ = bytes^


fn main():
    test_tensor_shape()
    test_tensor_spec()
    test_serialize()
