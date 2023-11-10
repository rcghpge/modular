# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from tensor import TensorShape, TensorSpec

from utils.vector import DynamicVector
from utils._serialize import _serialize_as_tensor


# CHECK: test_tensor_shape
fn test_tensor_shape():
    print("== test_tensor_shape")

    # CHECK: 1
    print(TensorShape(1).__str__())

    # CHECK: 1x2
    print(TensorShape(1, 2).__str__())

    # CHECK: 1x2x3
    print(TensorShape(1, 2, 3).__str__())

    # CHECK: 1x2x3x4
    print(TensorShape(1, 2, 3, 4).__str__())

    # CHECK: 1x2x3x4x5x6
    print(TensorShape(1, 2, 3, 4, 5, 6).__str__())

    # CHECK: 1x2x3x4x5x6x7
    print(TensorShape(1, 2, 3, 4, 5, 6, 7).__str__())

    # CHECK: 1x2x3x4x5x6x7x8x9x10
    print(TensorShape(1, 2, 3, 4, 5, 6, 7, 8, 9, 10).__str__())

    # CHECK: 1048576x2147483648x3298534883328
    print(TensorShape(1048576, 2147483648, 3298534883328).__str__())

    # CHECK: True
    print(TensorShape(1, 2, 3, 4) == TensorShape(1, 2, 3, 4))

    # CHECK: False
    print(TensorShape(1, 2, 3, 4) == TensorShape(1, 4, 5, 1))

    var shape_vec = DynamicVector[Int]()
    for i in range(1, 5):
        shape_vec.push_back(i)

    # CHECK: True
    print(TensorShape(shapes=shape_vec) == TensorShape(1, 2, 3, 4))

    shape_vec._del_old()


# CHECK: test_tensor_spec
fn test_tensor_spec():
    print("== test_tensor_spec")

    # CHECK: 1x2x3xfloat32
    print(TensorSpec(DType.float32, 1, 2, 3).__str__())

    # CHECK: 1x2x3x4x5x6xfloat32
    print(TensorSpec(DType.float32, 1, 2, 3, 4, 5, 6).__str__())

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

    var shape_vec = DynamicVector[Int]()
    for i in range(1, 5):
        shape_vec.push_back(i)

    # CHECK: 1x2x3x4xfloat32
    print(TensorSpec(DType.float32, shape_vec).__str__())

    shape_vec._del_old()

    # Check that dynamic dims work.
    # CHECK: True
    let shape = TensorShape(-9223372036854775808)
    let spec = TensorSpec(DType.float32, shape)
    print(shape == spec.shape)


# CHECK: test_serialize
fn test_serialize():
    print("== test_serialize")

    var spec = TensorSpec(DType.float32, 1)
    let bytes = _serialize_as_tensor(spec)
    # CHECK: True
    print(bytes.num_elements() == sizeof[TensorSpec]())

    # CHECK: True
    print(TensorSpec.from_bytes(bytes.data()) == spec)

    _ = bytes ^


fn main():
    test_tensor_shape()
    test_tensor_spec()
    test_serialize()
