# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from IO import print
from tensor import TensorSpec, TensorShape


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


# CHECK: test_tensor_spec
fn test_tensor_spec():
    print("== test_tensor_spec")

    # CHECK: 1x2x3xfloat32
    print(TensorSpec(DType.float32, 1, 2, 3).__str__())

    # CHECK: 1x2x3x4x5x6xfloat32
    print(TensorSpec(DType.float32, 1, 2, 3, 4, 5, 6).__str__())


fn main():
    test_tensor_shape()
    test_tensor_spec()
