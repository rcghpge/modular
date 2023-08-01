# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from IO import print
from DType import DType
from Index import StaticIntTuple
from Tensor import Tensor, TensorShape


# CHECK: test_tensor
fn test_tensor():
    print("== test_tensor")

    let tensor = Tensor[DType.float32](TensorShape(5, 2, 3))

    # CHECK: 5x2x3
    print(tensor.shape().__str__())

    # CHECK: 5x2x3xfloat32
    print(tensor.spec().__str__())

    # CHECK: 5
    print(tensor.spec()[0])

    # CHECK: 2
    print(tensor.spec()[1])

    # CHECK: 3
    print(tensor.spec()[2])

    # CHECK: 0
    print(tensor._compute_linear_offset(StaticIntTuple[3](0, 0, 0)))

    # CHECK: 0
    print(tensor._compute_linear_offset(0, 0, 0))

    # CHECK: 1
    print(tensor._compute_linear_offset(StaticIntTuple[3](0, 0, 1)))

    # CHECK: 1
    print(tensor._compute_linear_offset(0, 0, 1))

    # CHECK: 2
    print(tensor._compute_linear_offset(StaticIntTuple[3](0, 0, 2)))

    # CHECK: 2
    print(tensor._compute_linear_offset(0, 0, 2))

    # CHECK: 3
    print(tensor._compute_linear_offset(StaticIntTuple[3](0, 1, 0)))

    # CHECK: 3
    print(tensor._compute_linear_offset(0, 1, 0))

    # CHECK: 7
    print(tensor._compute_linear_offset(StaticIntTuple[3](0, 2, 1)))

    # CHECK: 7
    print(tensor._compute_linear_offset(0, 2, 1))

    # CHECK: 9
    print(tensor._compute_linear_offset(StaticIntTuple[3](1, 1, 0)))

    # CHECK: 9
    print(tensor._compute_linear_offset(1, 1, 0))

    # CHECK: 27
    print(tensor._compute_linear_offset(StaticIntTuple[3](4, 1, 0)))

    # CHECK: 27
    print(tensor._compute_linear_offset(4, 1, 0))

    # CHECK: 23
    print(tensor._compute_linear_offset(StaticIntTuple[3](3, 1, 2)))

    # CHECK: 23
    print(tensor._compute_linear_offset(3, 1, 2))


fn main():
    test_tensor()
