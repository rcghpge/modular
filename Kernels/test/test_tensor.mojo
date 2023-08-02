# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from DType import DType
from Index import StaticIntTuple, Index
from IO import print
from Random import rand, seed
from Range import range
from Tensor import Tensor, TensorShape
from Math import sqrt
from SIMD import Float64


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

    # CHECK: 3
    print(tensor.dim(2))


# CHECK: test_tensor_indexing
fn test_tensor_indexing():
    print("== test_tensor_indexing")

    let tensor = Tensor[DType.float32](TensorShape(5, 2, 3))

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


# CHECK: test_tensor_indexing2
fn test_tensor_indexing2():
    print("== test_tensor_indexing2")
    let tensor = Tensor[DType.float32](TensorShape(30, 20, 10))

    # CHECK: 223
    print(tensor._compute_linear_offset(1, 2, 3))


# CHECK: test_tensor_fill
fn test_tensor_fill():
    print("== test_tensor_fill")

    var tensor = Tensor[DType.float32](TensorShape(4, 5, 6))

    for i in range(4):
        for j in range(5):
            for k in range(6):
                tensor[Index(i, j, k)] = tensor._compute_linear_offset(i, j, k)

    # CHECK-NOT: ERROR
    for i in range(4):
        for j in range(5):
            for k in range(6):
                if tensor[i, j, k] != (i * 5 + j) * 6 + k:
                    print("ERROR: invalid index fill")


# CHECK: test_tensor_rand
fn test_tensor_rand():
    print("== test_tensor_rand")

    let tensor = rand[DType.float32](3, 3, 3)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                print(tensor[i, j, k])


# CHECK: test_euclidean_distance
fn test_euclidean_distance():
    print("== test_euclidean_distance")

    let num_elements = 100

    var a = Tensor[DType.float64](num_elements)
    var b = Tensor[DType.float64](num_elements)

    for i in range(num_elements):
        a[i] = i
        b[i] = num_elements - i

    var accum = Float64(0)
    for i in range(num_elements):
        let dist = a[i] - b[i]
        accum += dist * dist

    # CHECK: 577.408
    print(sqrt(accum))


fn main():
    test_tensor()
    test_tensor_indexing()
    test_tensor_indexing2()
    test_tensor_fill()
    test_tensor_rand()
    test_euclidean_distance()
