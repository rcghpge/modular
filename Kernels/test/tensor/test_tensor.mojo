# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota, sqrt
from random import rand, seed

from tensor import Tensor, TensorShape

from utils.index import Index, StaticIntTuple


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


# CHECK: test_tensor_init
fn test_tensor_init():
    print("== test_tensor_init")

    let tensor = Tensor[DType.float32](TensorShape(1, 2, 3))

    # CHECK: 0.0
    print(tensor[0, 0, 0])

    let t2 = Tensor[DType.float32](
        TensorShape(1, 2, 3), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
    )

    # CHECK: 0.6
    print(t2[0, 1, 2])


# CHECK-LABEL: test_tensor_equality
fn test_tensor_equality():
    print("== test_tensor_equality")

    var t1 = Tensor[DType.float32](TensorShape(2, 3))
    var t2 = Tensor[DType.float32](TensorShape(2, 2))
    var t3 = Tensor[DType.float32](TensorShape(2, 3))

    for i in range(2):
        for j in range(3):
            t1[Index(i, j)] = 1
            t3[Index(i, j)] = 3

    for i in range(2):
        for j in range(2):
            t2[Index(i, j)] = 2

    # CHECK: True
    print(t1 == t1)
    # CHECK: True
    print(t2 == t2)

    # CHECK: False
    print(t1 == t2)
    # CHECK: True
    print(t1 != t2)

    # CHECK: False
    print(t1 == t3)
    # CHECK: True
    print(t1 != t3)

    for i in range(2):
        for j in range(3):
            t3[Index(i, j)] = 1

    # CHECK: True
    print(t1 == t3)
    # CHECK: False
    print(t1 != t3)


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


# CHECK: test_image_to_grayscale
# COM: This example comes from the change log.
fn test_image_to_grayscale():
    print("== test_image_to_grayscale")

    let height = 256
    let width = 256
    let channels = 3

    # Create the tensor of dimensions height, width, channels and fill.
    var image = Tensor[DType.float32](height, width, channels)
    let num_elements = image.num_elements()
    for x in range(height):
        for y in range(width):
            for c in range(channels):
                image[Index(x, y, c)] = (
                    Float32((x * width + y) * channels + c) / num_elements
                )

    # Declare the grayscale image.
    var gray_scale_image = Tensor[DType.float32](height, width)

    # Perform the RGB to grayscale transform.
    for y in range(height):
        for x in range(width):
            let r = image[y, x, 0]
            let g = image[y, x, 1]
            let b = image[y, x, 2]
            gray_scale_image[Index(y, x)] = 0.299 * r + 0.587 * g + 0.114 * b

    # CHECK: 4.1453{{.*}}e-06
    print(gray_scale_image[0, 0])
    # CHECK: 0.0392192
    print(gray_scale_image[10, 10])
    # CHECK: 0.38676
    print(gray_scale_image[99, 3])
    # CHECK: 0.99998
    print(gray_scale_image[255, 255])


# CHECK-LABEL: test_pow
fn test_pow():
    print("== test_pow")
    let tensor = Tensor[DType.float32](2, 2)
    iota(tensor.data(), tensor.num_elements())
    let tensor_pow_2 = tensor**2
    # CHECK{LITERAL}: Tensor([[0.0, 1.0],
    # CHECK{LITERAL}: [2.0, 3.0]]
    print(tensor)
    # CHECK{LITERAL}: Tensor([[0.0, 1.0],
    # CHECK{LITERAL}: [4.0, 9.0]]
    print(tensor_pow_2)

    let tensor_pow_0 = tensor**0
    # CHECK{LITERAL}: Tensor([[1.0, 1.0],
    # CHECK{LITERAL}: [1.0, 1.0]]
    print(tensor_pow_0)

    var tensor_copy = tensor
    tensor_copy **= 2
    # CHECK{LITERAL}: Tensor([[0.0, 1.0],
    # CHECK{LITERAL}: [2.0, 3.0]]
    print(tensor)
    # CHECK{LITERAL}: Tensor([[0.0, 1.0],
    # CHECK{LITERAL}: [4.0, 9.0]]
    print(tensor_copy)

    var tensor_pow_neg1 = tensor
    tensor_pow_neg1[0] = 1.0
    tensor_pow_neg1 **= -1
    # CHECK{LITERAL}: Tensor([[1.0, 1.0],
    # CHECK: [0.5, 0.333{{.*}}]]
    print(tensor_pow_neg1)

    let tensor_large = Tensor[DType.float32](65)
    iota(tensor_large.data(), tensor_large.num_elements())
    # CHECK{LITERAL}: Tensor([[0.0, 1.0, 4.0, ..., 3844.0, 3969.0, 4096.0]]
    print(tensor_large**2)


# CHECK-LABEL: test_print
fn test_print():
    print("== test_print")
    # CHECK{LITERAL}: Tensor([[[0, 0, 0],
    # CHECK{LITERAL}: [0, 0, 0]],
    # CHECK{LITERAL}: [[0, 0, 0],
    # CHECK{LITERAL}: [0, 0, 0]]], dtype=index, shape=2x2x3)
    let tensor = Tensor[DType.index](2, 2, 3)
    print(tensor)


# CHECK-LABEL: test_print_small
fn test_print_small():
    print("== test_print_small")
    # CHECK{LITERAL}: Tensor([[[0, 1, 2],
    # CHECK{LITERAL}: [3, 4, 5]],
    # CHECK{LITERAL}: [[6, 7, 8],
    # CHECK{LITERAL}: [9, 10, 11]]], dtype=index, shape=2x2x3)
    let tensor = Tensor[DType.index](2, 2, 3)
    iota(tensor.data(), tensor.num_elements())
    print(tensor.__str__())


# CHECK-LABEL: test_print_large
fn test_print_large():
    print("== test_print_large")
    # CHECK{LITERAL}: Tensor([[[0, 1, 2, ..., 330, 331, 332],
    # CHECK-NEXT: [333, 334, 335, ..., 663, 664, 665],
    # CHECK-NEXT: [666, 667, 668, ..., 996, 997, 998],
    # CHECK-NEXT: ...,
    # CHECK: 5994, 5995, 5996, ..., 6324, 6325, 6326],
    # CHECK{LITERAL}: [1482183, 1482184, 1482185, ..., 1482513, 1482514, 1482515]]], dtype=index, shape=212x21x333)
    let tensor = Tensor[DType.index](212, 21, 333)
    iota(tensor.data(), tensor.num_elements())
    print(tensor.__str__())


fn main():
    test_tensor()
    test_tensor_init()
    test_tensor_equality()
    test_tensor_indexing()
    test_tensor_indexing2()
    test_tensor_fill()
    test_tensor_rand()
    test_euclidean_distance()
    test_image_to_grayscale()
    test_pow()
    test_print()
    test_print_small()
    test_print_large()
