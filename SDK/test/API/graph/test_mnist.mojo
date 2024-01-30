# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: mojo %s | FileCheck %s

from tensor import Tensor, TensorShape
from utils.index import Index

from max.graph import *

# Based on the MNIST example at https://www.tensorflow.org/datasets/keras_example:
#
#   func.func @serving_default(%arg0: tensor<1x28x28x1xf32> {tf_saved_model.index_path = ["inputs"]}) -> (tensor<1x10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "serving_default_inputs:0", outputs = "StatefulPartitionedCall_1:0"}, tf_saved_model.exported_names = ["serving_default"]} {
#     %cst = "tf.Const"() {value = ...
#     %cst_0 = "tf.Const"() {value = dense<[-0.0675942451, 0.0063267909, 7.43086217E-4, -0.0126994187, 0.0148473661, 0.108896509, -0.0398316309, 0.0461452715, -0.0281771384, -0.0431172103]> : tensor<10xf32>} : () -> tensor<10xf32>
#     %cst_1 = "tf.Const"() {value = ...
#     %cst_2 = "tf.Const"() {value = ...
#     %cst_3 = "tf.Const"() {value = dense<[-1, 784]> : tensor<2xi32>} : () -> tensor<2xi32>
#     %0 = "tf.Squeeze"(%arg0) {device = "", squeeze_dims = [-1]} : (tensor<1x28x28x1xf32>) -> tensor<1x28x28xf32>
#     %1 = "tf.Reshape"(%0, %cst_3) {device = ""} : (tensor<1x28x28xf32>, tensor<2xi32>) -> tensor<1x784xf32>
#     %2 = "tf.MatMul"(%1, %cst_1) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x784xf32>, tensor<784x128xf32>) -> tensor<1x128xf32>
#     %3 = "tf.BiasAdd"(%2, %cst_2) {data_format = "NHWC", device = ""} : (tensor<1x128xf32>, tensor<128xf32>) -> tensor<1x128xf32>
#     %4 = "tf.Relu"(%3) {device = ""} : (tensor<1x128xf32>) -> tensor<1x128xf32>
#     %5 = "tf.MatMul"(%4, %cst) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x128xf32>, tensor<128x10xf32>) -> tensor<1x10xf32>
#     %6 = "tf.BiasAdd"(%5, %cst_0) {data_format = "NHWC", device = ""} : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
#     return %6 : tensor<1x10xf32>
#   }


def test_mnist_low_level():
    var m = Module()

    var g = m.graph(
        "test_mnist_low_level",
        MOTensor(DType.float32, 1, 28, 28, 1),
        MOTensor(DType.float32, 1, 10),
    )
    let cst_data = Tensor[DType.float32](128, 10)
    cst_data._to_buffer().fill(0.5)
    let cst = g.op(
        "mo.constant",
        MOTensor(DType.float32, 128, 10),
        AttrMap(m.tensor_attr("value", cst_data)),
    )
    let cst_0 = g.op(
        "mo.constant",
        MOTensor(DType.float32, 1, 10),
        AttrMap(
            m.tensor_attr(
                "value",
                Tensor[DType.float32](
                    TensorShape(1, 10),
                    -0.0675942451,
                    0.0063267909,
                    7.43086217e-4,
                    -0.0126994187,
                    0.0148473661,
                    0.108896509,
                    -0.0398316309,
                    0.0461452715,
                    -0.0281771384,
                    -0.0431172103,
                ),
            )
        ),
    )

    let cst_1_data = Tensor[DType.float32](784, 128)
    cst_1_data._to_buffer().fill(0.5)
    let cst_1 = g.op(
        "mo.constant",
        MOTensor(DType.float32, 784, 128),
        AttrMap(m.tensor_attr("value", cst_1_data)),
    )
    let cst_2_data = Tensor[DType.float32](1, 128)
    cst_2_data._to_buffer().fill(0.5)
    let cst_2 = g.op(
        "mo.constant",
        MOTensor(DType.float32, 1, 128),
        AttrMap(m.tensor_attr("value", cst_2_data)),
    )
    let cst_3 = g.op(
        "mo.constant",
        MOTensor(DType.int32, 2),
        AttrMap(
            m.tensor_attr("value", Tensor[DType.int32](TensorShape(2), -1, 784))
        ),
    )
    let p1 = g.op("mo.reshape", (g[0], cst_3), MOTensor(DType.float32, 1, 784))
    let p2 = g.op("mo.matmul", (p1, cst_1), MOTensor(DType.float32, 1, 128))
    let p3 = g.op("mo.add", (p2, cst_2), MOTensor(DType.float32, 1, 128))
    let p4 = g.op("mo.relu", p3, MOTensor(DType.float32, 1, 128))
    let p5 = g.op("mo.matmul", (p4, cst), MOTensor(DType.float32, 1, 10))
    let p6 = g.op("mo.add", (p5, cst_0), MOTensor(DType.float32, 1, 10))
    _ = g.nvop("mo.output", p6)

    # CHECK: test_mnist_low_level
    m.verify()
    print(m.__str__())


def test_mnist_helpers():
    var m = Module()

    var g = m.graph(
        "test_mnist_helpers",
        MOTensor(DType.float32, 1, 28, 28, 1),
        MOTensor(DType.float32, 1, 10),
    )
    let cst_data = Tensor[DType.float32](128, 10)
    cst_data._to_buffer().fill(0.5)
    let cst = g.constant(cst_data)

    let cst_0 = g.constant(
        Tensor[DType.float32](
            TensorShape(1, 10),
            -0.0675942451,
            0.0063267909,
            7.43086217e-4,
            -0.0126994187,
            0.0148473661,
            0.108896509,
            -0.0398316309,
            0.0461452715,
            -0.0281771384,
            -0.0431172103,
        )
    )

    let cst_1_data = Tensor[DType.float32](784, 128)
    cst_1_data._to_buffer().fill(0.5)
    let cst_1 = g.constant(cst_1_data)

    let cst_2_data = Tensor[DType.float32](1, 128)
    cst_2_data._to_buffer().fill(0.5)
    let cst_2 = g.constant(cst_2_data)

    let p1 = ops.reshape(g[0], 1, 784)
    let p2 = p1 @ cst_1
    let p3 = p2 + cst_2
    let p4 = g.op("mo.relu", p3, MOTensor(DType.float32, 1, 128))
    let p5 = p4 @ cst
    let p6 = p5 + cst_0
    _ = g.output(p6)

    # CHECK: test_mnist_helpers
    m.verify()
    print(m.__str__())


def main():
    test_mnist_low_level()
    test_mnist_helpers()
