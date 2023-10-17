# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: address
# UNSUPPORTED: windows
# RUN: mojo -I %mobi_pkg_dir %s | FileCheck %s

import mobi
from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index

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


def mnist():
    var m = mobi.Module()

    var g = m.graph(
        "min_mnist",
        m.types(m.f32(1, 28, 28, 1)),
        m.types(m.f32(1, 10)),
    )
    var cst = g.op(
        "mo.constant",
        g.values(),
        m.types(m.f32(128, 10)),
        m.attrs(
            m.f32_attr(
                "value", m.f32(128, 10), "/Users/mdan/modular/.derived/p0.bin"
            )
        ),
    )
    var cst_0 = g.op(
        "mo.constant",
        g.values(),
        m.types(m.f32(1, 10)),
        m.attrs(
            m.f32_attr(
                "value",
                m.f32(1, 10),
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
        ),
    )
    var cst_1 = g.op(
        "mo.constant",
        g.values(),
        m.types(m.f32(784, 128)),
        m.attrs(
            m.f32_attr(
                "value", m.f32(784, 128), "/Users/mdan/modular/.derived/p2.bin"
            )
        ),
    )
    var cst_2 = g.op(
        "mo.constant",
        g.values(),
        m.types(m.f32(1, 128)),
        m.attrs(
            m.f32_attr(
                "value", m.f32(1, 128), "/Users/mdan/modular/.derived/p3.bin"
            )
        ),
    )
    var cst_3 = g.op(
        "mo.constant",
        g.values(),
        m.types(m.i32(2)),
        m.attrs(m.i32_attr("value", m.i32(2), -1, 784)),
    )
    var p1 = g.op(
        "mo.reshape", g.values(g[0], cst_3[0]), m.types(m.f32(1, 784))
    )
    var p2 = g.op(
        "mo.matmul", g.values(p1[0], cst_1[0]), m.types(m.f32(1, 128))
    )
    var p3 = g.op("mo.add", g.values(p2[0], cst_2[0]), m.types(m.f32(1, 128)))
    var p4 = g.op("mo.relu", g.values(p3[0]), m.types(m.f32(1, 128)))
    var p5 = g.op("mo.matmul", g.values(p4[0], cst[0]), m.types(m.f32(1, 10)))
    let p6 = g.op("mo.add", g.values(p5[0], cst_0[0]), m.types(m.f32(1, 10)))
    _ = g.op("mo.output", p6, m.types())

    # print(m.__str__())

    var input_tensor = Tensor[DType.float32](1, 28, 28, 1)
    with open("/Users/mdan/modular/.derived/two.bin", "r") as f:
        let str = f.read()
        let buf_read = Buffer[28 * 28, DType.float32](
            DTypePointer[DType.int8](str._buffer.data).bitcast[DType.float32]()
        )
        for i in range(28 * 28):
            input_tensor[i] = buf_read[i]

        _ = str ^
    # CHECK{LITERAL}: tensor([[-8.4354, -5.9548, 11.6617, ..., -7.9491, -0.2343, -7.3118]], dtype=f32, shape=[1,10])
    m.debug_print_execute(input_tensor)


def main():
    mnist()
