// RUN: mo-opt %s --mo-to-mogg | FileCheck %s
// TODO(GEX-1573): Convert the following sniff test to a complete test.

// CHECK-LABEL: mo.graph @mean
// CHECK: mogg.kernel
// CHECK: mogg.call.execute["mo.mean"]

mo.graph @mean(%input: !mo.tensor<[100, 50], f32>) -> (!mo.tensor<[100, 1], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %output = mo.mean(%input, %axis) : (!mo.tensor<[100, 50], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[100, 1], f32>
  mo.output %output : !mo.tensor<[100, 1], f32>
}
