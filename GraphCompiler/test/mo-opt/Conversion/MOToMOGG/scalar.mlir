// RUN: mo-opt %s --mogg-import-packages --mo-to-mogg | FileCheck %s
// TODO(GEX-1573): Convert the following sniff test to a complete test.

// CHECK-LABEL: mo.graph @mean
// CHECK: mogg.kernel
// CHECK: mogg.call.execute["{{.*}}.Mean"]

mo.graph @mean(%input: !mo.tensor<[100, 50], f32>) -> (!mo.tensor<[100, 1], f32>) {
  %output = mo.reduce.mean(%input) {axis = 1 : index} : (!mo.tensor<[100, 50], f32>) -> !mo.tensor<[100, 1], f32>
  mo.output %output : !mo.tensor<[100, 1], f32>
}
