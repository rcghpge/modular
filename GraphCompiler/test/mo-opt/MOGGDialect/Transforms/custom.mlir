// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-mogg --mogg-fuse-prologues --mogg-fuse-epilogues --cse | FileCheck %s

// CHECK-LABEL: mo.graph @input_output_fusion
// CHECK: mogg.kernel
// CHECK: mogg.bind {{.*}} elemwise_lambda
// CHECK: mogg.call.elementwise["{{.*}}.Ceil"]
// CHECK: mogg.call.foreach
// CHECK: mogg.kernel
// CHECK: mogg.call.execute["{{.*}}.TestCustomOp"]
// CHECK: mogg.kernel
// CHECK: mogg.bind {{.*}} elemwise_lambda
// CHECK: mogg.call.elementwise["{{.*}}.Activation"]
// CHECK: mogg.call.foreach

mo.graph @input_output_fusion(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mo.ceil(%arg0) : !mo.tensor<[2], f32>
  %1 = mo.custom {symbol = "test_custom_op", device = #M.device_ref<"cpu", 0>}(%0) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  %2 = mo.activation(%1) {activation = "relu"} : !mo.tensor<[2], f32>
  mo.output %2 : !mo.tensor<[2], f32>
}
