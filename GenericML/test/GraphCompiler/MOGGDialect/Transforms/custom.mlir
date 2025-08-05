// RUN: mo-opt %s --mo-to-mogg=kernel-library-paths="%kernel_api_test_lib" --fuse-prologues-epilogues --cse | FileCheck %s

// CHECK-LABEL: mo.graph @input_output_fusion
// CHECK: mogg.experimental.kernel
// CHECK: mogg.bind {{.*}} elemwise_lambda
// CHECK: mogg.call.elementwise["mo.ceil"]
// CHECK: mogg.call.foreach
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.execute["test_custom_op"]
// CHECK: mogg.experimental.kernel
// CHECK: mogg.bind {{.*}} elemwise_lambda
// CHECK: mogg.call.elementwise["mo.relu"]
// CHECK: mogg.call.foreach

mo.graph @input_output_fusion(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mo.ceil(%arg0) : !mo.tensor<[2], f32>
  %1 = mo.custom {symbol = "test_custom_op", device = #M.device_ref<"cpu", 0>}(%0) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  %2 = mo.relu(%1) : !mo.tensor<[2], f32>
  mo.output %2 : !mo.tensor<[2], f32>
}
