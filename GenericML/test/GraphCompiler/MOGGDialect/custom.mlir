// RUN: mo-opt %s --mo-to-mogg=kernel-library-paths="%extra_test_lib" --fuse-in-out-lambdas --cse | FileCheck %s

// CHECK: mo.graph @input_output_fusion(%[[ARG0:.*]]: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
// CHECK-NEXT: %0 = mogg.kernel (%[[ARG0]]) : (%[[ARG1:.*]]: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>{
// CHECK:      mogg.lambda @input_0_fn
// CHECK:        mogg.call "mo.ceil"
// CHECK:      mogg.lambda @output_0_fn = (%arg2: !mogg.indices, %arg3: !mogg.simd<f32>){
// CHECK-NEXT:   mogg.call "mo.relu"
// CHECK:      mogg.call "test_custom_op" [@input_0_fn, @output_0_fn]
mo.graph @input_output_fusion(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mo.ceil(%arg0) : !mo.tensor<[2], f32>
  %1 = mo.custom {symbol = "test_custom_op"}(%0) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  %2 = mo.relu(%1) : !mo.tensor<[2], f32>
  mo.output %2 : !mo.tensor<[2], f32>
}

mo.graph @custom_with_params(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %1 = mo.custom {symbol = "test_custom_op_params", extra_param = 42, extra_param2 = "My String Param"}(%arg0) : (!mo.tensor<[2], f32>) -> !mo.tensor<[2], f32>
  mo.output %1 : !mo.tensor<[2], f32>
}

// Check the parameters have survived into MOGG.
// CHECK-LABEL: mo.graph @custom_with_params
// CHECK: mogg.call "test_custom_op_params" []({{.*}}) {extra_param = 42 : i64, extra_param2 = "My String Param"}
