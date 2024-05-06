// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @bcast_like_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f64>) -> !mo.tensor<?, f32> {
  %1 = rmo.broadcast_like(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bcast_like_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>, [[VARG1:[^ ]*]]: !mo.tensor<?, f64>) -> !mo.tensor<?, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.broadcast_like([[VARG0]], [[VARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<?, f64>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>
// CHECK-NEXT: }

// -----

mo.graph @bcast_like_not_ok_bad_operands(%arg0: !mo.tensor<?, f64>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error@below {{Input and output tensors do not share dtype!}}
  %1 = rmo.broadcast_like(%arg0, %arg1) : (!mo.tensor<?, f64>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @bcast_like_not_ok_bad_results(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f64> {
  // expected-error@below {{Input and output tensors do not share dtype!}}
  %1 = rmo.broadcast_like(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f64>
  mo.output %1 : !mo.tensor<?, f64>
}
