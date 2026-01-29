// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @bcast_like_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.broadcast_like(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f64>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @bcast_like_ok<[[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape, [[SH2:[^ ]*]]: !mosh.ape>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH1]], f64>) -> !mo.tensor<[[SH2]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.broadcast_like([[VARG0]], [[VARG1]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f64>) -> !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[[SH2]], f32>
// CHECK-NEXT: }

// -----

mo.graph @bcast_like_not_ok_bad_operands<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f64>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  // expected-error@below {{Input and output tensors do not share dtype!}}
  %1 = rmo.broadcast_like(%arg0, %arg1) : (!mo.tensor<Sh0, f64>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @bcast_like_not_ok_bad_results<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64> {
  // expected-error@below {{Input and output tensors do not share dtype!}}
  %1 = rmo.broadcast_like(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f64>
  mo.output %1 : !mo.tensor<Sh2, f64>
}
