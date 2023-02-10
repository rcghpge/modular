// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s

// CHECK: mo.graph @rebinds<[[SH0:.*]]: !mo.shape>(%[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH0]], f32> {
mo.graph @rebinds<Sh0: !mo.shape -> Sh1: !mo.shape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32> {
  kgen.param.declare Sh2: !mo.shape = <Sh0>
  %0 = mo.rebind(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<Sh2, f32>
  // CHECK-NEXT: %[[VAL0:.*]] = mo.gelu(%[[ARG0]]) : !mo.tensor<[[SH0]], f32>
  %1 = mo.gelu(%0) : !mo.tensor<Sh2, f32>
  %2 = mo.rebind(%1) : !mo.tensor<Sh2, f32> -> !mo.tensor<Sh1, f32>
  // CHECK-NEXT: mo.output %[[VAL0]] : !mo.tensor<[[SH0]], f32>
  mo.output<:!mo.shape Sh2> %2 : !mo.tensor<Sh1, f32>
}

// CHECK: mo.graph @output_rebind_kept<[[SH0:.*]]: !mo.shape, [[SH1:.*]]: !mo.shape -> [[SH2:.*]]: !mo.shape>(
mo.graph @output_rebind_kept<Sh0: !mo.shape, Sh1: !mo.shape -> Sh2: !mo.shape>(
    // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
    %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> {
) -> !mo.tensor<Sh2, f32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.batch_matmul<() -> [[SH3:.*]]: !mo.shape>(%[[ARG0]], %[[ARG1]]) : (
  // CHECK-SAME:   !mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH3]], f32>
  %0 = mo.batch_matmul<() -> Sh3: !mo.shape>(%arg0, %arg1) : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh3, f32>
  // CHECK-NEXT: %[[VAL1:.*]] = mo.rebind(%[[VAL0]]) : !mo.tensor<[[SH3]], f32> -> !mo.tensor<[[SH2]], f32>
  %1 = mo.rebind(%0) : !mo.tensor<Sh3, f32> -> !mo.tensor<Sh2, f32>
  // CHECK-NEXT: mo.output<:!mo.shape [[SH3]]> %[[VAL1]] : !mo.tensor<[[SH2]], f32>
  mo.output<:!mo.shape Sh3> %1 : !mo.tensor<Sh2, f32>
}
