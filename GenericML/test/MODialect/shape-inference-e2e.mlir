// RUN: mo-opt %s --resolve-unknown-parameters --materialize-fw-shape-funcs --propagate-shapes --canonicalize | FileCheck %s

// CHECK: mo.graph @unresolved_rebinds<[[SH0:.*]]: !mo.shape, [[SH1:.*]]: !mo.shape -> [[SH3:.*]]: !mo.shape>(
mo.graph @unresolved_rebinds(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
  %arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH3]], f32> {
) -> !mo.tensor<?, f32> {
  // CHECK: %[[ARG1REB:.*]] = mo.rebind(%[[ARG1]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH0]], f32>
  // CHECK: %[[VAL0:.*]] = mo.add(%[[ARG0]], %[[ARG1REB]]) : !mo.tensor<[[SH0]], f32>
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<?, f32>
  // CHECK: %[[VAL1:.*]] = mo.gelu(%[[VAL0]]) : !mo.tensor<[[SH0]], f32>
  %1 = mo.gelu(%0) : !mo.tensor<?, f32>
  // CHECK: kgen.param.declare [[SH2:.*]]: !mo.shape = <apply(:(!mo.shape, !mo.shape) -> !mo.shape @batch_matmul_fw, [[SH0]], [[SH1]])>
  // CHECK: %[[VAL2:.*]] = mo.batch_matmul(%[[VAL1]], %[[ARG1]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[[SH2]], f32>
  %2 = mo.batch_matmul(%1, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  // CHECK: %[[VAL3:.*]] = mo.relu(%[[VAL2]]) : !mo.tensor<[[SH2]], f32>
  %3 = mo.relu(%2) : !mo.tensor<?, f32>
  // CHECK: %[[VAL4:.*]] = mo.gelu(%[[VAL3]]) : !mo.tensor<[[SH2]], f32>
  %4 = mo.gelu(%3) : !mo.tensor<?, f32>
  // CHECK: %[[VAL5:.*]] = mo.sub(%[[VAL4]], %[[VAL3]]) : !mo.tensor<[[SH2]], f32>
  %5 = mo.sub(%4, %3) : !mo.tensor<?, f32>
  // CHECK: %[[VAL6:.*]] = mo.rebind(%[[VAL5]]) : !mo.tensor<[[SH2]], f32> -> !mo.tensor<[[SH3]], f32>
  // CHECK: mo.output %[[VAL6]] : !mo.tensor<[[SH3]], f32>
  mo.output %5 : !mo.tensor<?, f32>
}
