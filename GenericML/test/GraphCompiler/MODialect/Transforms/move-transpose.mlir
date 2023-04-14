// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// CHECK: mo.graph @sink_transpose_through_unary_ops(
mo.graph @sink_transpose_through_unary_ops(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[2, 2, 2], f32>
  %input: !mo.tensor<[2, 2, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[2, 2, 2], f32> {
) -> !mo.tensor<[2, 2, 2], f32> {
  // CHECK-NEXT: %[[PERM:.*]] = mo.constant {value = #M.dense_array<2, 1, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %perm = mo.constant {value = #M.dense_array<2, 1, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[RELU:.*]] = mo.relu(%[[INPUT]]) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: %[[ABS:.*]] = mo.abs(%[[RELU]]) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = mo.transpose(%[[ABS]], %[[PERM]])
  // CHECK-SAME: : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %transpose = mo.transpose(%input, %perm) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %relu = mo.relu(%transpose) : !mo.tensor<[2, 2, 2], f32>
  %abs = mo.abs(%relu) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: mo.output %[[TRANSPOSE]] : !mo.tensor<[2, 2, 2], f32>
  mo.output %abs : !mo.tensor<[2, 2, 2], f32>
}
