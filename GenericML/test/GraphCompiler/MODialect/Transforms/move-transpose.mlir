// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// CHECK: mo.graph @lift_transpose(
mo.graph @lift_transpose(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[2, 2, 2], f32>
  %input: !mo.tensor<[2, 2, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[2, 2, 2], f32> {
) -> !mo.tensor<[2, 2, 2], f32> {
  // CHECK-NEXT: %[[PERM:.*]] = mo.constant {value = #M.dense_array<2, 1, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %perm = mo.constant {value = #M.dense_array<2, 1, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = mo.transpose(%[[INPUT]], %[[PERM]])
  // CHECK-SAME: : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: %[[RELU:.*]] = mo.relu(%[[TRANSPOSE]]) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: %[[ABS:.*]] = mo.abs(%[[RELU]]) : !mo.tensor<[2, 2, 2], f32>
  %relu = mo.relu(%input) : !mo.tensor<[2, 2, 2], f32>
  %abs = mo.abs(%relu) : !mo.tensor<[2, 2, 2], f32>
  %res = mo.transpose(%abs, %perm) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: mo.output %[[ABS]] : !mo.tensor<[2, 2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2, 2], f32>
}
