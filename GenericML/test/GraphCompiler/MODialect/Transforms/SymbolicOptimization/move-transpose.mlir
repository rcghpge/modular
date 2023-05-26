// RUN: mo-opt %s --symbolic-optimize | FileCheck %s

// CHECK: mo.graph @sink_transpose_through_binary_ops(
mo.graph @sink_transpose_through_binary_ops(
  // CHECK-SAME: %[[INPUT1:arg0]]: !mo.tensor<[2, 2, 2], f32>,
  %input1: !mo.tensor<[2, 2, 2], f32>,
  // CHECK-SAME: %[[INPUT2:.*]]: !mo.tensor<[2, 2, 2], f32>,
  %input2: !mo.tensor<[2, 2, 2], f32>,
  // CHECK-SAME: %[[INPUT3:.*]]: !mo.tensor<[2, 2, 2], f32>
  %input3: !mo.tensor<[2, 2, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[2, 2, 2], f32> {
) -> !mo.tensor<[2, 2, 2], f32> {
  // CHECK-NEXT: %[[PERM:.*]] = mo.constant {value = #M.dense_array<2, 1, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %perm = mo.constant {value = #M.dense_array<2, 1, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[ADD1:.*]] = mo.add(%[[INPUT1]], %[[INPUT2]]) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: %[[ADD2:.*]] = mo.add(%[[ADD1]], %[[INPUT3]]) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = mo.transpose(%[[ADD2]], %[[PERM]])
  // CHECK-SAME: : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %transpose1 = mo.transpose(%input1, %perm) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %transpose2 = mo.transpose(%input2, %perm) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %transpose3 = mo.transpose(%input3, %perm) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %add1 = mo.add(%transpose1, %transpose2) : !mo.tensor<[2, 2, 2], f32>
  %add2 = mo.add(%add1, %transpose3) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: mo.output %[[TRANSPOSE]] : !mo.tensor<[2, 2, 2], f32>
  mo.output %add2 : !mo.tensor<[2, 2, 2], f32>
}

// CHECK: mo.graph @sink_transpose_through_broadcast_ops<[[D0:.*]], [[D1:.*]], [[D2:.*]]>(
mo.graph @sink_transpose_through_broadcast_ops<D0, D1, D2>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[2, 2, 2], f32>
  %input: !mo.tensor<[2, 2, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[2, 3, [[D0]], [[D1]], [[D2]]], f32> {
) -> !mo.tensor<[2, 3, D0, D1, D2], f32> {
  // CHECK-NEXT: %[[PERM:.*]] = mo.constant {value = #M.dense_array<0, 1, 4, 2, 3> : !M.array<5xsi64>} : !mo.tensor<[5], si64>
  %perm = mo.constant {value = #M.dense_array<2, 0, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[BROADCAST:.*]] = mo.static.broadcast_to(%[[INPUT]])
  // CHECK-SAME: : !mo.tensor<[2, 2, 2], f32> -> !mo.tensor<[2, 3, [[D1]], [[D2]], [[D0]]], f32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = mo.transpose(%[[BROADCAST]], %[[PERM]])
  // CHECK-SAME: : (!mo.tensor<[2, 3, [[D1]], [[D2]], [[D0]]], f32>, !mo.tensor<[5], si64>) -> !mo.tensor<[2, 3, [[D0]], [[D1]], [[D2]]], f32>
  %transpose = mo.transpose(%input, %perm) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %broadcast = mo.static.broadcast_to(%transpose) : !mo.tensor<[2, 2, 2], f32> -> !mo.tensor<[2, 3, D0, D1, D2], f32>
  // CHECK-NEXT: mo.output %[[TRANSPOSE]] : !mo.tensor<[2, 3, [[D0]], [[D1]], [[D2]]], f32>
  mo.output %broadcast : !mo.tensor<[2, 3, D0, D1, D2], f32>
}

// CHECK: mo.graph @sink_transpose_through_multi_use_unary_ops(
mo.graph @sink_transpose_through_multi_use_unary_ops(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[2, 2, 2], f32>
  %input: !mo.tensor<[2, 2, 2], f32>
// CHECK-SAME: ) -> (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) {
) -> (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>) {
  // CHECK-NEXT: %[[MERGED_PERM:.*]] = mo.constant {value = #M.dense_array<1, 2, 0> : !M.array<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[PERM:.*]] = mo.constant {value = #M.dense_array<2, 0, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %perm = mo.constant {value = #M.dense_array<2, 0, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[RELU:.*]] = mo.relu(%[[INPUT]]) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: %[[RELU_T:.*]] = mo.transpose(%[[RELU]], %[[MERGED_PERM]])
  // CHECK-SAME: : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: %[[ABS:.*]] = mo.abs(%[[RELU]]) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: %[[ABS_T:.*]] = mo.transpose(%[[ABS]], %[[PERM]])
  // CHECK-SAME: : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %transpose1 = mo.transpose(%input, %perm) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %relu = mo.relu(%transpose1) : !mo.tensor<[2, 2, 2], f32>
  %transpose2 = mo.transpose(%relu, %perm) : (!mo.tensor<[2, 2, 2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[2, 2, 2], f32>
  %abs = mo.abs(%relu) : !mo.tensor<[2, 2, 2], f32>
  // CHECK-NEXT: mo.output %[[RELU_T]], %[[ABS_T]] : !mo.tensor<[2, 2, 2], f32>
  mo.output %transpose2, %abs : !mo.tensor<[2, 2, 2], f32>, !mo.tensor<[2, 2, 2], f32>
}

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
