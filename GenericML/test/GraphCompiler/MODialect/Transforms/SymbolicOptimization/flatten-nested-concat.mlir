// RUN: mo-opt %s --symbolic-optimize | FileCheck %s

// CHECK: mo.graph @nested_concat(
mo.graph @nested_concat(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[10, 20], f32>
  %arg0: !mo.tensor<[10, 20], f32>
// CHECK-SAME: ) -> !mo.tensor<[30, 20], f32> no_inline {
) -> !mo.tensor<[30, 20], f32> {
  // concat axis
  // CHECK-NEXT: %[[AX0:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %ax0 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>

  // concat inputs
  // CHECK-NEXT: %[[T0:.*]] = mo.add(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  // CHECK-NEXT: %[[T1:.*]] = mo.sub(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  // CHECK-NEXT: %[[T2:.*]] = mo.mul(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  %t0 = mo.add(%arg0, %arg0) : !mo.tensor<[10, 20], f32>
  %t1 = mo.sub(%arg0, %arg0) : !mo.tensor<[10, 20], f32>
  %t2 = mo.mul(%arg0, %arg0) : !mo.tensor<[10, 20], f32>

  // nested concats; notice they turned into a single concat on all the inputs
  // CHECK-NEXT: %[[RES:.*]] = mo.concat[%[[AX0]] : !mo.tensor<[], si64>] (%[[T0]], %[[T1]], %[[T2]])
  // CHECK-SAME: : (!mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[30, 20], f32>
  %c0 = mo.concat[%ax0 : !mo.tensor<[], si64>] (%t0, %t1)
  : (!mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[20, 20], f32>
  %res = mo.concat[%ax0 : !mo.tensor<[], si64>] (%c0, %t2)
  : (!mo.tensor<[20, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[30, 20], f32>

  // CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[30, 20], f32>
  mo.output %res : !mo.tensor<[30, 20], f32>
}

// No effect
// CHECK: mo.graph @nested_concat_dyn_axis(
mo.graph @nested_concat_dyn_axis(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[10, 20], f32>,
  %arg0: !mo.tensor<[10, 20], f32>,
  // CHECK-SAME: %[[AX0:.*]]: !mo.tensor<[], si64>
  %ax0 : !mo.tensor<[], si64>
// CHECK-SAME: ) -> !mo.tensor<[30, 20], f32> no_inline {
) -> !mo.tensor<[30, 20], f32> {
  // concat inputs
  // CHECK-NEXT: %[[T0:.*]] = mo.add(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  // CHECK-NEXT: %[[T1:.*]] = mo.sub(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  // CHECK-NEXT: %[[T2:.*]] = mo.mul(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  %t0 = mo.add(%arg0, %arg0) : !mo.tensor<[10, 20], f32>
  %t1 = mo.sub(%arg0, %arg0) : !mo.tensor<[10, 20], f32>
  %t2 = mo.mul(%arg0, %arg0) : !mo.tensor<[10, 20], f32>

  // nested concats; notice they didn't get merged, although we should be able
  // relax the pattern matching and support this case if needed.
  // CHECK-NEXT: %[[C0:.*]] = mo.concat[%[[AX0]] : !mo.tensor<[], si64>] (%[[T0]], %[[T1]])
  // CHECK-SAME: : (!mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[20, 20], f32>
  %c0 = mo.concat[%ax0 : !mo.tensor<[], si64>] (%t0, %t1)
  : (!mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[20, 20], f32>

  // CHECK-NEXT: %[[RES:.*]] = mo.concat[%[[AX0]] : !mo.tensor<[], si64>] (%[[C0]], %[[T2]])
  // CHECK-SAME: : (!mo.tensor<[20, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[30, 20], f32>
  %res = mo.concat[%ax0 : !mo.tensor<[], si64>] (%c0, %t2)
  : (!mo.tensor<[20, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[30, 20], f32>

  // CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[30, 20], f32>
  mo.output %res : !mo.tensor<[30, 20], f32>
}

// No effect
// CHECK: mo.graph @nested_concat_no_nested_concat(
mo.graph @nested_concat_no_nested_concat(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[10, 20], f32>
  %arg0: !mo.tensor<[10, 20], f32>
// CHECK-SAME: ) -> !mo.tensor<[30, 20], f32> no_inline {
) -> !mo.tensor<[30, 20], f32> {
  // concat axis
  // CHECK-NEXT: %[[AX0:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %ax0 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>

  // concat inputs
  // CHECK-NEXT: %[[T0:.*]] = mo.add(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  // CHECK-NEXT: %[[T1:.*]] = mo.sub(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  // CHECK-NEXT: %[[T2:.*]] = mo.mul(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  %t0 = mo.add(%arg0, %arg0) : !mo.tensor<[10, 20], f32>
  %t1 = mo.sub(%arg0, %arg0) : !mo.tensor<[10, 20], f32>
  %t2 = mo.mul(%arg0, %arg0) : !mo.tensor<[10, 20], f32>

  // single concat, nothing to merge
  // CHECK-NEXT: %[[RES:.*]] = mo.concat[%[[AX0]] : !mo.tensor<[], si64>] (%[[T0]], %[[T1]], %[[T2]])
  // CHECK-SAME: : (!mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[30, 20], f32>
  %res = mo.concat[%ax0 : !mo.tensor<[], si64>] (%t0, %t1, %t2)
  : (!mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[30, 20], f32>

  // CHECK-NEXT:   mo.output %[[RES]] : !mo.tensor<[30, 20], f32>
  mo.output %res : !mo.tensor<[30, 20], f32>
}

// No effect
// CHECK: mo.graph @nested_concat_multi_user_input(
mo.graph @nested_concat_multi_user_input(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[10, 20], f32>
  %arg0: !mo.tensor<[10, 20], f32>
// CHECK-SAME: ) -> (!mo.tensor<[30, 20], f32>, !mo.tensor<[20, 20], f32>) no_inline {
) -> (!mo.tensor<[30, 20], f32>, !mo.tensor<[20, 20], f32>) {
  // concat axis
  // CHECK-NEXT: %[[AX0:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %ax0 = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>

  // concat inputs
  // CHECK-NEXT: %[[T0:.*]] = mo.add(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  // CHECK-NEXT: %[[T1:.*]] = mo.sub(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  // CHECK-NEXT: %[[T2:.*]] = mo.mul(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[10, 20], f32>
  %t0 = mo.add(%arg0, %arg0) : !mo.tensor<[10, 20], f32>
  %t1 = mo.sub(%arg0, %arg0) : !mo.tensor<[10, 20], f32>
  %t2 = mo.mul(%arg0, %arg0) : !mo.tensor<[10, 20], f32>

  // nested concats; won't merge since `%c0` is used afterwards.
  // CHECK-NEXT: %[[C0:.*]] = mo.concat[%[[AX0]] : !mo.tensor<[], si64>] (%[[T0]], %[[T1]])
  // CHECK-SAME: : (!mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[20, 20], f32>
  %c0 = mo.concat[%ax0 : !mo.tensor<[], si64>] (%t0, %t1)
  : (!mo.tensor<[10, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[20, 20], f32>

  // CHECK-NEXT: %[[RES:.*]] = mo.concat[%[[AX0]] : !mo.tensor<[], si64>] (%[[C0]], %[[T2]])
  // CHECK-SAME: : (!mo.tensor<[20, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[30, 20], f32>
  %res = mo.concat[%ax0 : !mo.tensor<[], si64>] (%c0, %t2)
  : (!mo.tensor<[20, 20], f32>, !mo.tensor<[10, 20], f32>) -> !mo.tensor<[30, 20], f32>

  // CHECK-NEXT: mo.output %[[RES]], %[[C0]] : !mo.tensor<[30, 20], f32>, !mo.tensor<[20, 20], f32>
  mo.output %res, %c0 : !mo.tensor<[30, 20], f32>, !mo.tensor<[20, 20], f32>
}
