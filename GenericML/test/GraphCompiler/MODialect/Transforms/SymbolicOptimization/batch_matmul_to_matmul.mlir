// RUN: mo-opt --symbolic-optimize %s | FileCheck %s


// CHECK:      mo.graph @bmm_no_rebind(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3, 2], f32>,
// CHECK-SAME:   %[[WEIGHT:.*]]: !mo.tensor<[2, 3], f32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3, 3], f32> no_inline {
mo.graph @bmm_no_rebind(
  %input: !mo.tensor<[4, 3, 2], f32>,
  %weight: !mo.tensor<[2, 3], f32>
) -> (!mo.tensor<[4, 3, 3], f32>) {

  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[4, 3, 2], f32> -> !mo.tensor<[12, 2], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.matmul(%[[INPUT_RS]], %[[WEIGHT]]) : (!mo.tensor<[12, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[12, 3], f32>
  // CHECK-NEXT: %[[RES_RS:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[12, 3], f32> -> !mo.tensor<[4, 3, 3], f32>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[4, 2, 3], f32>
  %res = mo.batch_matmul(%input, %weight_bcast) : (!mo.tensor<[4, 3, 2], f32>, !mo.tensor<[4, 2, 3], f32>) -> !mo.tensor<[4, 3, 3], f32>

  // CHECK-NEXT: mo.output %[[RES_RS]] : !mo.tensor<[4, 3, 3], f32>
  mo.output %res : !mo.tensor<[4, 3, 3], f32>
}


// CHECK:      mo.graph @bmm_with_rebind<[[D0:.*]], [[D1:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3, [[D0]]], f32>,
// CHECK-SAME:   %[[WEIGHT:.*]]: !mo.tensor<[[[D1]], 3], f32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3, 3], f32> no_inline {
mo.graph @bmm_with_rebind<D0, D1>(
  %input: !mo.tensor<[4, 3, D0], f32>,
  %weight: !mo.tensor<[D1, 3], f32>
) -> (!mo.tensor<[4, 3, 3], f32>) {

  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[4, 3, [[D0]]], f32> -> !mo.tensor<[12, [[D0]]], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.matmul(%[[INPUT_RS]], %[[WEIGHT]]) : (!mo.tensor<[12, [[D0]]], f32>, !mo.tensor<[[[D1]], 3], f32>) -> !mo.tensor<[12, 3], f32>
  // CHECK-NEXT: %[[RES_RS:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[12, 3], f32> -> !mo.tensor<[4, 3, 3], f32>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[D1, 3], f32> -> !mo.tensor<[4, D1, 3], f32>
  %weight_bcast_rb = mo.rebind(%weight_bcast) : !mo.tensor<[4, D1, 3], f32> -> !mo.tensor<[4, D0, 3], f32>
  %res = mo.batch_matmul(%input, %weight_bcast_rb) : (!mo.tensor<[4, 3, D0], f32>, !mo.tensor<[4, D0, 3], f32>) -> !mo.tensor<[4, 3, 3], f32>

  // CHECK-NEXT: mo.output %[[RES_RS]] : !mo.tensor<[4, 3, 3], f32>
  mo.output %res : !mo.tensor<[4, 3, 3], f32>
}
