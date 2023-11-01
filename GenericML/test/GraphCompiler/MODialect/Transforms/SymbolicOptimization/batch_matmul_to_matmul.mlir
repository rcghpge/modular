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

// CHECK:      mo.graph @bmm_with_device(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3, 2], f32>,
// CHECK-SAME:   %[[WEIGHT:.*]]: !mo.tensor<[2, 3], f32>
// CHECK-SAME: ) -> !mo.tensor<[4, 3, 3], f32> no_inline {
mo.graph @bmm_with_device(
  %input: !mo.tensor<[4, 3, 2], f32>,
  %weight: !mo.tensor<[2, 3], f32>
) -> (!mo.tensor<[4, 3, 3], f32>) {
  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) {device = #M.device_ref<"dev", 0>} : !mo.tensor<[4, 3, 2], f32> -> !mo.tensor<[12, 2], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.matmul(%[[INPUT_RS]], %[[WEIGHT]]) {device = #M.device_ref<"dev", 0>} : (!mo.tensor<[12, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[12, 3], f32>
  // CHECK-NEXT: %[[RES_RS:.*]] = mo.static.reshape(%[[RES]]) {device = #M.device_ref<"dev", 0>} : !mo.tensor<[12, 3], f32> -> !mo.tensor<[4, 3, 3], f32>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[4, 2, 3], f32>
  %res = mo.batch_matmul(%input, %weight_bcast) { device = #M.device_ref<"dev", 0> }: (!mo.tensor<[4, 3, 2], f32>, !mo.tensor<[4, 2, 3], f32>) -> !mo.tensor<[4, 3, 3], f32>

  // CHECK-NEXT: mo.output %[[RES_RS]] : !mo.tensor<[4, 3, 3], f32>
  mo.output %res : !mo.tensor<[4, 3, 3], f32>
}

// CHECK:      mo.graph @bmm_mixed_precision(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[4, 3, 2], f16>,
// CHECK-SAME:   %[[WEIGHT:.*]]: !mo.tensor<[2, 3], bf16>
// CHECK-SAME: ) -> !mo.tensor<[4, 3, 3], f32> no_inline {
mo.graph @bmm_mixed_precision(
  %input: !mo.tensor<[4, 3, 2], f16>,
  %weight: !mo.tensor<[2, 3], bf16>
) -> (!mo.tensor<[4, 3, 3], f32>) {

  // CHECK-NEXT: %[[INPUT_RS:.*]] = mo.static.reshape(%[[INPUT]]) : !mo.tensor<[4, 3, 2], f16> -> !mo.tensor<[12, 2], f16>
  // CHECK-NEXT: %[[RES:.*]] = mo.matmul(%[[INPUT_RS]], %[[WEIGHT]]) : (!mo.tensor<[12, 2], f16>, !mo.tensor<[2, 3], bf16>) -> !mo.tensor<[12, 3], f32>
  // CHECK-NEXT: %[[RES_RS:.*]] = mo.static.reshape(%[[RES]]) : !mo.tensor<[12, 3], f32> -> !mo.tensor<[4, 3, 3], f32>
  %weight_bcast = mo.static.broadcast_to(%weight) : !mo.tensor<[2, 3], bf16> -> !mo.tensor<[4, 2, 3], bf16>
  %res = mo.batch_matmul(%input, %weight_bcast) : (!mo.tensor<[4, 3, 2], f16>, !mo.tensor<[4, 2, 3], bf16>) -> !mo.tensor<[4, 3, 3], f32>

  // CHECK-NEXT: mo.output %[[RES_RS]] : !mo.tensor<[4, 3, 3], f32>
  mo.output %res : !mo.tensor<[4, 3, 3], f32>
}

// CHECK:      mo.graph @bmm_batch_size_one(
// CHECK-SAME:   %[[A:.*]]: !mo.tensor<[1, 2, 3], f32>
// CHECK-SAME:   %[[B:.*]]: !mo.tensor<[1, 3, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[1, 2, 2], f32> no_inline {
mo.graph @bmm_batch_size_one(
  %A: !mo.tensor<[1, 2, 3], f32>,
  %B: !mo.tensor<[1, 3, 2], f32>
) -> (!mo.tensor<[1, 2, 2], f32>) {

  // CHECK-NEXT: %[[A_RESHAPE:.*]] = mo.static.reshape(%[[A]]) : !mo.tensor<[1, 2, 3], f32> -> !mo.tensor<[2, 3], f32>
  // CHECK-NEXT: %[[B_RESHAPE:.*]] = mo.static.reshape(%[[B]]) : !mo.tensor<[1, 3, 2], f32> -> !mo.tensor<[3, 2], f32>
  // CHECK-NEXT: %[[MATMUL:.*]] = mo.matmul(%[[A_RESHAPE]], %[[B_RESHAPE]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[3, 2], f32>) -> !mo.tensor<[2, 2], f32>
  // CHECK-NEXT: %[[OUTPUT:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[1, 2, 2], f32>
  %res = mo.batch_matmul(%A, %B) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 3, 2], f32>) -> !mo.tensor<[1, 2, 2], f32>

  // CHECK-NEXT: mo.output %[[OUTPUT]] : !mo.tensor<[1, 2, 2], f32>
  mo.output %res : !mo.tensor<[1, 2, 2], f32>
}

// CHECK:      mo.graph @bmm_batch_size_ones(
// CHECK-SAME:   %[[A:.*]]: !mo.tensor<[1, 1, 1, 2, 3], f32>
// CHECK-SAME:   %[[B:.*]]: !mo.tensor<[1, 1, 1, 3, 2], f32>
// CHECK-SAME: ) -> !mo.tensor<[1, 1, 1, 2, 2], f32> no_inline {
mo.graph @bmm_batch_size_ones(
  %A: !mo.tensor<[1, 1, 1, 2, 3], f32>,
  %B: !mo.tensor<[1, 1, 1, 3, 2], f32>
) -> (!mo.tensor<[1, 1, 1, 2, 2], f32>) {

  // CHECK-NEXT: %[[A_RESHAPE:.*]] = mo.static.reshape(%[[A]]) : !mo.tensor<[1, 1, 1, 2, 3], f32> -> !mo.tensor<[2, 3], f32>
  // CHECK-NEXT: %[[B_RESHAPE:.*]] = mo.static.reshape(%[[B]]) : !mo.tensor<[1, 1, 1, 3, 2], f32> -> !mo.tensor<[3, 2], f32>
  // CHECK-NEXT: %[[MATMUL:.*]] = mo.matmul(%[[A_RESHAPE]], %[[B_RESHAPE]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[3, 2], f32>) -> !mo.tensor<[2, 2], f32>
  // CHECK-NEXT: %[[OUTPUT:.*]] = mo.static.reshape(%[[MATMUL]]) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[1, 1, 1, 2, 2], f32>
  %res = mo.batch_matmul(%A, %B) : (!mo.tensor<[1, 1, 1, 2, 3], f32>, !mo.tensor<[1, 1, 1, 3, 2], f32>) -> !mo.tensor<[1, 1, 1, 2, 2], f32>

  // CHECK-NEXT: mo.output %[[OUTPUT]] : !mo.tensor<[1, 1, 1, 2, 2], f32>
  mo.output %res : !mo.tensor<[1, 1, 1, 2, 2], f32>
}
