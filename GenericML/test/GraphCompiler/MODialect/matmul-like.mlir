// RUN: mo-opt %s | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @batch_matmul(%[[ARG0:.*]]: !mo.tensor<[3, 10, 20], f32>, %[[ARG1:.*]]: !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32> {
mo.graph @batch_matmul(%arg0: !mo.tensor<[3, 10, 20], f32>, %arg1: !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3, 10, 20], f32>, !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[3, 10, 20], f32>, !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32>
  mo.output %0 : !mo.tensor<[3, 10, 5], f32>
}

// CHECK: mo.graph @batch_matmul_4D<D0, D1>(%[[ARG0:.*]]: !mo.tensor<[7, 3, 10, 20], f32>, %[[ARG1:.*]]: !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[D0, D1, 10, 5], f32> {
mo.graph @batch_matmul_4D<D0, D1>(%arg0: !mo.tensor<[7, 3, 10, 20], f32>, %arg1: !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[D0, D1, 10, 5], f32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[7, 3, 10, 20], f32>, !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[D0, D1, 10, 5], f32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[7, 3, 10, 20], f32>, !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[D0, D1, 10, 5], f32>
  mo.output %0 : !mo.tensor<[D0, D1, 10, 5], f32>
}

// CHECK: mo.graph @batch_matmul_mixed_precision(%[[ARG0:.*]]: !mo.tensor<[3, 10, 20], ui8>, %[[ARG1:.*]]: !mo.tensor<[3, 20, 5], si8>) -> !mo.tensor<[3, 10, 5], si32> {
mo.graph @batch_matmul_mixed_precision(%arg0: !mo.tensor<[3, 10, 20], ui8>, %arg1: !mo.tensor<[3, 20, 5], si8>) -> !mo.tensor<[3, 10, 5], si32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3, 10, 20], ui8>, !mo.tensor<[3, 20, 5], si8>) -> !mo.tensor<[3, 10, 5], si32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[3, 10, 20], ui8>, !mo.tensor<[3, 20, 5], si8>) -> !mo.tensor<[3, 10, 5], si32>
  mo.output %0 : !mo.tensor<[3, 10, 5], si32>
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @matmul(%[[ARG0:.*]]: !mo.tensor<[10, 20], f32>, %[[ARG1:.*]]: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32> {
mo.graph @matmul(%arg0: !mo.tensor<[10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> (!mo.tensor<[10, 5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>)   -> !mo.tensor<[10, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>)   -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// CHECK: mo.graph @matmul_dynamic<D0, D1, D2, D3, D4, D5>(%[[ARG0:.*]]: !mo.tensor<[D0, D1], f32>, %[[ARG1:.*]]: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D4, D5], f32> {
mo.graph @matmul_dynamic<D0, D1, D2, D3, D4, D5>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f32>) -> (!mo.tensor<[D4, D5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>)   -> !mo.tensor<[D4, D5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>)   -> !mo.tensor<[D4, D5], f32>
  mo.output %0 : !mo.tensor<[D4, D5], f32>
}

// CHECK: mo.graph @matmul_dyndim_1<D0, D1>
mo.graph @matmul_dyndim_1<D0, D1>(%arg0: !mo.tensor<[D0, 10], f32>, %arg1: !mo.tensor<[10, 5], f32>) -> (!mo.tensor<[D1, 5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[D0, 10], f32>, !mo.tensor<[10, 5], f32>) -> !mo.tensor<[D1, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[D0, 10], f32>, !mo.tensor<[10, 5], f32>) -> !mo.tensor<[D1, 5], f32>
  mo.output %0 : !mo.tensor<[D1, 5], f32>
}

// CHECK: mo.graph @matmul_dyndim_2<D0, D1>
mo.graph @matmul_dyndim_2<D0, D1>(%arg0: !mo.tensor<[10, 10], f32>, %arg1: !mo.tensor<[10, D0], f32>) -> (!mo.tensor<[10, D1], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[10, D0], f32>) -> !mo.tensor<[10, D1], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[10, D0], f32>) -> !mo.tensor<[10, D1], f32>
  mo.output %0 : !mo.tensor<[10, D1], f32>
}

// CHECK: mo.graph @matmul_dyndim_K<D0, D1>
mo.graph @matmul_dyndim_K<D0, D1>(%arg0: !mo.tensor<[10, D0], f32>, %arg1: !mo.tensor<[D1, 5], f32>) -> (!mo.tensor<[10, 5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 5], f32>) -> !mo.tensor<[10, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, D0], f32>, !mo.tensor<[D1, 5], f32>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// CHECK: mo.graph @matmul_fp16_mixed_precision<D0, D1>
mo.graph @matmul_fp16_mixed_precision<D0, D1>(%arg0: !mo.tensor<[10, D0], f16>, %arg1: !mo.tensor<[D1, 5], f16>) -> (!mo.tensor<[10, 5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, D0], f16>, !mo.tensor<[D1, 5], f16>) -> !mo.tensor<[10, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, D0], f16>, !mo.tensor<[D1, 5], f16>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// CHECK: mo.graph @matmul_si8_ui8_si32_mixed_precision<D0, D1>
mo.graph @matmul_si8_ui8_si32_mixed_precision<D0, D1>(%arg0: !mo.tensor<[10, D0], si8>, %arg1: !mo.tensor<[D1, 5], ui8>) -> (!mo.tensor<[10, 5], si32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, D0], si8>, !mo.tensor<[D1, 5], ui8>) -> !mo.tensor<[10, 5], si32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, D0], si8>, !mo.tensor<[D1, 5], ui8>) -> !mo.tensor<[10, 5], si32>
  mo.output %0 : !mo.tensor<[10, 5], si32>
}

// CHECK-LABEL: mo.graph @batch_matmul_unranked_input_ranked_output
// CHECK-SAME:  <[[SH0:[^,>]*]]: !mosh.ape, [[SH1:[^,>]*]]: !mosh.ape, D0, D1, D2>(
// CHECK-SAME:    %[[LHS:.*]]: !mo.tensor<[[SH0]], f32>,
// CHECK-SAME:    %[[RHS:.*]]: !mo.tensor<[[SH1]], f32>
// CHECK-SAME:  ) -> !mo.tensor<[D0, D1, D2], f32> {
mo.graph @batch_matmul_unranked_input_ranked_output<Sh0: !mosh.ape, Sh1: !mosh.ape, D0, D1, D2>(
  %lhs: !mo.tensor<Sh0, f32>,
  %rhs: !mo.tensor<Sh1, f32>
) -> !mo.tensor<[D0, D1, D2], f32> {

  // CHECK-NEXT: %[[RES:.*]] = mo.batch_matmul(%[[LHS]], %[[RHS]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>) -> !mo.tensor<[D0, D1, D2], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[D0, D1, D2], f32>
  %res = mo.batch_matmul(%lhs, %rhs) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<[D0, D1, D2], f32>
  mo.output %res : !mo.tensor<[D0, D1, D2], f32>
}

// CHECK-LABEL: mo.graph @batch_matmul_ranked_input_unranked_output
// CHECK-SAME:  <D0, D1, D2, D3, D4, D5, [[SH0:[^,>]*]]: !mosh.ape>(
// CHECK-SAME:    %[[LHS:.*]]: !mo.tensor<[D0, D1, D2], f32>,
// CHECK-SAME:    %[[RHS:.*]]: !mo.tensor<[D3, D4, D5], f32>
// CHECK-SAME:  ) -> !mo.tensor<[[SH0]], f32> {
mo.graph @batch_matmul_ranked_input_unranked_output<D0, D1, D2, D3, D4, D5, Sh0: !mosh.ape>(
  %lhs: !mo.tensor<[D0, D1, D2], f32>,
  %rhs: !mo.tensor<[D3, D4, D5], f32>
) -> !mo.tensor<Sh0, f32> {

  // CHECK-NEXT: %[[RES:.*]] = mo.batch_matmul(%[[LHS]], %[[RHS]]) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[[SH0]], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[[SH0]], f32>
  %res = mo.batch_matmul(%lhs, %rhs) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<Sh0, f32>
  mo.output %res : !mo.tensor<Sh0, f32>
}
