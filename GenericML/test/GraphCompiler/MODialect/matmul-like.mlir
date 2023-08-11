// RUN: mo-opt %s | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @batch_matmul(%[[ARG0:.*]]: !mo.tensor<[3, 10, 20], f32>, %[[ARG1:.*]]: !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32> no_inline {
mo.graph @batch_matmul(%arg0: !mo.tensor<[3, 10, 20], f32>, %arg1: !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[3, 10, 20], f32>, !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[3, 10, 20], f32>, !mo.tensor<[3, 20, 5], f32>) -> !mo.tensor<[3, 10, 5], f32>
  mo.output %0 : !mo.tensor<[3, 10, 5], f32>
}

// CHECK: mo.graph @batch_matmul_4D(%[[ARG0:.*]]: !mo.tensor<[7, 3, 10, 20], f32>, %[[ARG1:.*]]: !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[?, ?, 10, 5], f32> no_inline {
mo.graph @batch_matmul_4D(%arg0: !mo.tensor<[7, 3, 10, 20], f32>, %arg1: !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[?, ?, 10, 5], f32> {
  // CHECK-NEXT: mo.batch_matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[7, 3, 10, 20], f32>, !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[?, ?, 10, 5], f32>
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[7, 3, 10, 20], f32>, !mo.tensor<[7, 3, 20, 5], f32>) -> !mo.tensor<[?, ?, 10, 5], f32>
  mo.output %0 : !mo.tensor<[?, ?, 10, 5], f32>
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @matmul(%[[ARG0:.*]]: !mo.tensor<[10, 20], f32>, %[[ARG1:.*]]: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32> no_inline {
mo.graph @matmul(%arg0: !mo.tensor<[10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> (!mo.tensor<[10, 5], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>)   -> !mo.tensor<[10, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>)   -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// CHECK: mo.graph @matmul_dynamic(%[[ARG0:.*]]: !mo.tensor<[?, ?], f32>, %[[ARG1:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32> no_inline {
mo.graph @matmul_dynamic(%arg0: !mo.tensor<[?, ?], f32>, %arg1: !mo.tensor<[?, ?], f32>) -> (!mo.tensor<[?, ?], f32>) {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>)   -> !mo.tensor<[?, ?], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>)   -> !mo.tensor<[?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?], f32>
}

// CHECK: mo.graph @matmul_dyndim_1
mo.graph @matmul_dyndim_1(%arg0: !mo.tensor<[?, 10], f32>, %arg1: !mo.tensor<[10, 5], f32>) -> (!mo.tensor<[?, 5], f32>) no_inline {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[?, 10], f32>, !mo.tensor<[10, 5], f32>) -> !mo.tensor<[?, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[?, 10], f32>, !mo.tensor<[10, 5], f32>) -> !mo.tensor<[?, 5], f32>
  mo.output %0 : !mo.tensor<[?, 5], f32>
}

// CHECK: mo.graph @matmul_dyndim_2
mo.graph @matmul_dyndim_2(%arg0: !mo.tensor<[10, 10], f32>, %arg1: !mo.tensor<[10, ?], f32>) -> (!mo.tensor<[10, ?], f32>) no_inline {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[10, ?], f32>) -> !mo.tensor<[10, ?], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[10, ?], f32>) -> !mo.tensor<[10, ?], f32>
  mo.output %0 : !mo.tensor<[10, ?], f32>
}

// CHECK: mo.graph @matmul_dyndim_K
mo.graph @matmul_dyndim_K(%arg0: !mo.tensor<[10, ?], f32>, %arg1: !mo.tensor<[?, 5], f32>) -> (!mo.tensor<[10, 5], f32>) no_inline {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, ?], f32>, !mo.tensor<[?, 5], f32>) -> !mo.tensor<[10, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, ?], f32>, !mo.tensor<[?, 5], f32>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// CHECK: mo.graph @matmul_fp16_mixed_precision
mo.graph @matmul_fp16_mixed_precision(%arg0: !mo.tensor<[10, ?], f16>, %arg1: !mo.tensor<[?, 5], f16>) -> (!mo.tensor<[10, 5], f32>) no_inline {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, ?], f16>, !mo.tensor<[?, 5], f16>) -> !mo.tensor<[10, 5], f32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, ?], f16>, !mo.tensor<[?, 5], f16>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// CHECK: mo.graph @matmul_si8_ui8_si32_mixed_precision
mo.graph @matmul_si8_ui8_si32_mixed_precision(%arg0: !mo.tensor<[10, ?], si8>, %arg1: !mo.tensor<[?, 5], ui8>) -> (!mo.tensor<[10, 5], si32>) no_inline {
  // CHECK-NEXT: mo.matmul(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[10, ?], si8>, !mo.tensor<[?, 5], ui8>) -> !mo.tensor<[10, 5], si32>
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, ?], si8>, !mo.tensor<[?, 5], ui8>) -> !mo.tensor<[10, 5], si32>
  mo.output %0 : !mo.tensor<[10, 5], si32>
}
