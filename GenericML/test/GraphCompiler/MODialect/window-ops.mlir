// RUN: mo-opt -cse %s | mo-opt | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @conv2d(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 5, 5, 16], f32>,
// CHECK-SAME: %[[FILTER:.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[10, 2, 2, 32], f32> no_inline {
mo.graph @conv2d(
  %input: !mo.tensor<[10, 5, 5, 16], f32>,
  %filter: !mo.tensor<[3, 3, 16, 32], f32>
) -> !mo.tensor<[10, 2, 2, 32], f32> {
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.conv(%input, %filter) [strides = %st, dilations = %di] : (
    !mo.tensor<[10, 5, 5, 16], f32>, !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[10, 2, 2, 32], f32>
  mo.output %0 : !mo.tensor<[10, 2, 2, 32], f32>
}

// CHECK: mo.graph @conv_dyn_rank(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<?, f32>, %[[FILTER:.*]]: !mo.tensor<?, f32>,
// CHECK-SAME: %[[ST:.*]]: !mo.tensor<?, si64>, %[[DI:.*]]: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> no_inline {
mo.graph @conv_dyn_rank(
  %input: !mo.tensor<?, f32>,
  %filter: !mo.tensor<?, f32>,
  %st: !mo.tensor<?, si64>,
  %di: !mo.tensor<[?], si64>
) -> !mo.tensor<?, f32> {
  // CHECK: mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.conv(%input, %filter) [strides = %st, dilations = %di] : (
    !mo.tensor<?, f32>, !mo.tensor<?, f32>,
    !mo.tensor<?, si64>, !mo.tensor<[?], si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

//===----------------------------------------------------------------------===//
// AvgPoolOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @avg_pool(%[[INPUT:.*]]: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> no_inline {
mo.graph @avg_pool(%input: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
  // CHECK-DAG: %[[FS:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: mo.avg_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.avg_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[20, 9, 5, 32], f32>
  mo.output %0 : !mo.tensor<[20, 9, 5, 32], f32>
}

// CHECK: mo.graph @avg_pool_dyn_rank(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<?, f32>,
// CHECK-SAME: %[[FS:.*]]: !mo.tensor<?, si64>, %[[ST:.*]]: !mo.tensor<?, si64>,
// CHECK-SAME: %[[DI:.*]]: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> no_inline {
mo.graph @avg_pool_dyn_rank(
    %input: !mo.tensor<?, f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>
  ) -> !mo.tensor<?, f32> {
  // CHECK: mo.avg_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.avg_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di] : (
    !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @max_pool(%[[INPUT:.*]]: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> no_inline {
mo.graph @max_pool(%input: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
  // CHECK-DAG: %[[FS:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: mo.max_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.max_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[20, 9, 5, 32], f32>
  mo.output %0 : !mo.tensor<[20, 9, 5, 32], f32>
}

// CHECK: mo.graph @max_pool_dyn_rank(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<?, f32>,
// CHECK-SAME: %[[FS:.*]]: !mo.tensor<?, si64>, %[[ST:.*]]: !mo.tensor<?, si64>,
// CHECK-SAME: %[[DI:.*]]: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> no_inline {
mo.graph @max_pool_dyn_rank(
    %input: !mo.tensor<?, f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>
  ) -> !mo.tensor<?, f32> {
  // CHECK: mo.max_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.max_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di] : (
    !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
