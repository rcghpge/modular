// RUN: mo-opt -cse %s | mo-opt | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @conv2d(%[[ARG0:.*]]: !mo.tensor<[5, 5], f32>, %[[ARG1:.*]]: !mo.tensor<[3, 3], f32>) -> !mo.tensor<[2, 2], f32> {
mo.graph @conv2d(%arg0: !mo.tensor<[5, 5], f32>, %arg1: !mo.tensor<[3, 3], f32>) -> !mo.tensor<[2, 2], f32> {
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: mo.conv(%[[ARG0]], %[[ARG1]]) [strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.conv(%arg0, %arg1) [strides = %st, dilations = %di] : (
    !mo.tensor<[5, 5], f32>, !mo.tensor<[3, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @conv4d(%[[ARG0:.*]]: !mo.tensor<[5, 5, 16, 1], f32>, %[[ARG1:.*]]: !mo.tensor<[3, 3, 16, 1], f32>) -> !mo.tensor<[2, 2, 1, 1], f32> {
mo.graph @conv4d(%arg0: !mo.tensor<[5, 5, 16, 1], f32>, %arg1: !mo.tensor<[3, 3, 16, 1], f32>) -> !mo.tensor<[2, 2, 1, 1], f32> {
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<2, 2, 1, 1> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1, 1, 1> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %st = mo.constant {value = #M.dense_array<2, 2, 1, 1> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %di = mo.constant {value = #M.dense_array<1, 1, 1, 1> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK: mo.conv(%[[ARG0]], %[[ARG1]]) [strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.conv(%arg0, %arg1) [strides = %st, dilations = %di] : (
    !mo.tensor<[5, 5, 16, 1], f32>, !mo.tensor<[3, 3, 16, 1], f32>, !mo.tensor<[4], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[2, 2, 1, 1], f32>
  mo.output %0 : !mo.tensor<[2, 2, 1, 1], f32>
}

// CHECK: mo.graph @conv_dyn_rank(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<?, f32>, %[[ARG1:.*]]: !mo.tensor<?, f32>,
// CHECK-SAME: %[[ST:.*]]: !mo.tensor<?, si64>, %[[DI:.*]]: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
mo.graph @conv_dyn_rank(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<[?], si64>) -> !mo.tensor<?, f32> {
  // CHECK: mo.conv(%[[ARG0]], %[[ARG1]]) [strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.conv(%arg0, %arg1) [strides = %st, dilations = %di] : (
    !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[?], si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @max_pool(%[[ARG0:.*]]: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[9, 5], f32> {
mo.graph @max_pool(%arg0: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[9, 5], f32> {
  // CHECK-DAG: %[[FS:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: mo.max_pool(%[[ARG0]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di] : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[9, 5], f32>
  mo.output %0 : !mo.tensor<[9, 5], f32>
}

// CHECK: mo.graph @max_pool_dyn_rank(
// CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[10, 10], f32>,
// CHECK-SAME: %[[FS:.*]]: !mo.tensor<?, si64>, %[[ST:.*]]: !mo.tensor<?, si64>,
// CHECK-SAME: %[[DI:.*]]: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
mo.graph @max_pool_dyn_rank(
    %arg0: !mo.tensor<[10, 10], f32>, %fs: !mo.tensor<?, si64>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>
  ) -> !mo.tensor<?, f32> {
  // CHECK: mo.max_pool(%[[ARG0]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]]]
  %0 = mo.max_pool(%arg0) [filter_shape = %fs, strides = %st, dilations = %di] : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
