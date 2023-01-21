// RUN: mo-opt -cse %s | mo-opt | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BroadCastOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @broadcast_same_rank(%[[FROM:.*]]: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @broadcast_same_rank(%from: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  // CHECK: %[[TO:.*]] = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %to  = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.broadcast_to(%[[FROM]], %[[TO]]) : (!mo.tensor<[1, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[1, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @broadcast_diff_rank(%[[FROM:.*]]: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @broadcast_diff_rank(%from: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 3], f32> {
  // CHECK: %[[TO:.*]] = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %to  = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.broadcast_to(%[[FROM]], %[[TO]]) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @concat(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32> {
mo.graph @concat(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32> {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK: mo.concat[%[[VAL0]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32>
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32>
  mo.output %0 : !mo.tensor<[2, 8], f32>
}

// CHECK: mo.graph @concat_dyn_rank(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32> {
mo.graph @concat_dyn_rank(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32> {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK: mo.concat[%[[VAL0]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32>
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<?, f32>) -> !mo.tensor<[2, 8], f32>
  mo.output %0 : !mo.tensor<[2, 8], f32>
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @gather(%[[PARAMS:.*]]: !mo.tensor<[2, 2], f32>, %[[INDICES:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
mo.graph @gather(%params: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK: mo.gather(%[[PARAMS]], %[[INDICES]], %[[AXIS]]) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  %0 = mo.gather(%params, %indices, %axis) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @gather_dyn_rank_param(%[[PARAMS:.*]]: !mo.tensor<?, f32>, %[[INDICES:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
mo.graph @gather_dyn_rank_param(%params: !mo.tensor<?, f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK: mo.gather(%[[PARAMS]], %[[INDICES]], %[[AXIS]]) : (!mo.tensor<?, f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  %0 = mo.gather(%params, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @shape_of_static_rank(%[[ARG:.*]]: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
mo.graph @shape_of_static_rank(%arg: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64> {
  // CHECK: %[[RES:.*]] = mo.shape_of(%[[ARG]]) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2], si64>
  mo.output %res : !mo.tensor<[2], si64>
}

// CHECK: mo.graph @shape_of_dyn_rank(%[[ARG:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
mo.graph @shape_of_dyn_rank(%arg: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64> {
  // CHECK: %[[RES:.*]] = mo.shape_of(%[[ARG]]) : (!mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
  %res = mo.shape_of(%arg) : (!mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[?], si64>
  mo.output %res : !mo.tensor<[?], si64>
}

//===----------------------------------------------------------------------===//
// NumpyBroadcastOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @broadcast_shape(%[[ARG0:.*]]: !mo.tensor<[1, 2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 1], f32>) -> !mo.tensor<[3], si64>
mo.graph @broadcast_shape(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[2, 1], f32>) -> !mo.tensor<[3], si64> {
  // CHECK: %[[RES:.*]] = mo.broadcast_shape(%[[ARG0:.*]], %[[ARG1:.*]]) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[2, 1], f32>) -> !mo.tensor<[3], si64>
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[2, 1], f32>) -> !mo.tensor<[3], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[3], si64>
  mo.output %res : !mo.tensor<[3], si64>
}
