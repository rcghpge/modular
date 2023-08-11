// RUN: mo-opt %s --symbolic-optimize --cse | FileCheck %s

// CHECK: mo.graph @broadcast_reshape_scalar(%[[ARG0:.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[64], f32> no_inline {
mo.graph @broadcast_reshape_scalar(%arg0 : !mo.tensor<[], f32>) -> !mo.tensor<[64], f32>{
  // CHECK: %[[VAL0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[], f32> -> !mo.tensor<[64], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<[64, 1], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: mo.output %[[VAL0]] : !mo.tensor<[64], f32>
  mo.output %1 : !mo.tensor<[64], f32>
// CHECK: }
}

// CHECK: mo.graph @broadcast_reshape_scalar_vector(%[[ARG0:.*]]: !mo.tensor<[1], f32>) -> !mo.tensor<[64], f32> no_inline {
mo.graph @broadcast_reshape_scalar_vector(%arg0 : !mo.tensor<[1], f32>) -> !mo.tensor<[64], f32>{
  // CHECK: %[[VAL0:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[1], f32> -> !mo.tensor<[], f32>
  // CHECK: %[[VAL1:.*]] = mo.static.broadcast_to(%[[VAL0]]) : !mo.tensor<[], f32> -> !mo.tensor<[64], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[1], f32> -> !mo.tensor<[64, 1], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: mo.output %[[VAL1]] : !mo.tensor<[64], f32>
  mo.output %1 : !mo.tensor<[64], f32>
// CHECK: }
}

// CHECK: mo.graph @broadcast_reshape_scalar_matrix(%[[ARG0:.*]]: !mo.tensor<[1, 1], f32>) -> !mo.tensor<[64], f32> no_inline {
mo.graph @broadcast_reshape_scalar_matrix(%arg0 : !mo.tensor<[1, 1], f32>) -> !mo.tensor<[64], f32>{
  // CHECK: %[[VAL0:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[1, 1], f32> -> !mo.tensor<[], f32>
  // CHECK: %[[VAL1:.*]] = mo.static.broadcast_to(%[[VAL0]]) : !mo.tensor<[], f32> -> !mo.tensor<[64], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[1, 1], f32> -> !mo.tensor<[64, 1], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: mo.output %[[VAL1]] : !mo.tensor<[64], f32>
  mo.output %1 : !mo.tensor<[64], f32>
// CHECK: }
}

//no change
// CHECK: mo.graph @broadcast_reshape_vector(%[[ARG0:.*]]: !mo.tensor<[2, 1], f32>) -> !mo.tensor<[64], f32> no_inline {
mo.graph @broadcast_reshape_vector(%arg0 : !mo.tensor<[2, 1], f32>) -> !mo.tensor<[64], f32>{
  // CHECK: %[[VAL0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2, 32], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[2, 1], f32> -> !mo.tensor<[2, 32], f32>
  // CHECK: %[[VAL1:.*]] = mo.static.reshape(%[[VAL0]]) : !mo.tensor<[2, 32], f32> -> !mo.tensor<[64], f32>
  %1 = mo.static.reshape(%0) : !mo.tensor<[2, 32], f32> -> !mo.tensor<[64], f32>
  // CHECK: mo.output %[[VAL1]] : !mo.tensor<[64], f32>
  mo.output %1 : !mo.tensor<[64], f32>
// CHECK: }
}


