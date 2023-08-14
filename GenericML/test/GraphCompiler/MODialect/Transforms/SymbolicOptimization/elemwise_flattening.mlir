// RUN: mo-opt %s --symbolic-optimize --cse | FileCheck %s

// CHECK: mo.graph @erf(%[[ARG0:.*]]: !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f32> no_inline {
mo.graph @erf(%arg0 : !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f32>{
  // CHECK: %[[VAL0:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: %[[VAL1:.*]] = mo.erf(%[[VAL0]]) : !mo.tensor<[64], f32>
  // CHECK: %[[VAL2:.*]] = mo.static.reshape(%[[VAL1]]) : !mo.tensor<[64], f32> -> !mo.tensor<[64, 1], f32>
  %out = mo.erf(%arg0) : !mo.tensor<[64, 1], f32>
  // CHECK: mo.output %[[VAL2]] : !mo.tensor<[64, 1], f32>
  mo.output %out : !mo.tensor<[64, 1], f32>
// CHECK: }
}

// CHECK: mo.graph @add(%[[ARG0:.*]]: !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f32> no_inline {
mo.graph @add(%arg0 : !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f32>{
  // CHECK: %[[VAL0:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: %[[VAL1:.*]] = mo.add(%[[VAL0]], %[[VAL0]]) : !mo.tensor<[64], f32>
  // CHECK: %[[VAL2:.*]] = mo.static.reshape(%[[VAL1]]) : !mo.tensor<[64], f32> -> !mo.tensor<[64, 1], f32>
  %out = mo.add(%arg0, %arg0) : !mo.tensor<[64, 1], f32>
  // CHECK: mo.output %[[VAL2]] : !mo.tensor<[64, 1], f32>
  mo.output %out : !mo.tensor<[64, 1], f32>
}
// CHECK: }

// No effect
// CHECK: mo.graph @erf_non_vector(%[[ARG0:.*]]: !mo.tensor<[64, 4], f32>) -> !mo.tensor<[64, 4], f32> no_inline {
mo.graph @erf_non_vector(%arg0 : !mo.tensor<[64, 4], f32>) -> !mo.tensor<[64, 4], f32>{
  // CHECK: %[[VAL0:.*]] = mo.erf(%[[ARG0]]) : !mo.tensor<[64, 4], f32>
  %out = mo.erf(%arg0) : !mo.tensor<[64, 4], f32>
  // CHECK: mo.output %[[VAL0]] : !mo.tensor<[64, 4], f32>
  mo.output %out : !mo.tensor<[64, 4], f32>
// CHECK: }
}

// No effect
// CHECK: mo.graph @erf_unit_square_matrix(%[[ARG0:.*]]: !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32> no_inline {
mo.graph @erf_unit_square_matrix(%arg0 : !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32>{
  // CHECK: %[[VAL0:.*]] = mo.erf(%[[ARG0]]) : !mo.tensor<[1, 1], f32>
  %out = mo.erf(%arg0) : !mo.tensor<[1, 1], f32>
  // CHECK: mo.output %[[VAL0]] : !mo.tensor<[1, 1], f32>
  mo.output %out : !mo.tensor<[1, 1], f32>
// CHECK: }
}

// No effect
// CHECK: mo.graph @add_unit_square_matrix(%[[ARG0:.*]]: !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32> no_inline {
mo.graph @add_unit_square_matrix(%arg0 : !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32>{
  // CHECK: %[[VAL0:.*]] = mo.add(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[1, 1], f32>
  %out = mo.add(%arg0, %arg0) : !mo.tensor<[1, 1], f32>
  // CHECK: mo.output %[[VAL0]] : !mo.tensor<[1, 1], f32>
  mo.output %out : !mo.tensor<[1, 1], f32>
// CHECK: }
}

// CHECK: mo.graph @add_bs1(%[[ARG0:.*]]: !mo.tensor<[1, 64], f32>) -> !mo.tensor<[1, 64], f32> no_inline {
mo.graph @add_bs1(%arg0 : !mo.tensor<[1, 64], f32>) -> !mo.tensor<[1, 64], f32>{
  // CHECK: %[[VAL0:.*]] = mo.add(%[[ARG0]], %[[ARG0]]) : !mo.tensor<[1, 64], f32>
  %out = mo.add(%arg0, %arg0) : !mo.tensor<[1, 64], f32>
  // CHECK: mo.output %[[VAL0]] : !mo.tensor<[1, 64], f32>
  mo.output %out : !mo.tensor<[1, 64], f32>
// CHECK: }
}


