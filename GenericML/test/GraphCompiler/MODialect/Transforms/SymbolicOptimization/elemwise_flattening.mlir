// RUN: mo-opt %s --mo-symbolic-optimize --cse | FileCheck %s

// CHECK: mo.graph @erf(%[[ARG0:.*]]: !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f32> {
mo.graph @erf(%arg0 : !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f32>{
  // CHECK: %[[VAL0:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: %[[VAL1:.*]] = mo.erf(%[[VAL0]]) : !mo.tensor<[64], f32>
  // CHECK: %[[VAL2:.*]] = mo.static.reshape(%[[VAL1]]) : !mo.tensor<[64], f32> -> !mo.tensor<[64, 1], f32>
  %out = mo.erf(%arg0) : !mo.tensor<[64, 1], f32>
  // CHECK: mo.output %[[VAL2]] : !mo.tensor<[64, 1], f32>
  mo.output %out : !mo.tensor<[64, 1], f32>
// CHECK: }
}

// CHECK: mo.graph @erf_with_device(%[[ARG0:.*]]: !mo.tensor<[64, 1], f32, dev:0>) -> !mo.tensor<[64, 1], f32, dev:0> {
mo.graph @erf_with_device(%arg0 : !mo.tensor<[64, 1], f32, dev:0>) -> !mo.tensor<[64, 1], f32, dev:0>{
  // CHECK: %[[OUT:.*]] = mo.erf(%[[ARG0]]) : !mo.tensor<[64, 1], f32, dev:0>
  %out = mo.erf(%arg0) : !mo.tensor<[64, 1], f32, dev:0>
  // CHECK: mo.output %[[OUT]] : !mo.tensor<[64, 1], f32, dev:0>
  mo.output %out : !mo.tensor<[64, 1], f32, dev:0>
// CHECK: }
}

mo.graph @cast(%arg0 : !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f64>{
  // CHECK: %[[VAL0:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: %[[VAL1:.*]] = mo.cast(%[[VAL0]]) : (!mo.tensor<[64], f32>) -> !mo.tensor<[64], f64>
  // CHECK: %[[VAL2:.*]] = mo.static.reshape(%[[VAL1]]) : !mo.tensor<[64], f64> -> !mo.tensor<[64, 1], f64>
  %out = mo.cast(%arg0) : (!mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f64>
  // CHECK: mo.output %[[VAL2]] : !mo.tensor<[64, 1], f64>
  mo.output %out : !mo.tensor<[64, 1], f64>
// CHECK: }
}

// CHECK: mo.graph @add(%[[ARG0:.*]]: !mo.tensor<[64, 1], f32>, %[[ARG1:.*]]: !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f32> {
mo.graph @add(%arg0 : !mo.tensor<[64, 1], f32>, %arg1 : !mo.tensor<[64, 1], f32>) -> !mo.tensor<[64, 1], f32>{
  // CHECK: %[[VAL0:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: %[[VAL1:.*]] = mo.static.reshape(%[[ARG1]]) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: %[[VAL2:.*]] = mo.add(%[[VAL0]], %[[VAL1]]) : !mo.tensor<[64], f32>
  // CHECK: %[[VAL3:.*]] = mo.static.reshape(%[[VAL2]]) : !mo.tensor<[64], f32> -> !mo.tensor<[64, 1], f32>
  %out = mo.add(%arg0, %arg1) : !mo.tensor<[64, 1], f32>
  // CHECK: mo.output %[[VAL3]] : !mo.tensor<[64, 1], f32>
  mo.output %out : !mo.tensor<[64, 1], f32>
}
// CHECK: }

// CHECK: mo.graph @pow(%[[ARG0:.*]]: !mo.tensor<[64, 1], f32>, %[[ARG1:.*]]: !mo.tensor<[64, 1], si32>) -> !mo.tensor<[64, 1], f32>
mo.graph @pow(%arg0 : !mo.tensor<[64, 1], f32>, %arg1: !mo.tensor<[64, 1], si32>) -> !mo.tensor<[64, 1], f32>{
  // CHECK: %[[RESHAPE1:.*]] = mo.static.reshape(%[[ARG0]]) : !mo.tensor<[64, 1], f32> -> !mo.tensor<[64], f32>
  // CHECK: %[[RESHAPE2:.*]] = mo.static.reshape(%[[ARG1]]) : !mo.tensor<[64, 1], si32> -> !mo.tensor<[64], si32>
  // CHECK: %[[POW:.*]] = mo.pow(%[[RESHAPE1]] : !mo.tensor<[64], f32>, %[[RESHAPE2]] : !mo.tensor<[64], si32>) : !mo.tensor<[64], f32>
  %out = mo.pow(%arg0 : !mo.tensor<[64, 1], f32>, %arg1 : !mo.tensor<[64, 1], si32>) : !mo.tensor<[64, 1], f32>
  // CHECK: %[[ANSWER:.*]] = mo.static.reshape(%[[POW]]) : !mo.tensor<[64], f32> -> !mo.tensor<[64, 1], f32>
  // CHECK: mo.output %[[ANSWER]] : !mo.tensor<[64, 1], f32>
  mo.output %out : !mo.tensor<[64, 1], f32>
}

// No effect
// CHECK: mo.graph @erf_non_vector(%[[ARG0:.*]]: !mo.tensor<[64, 4], f32>) -> !mo.tensor<[64, 4], f32> {
mo.graph @erf_non_vector(%arg0 : !mo.tensor<[64, 4], f32>) -> !mo.tensor<[64, 4], f32>{
  // CHECK: %[[VAL0:.*]] = mo.erf(%[[ARG0]]) : !mo.tensor<[64, 4], f32>
  %out = mo.erf(%arg0) : !mo.tensor<[64, 4], f32>
  // CHECK: mo.output %[[VAL0]] : !mo.tensor<[64, 4], f32>
  mo.output %out : !mo.tensor<[64, 4], f32>
// CHECK: }
}

// No effect
// CHECK: mo.graph @erf_unit_square_matrix(%[[ARG0:.*]]: !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32> {
mo.graph @erf_unit_square_matrix(%arg0 : !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32>{
  // CHECK: %[[VAL0:.*]] = mo.erf(%[[ARG0]]) : !mo.tensor<[1, 1], f32>
  %out = mo.erf(%arg0) : !mo.tensor<[1, 1], f32>
  // CHECK: mo.output %[[VAL0]] : !mo.tensor<[1, 1], f32>
  mo.output %out : !mo.tensor<[1, 1], f32>
// CHECK: }
}

// No effect
// CHECK: mo.graph @add_unit_square_matrix(%[[ARG0:.*]]: !mo.tensor<[1, 1], f32>, %[[ARG1:.*]]: !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32> {
mo.graph @add_unit_square_matrix(%arg0 : !mo.tensor<[1, 1], f32>, %arg1 : !mo.tensor<[1, 1], f32>) -> !mo.tensor<[1, 1], f32>{
  // CHECK: %[[VAL0:.*]] = mo.add(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[1, 1], f32>
  %out = mo.add(%arg0, %arg1) : !mo.tensor<[1, 1], f32>
  // CHECK: mo.output %[[VAL0]] : !mo.tensor<[1, 1], f32>
  mo.output %out : !mo.tensor<[1, 1], f32>
// CHECK: }
}

// CHECK: mo.graph @add_bs1(%[[ARG0:.*]]: !mo.tensor<[1, 64], f32>, %[[ARG1:.*]]: !mo.tensor<[1, 64], f32>) -> !mo.tensor<[1, 64], f32> {
mo.graph @add_bs1(%arg0 : !mo.tensor<[1, 64], f32>, %arg1 : !mo.tensor<[1, 64], f32>) -> !mo.tensor<[1, 64], f32>{
  // CHECK: %[[VAL0:.*]] = mo.add(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[1, 64], f32>
  %out = mo.add(%arg0, %arg1) : !mo.tensor<[1, 64], f32>
  // CHECK: mo.output %[[VAL0]] : !mo.tensor<[1, 64], f32>
  mo.output %out : !mo.tensor<[1, 64], f32>
// CHECK: }
}
