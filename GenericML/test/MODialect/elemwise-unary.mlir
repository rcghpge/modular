// RUN: mo-opt %s | FileCheck %s

// CHECK: mo.graph @abs(%[[ARG0:.*]]: !mo.tensor<4x5xf32>) -> !mo.tensor<4x5xf32> {
mo.graph @abs(%arg0: !mo.tensor<4x5xf32>) -> (!mo.tensor<4x5xf32>) {
  // CHECK-NEXT: mo.abs(%[[ARG0]]) : !mo.tensor<4x5xf32>
  %0 = mo.abs(%arg0) : !mo.tensor<4x5xf32>
  mo.output %0 : !mo.tensor<4x5xf32>
}

// CHECK: mo.graph @relu(%[[ARG0:.*]]: !mo.tensor<4x5xf32>) -> !mo.tensor<4x5xf32> {
mo.graph @relu(%arg0: !mo.tensor<4x5xf32>) -> (!mo.tensor<4x5xf32>) {
  // CHECK-NEXT: mo.relu(%[[ARG0]]) : !mo.tensor<4x5xf32>
  %0 = mo.relu(%arg0) : !mo.tensor<4x5xf32>
  mo.output %0 : !mo.tensor<4x5xf32>
}

// CHECK: mo.graph @gelu(%[[ARG0:.*]]: !mo.tensor<4x5xf32>) -> !mo.tensor<4x5xf32> {
mo.graph @gelu(%arg0: !mo.tensor<4x5xf32>) -> (!mo.tensor<4x5xf32>) {
  // CHECK-NEXT: mo.gelu(%[[ARG0]]) : !mo.tensor<4x5xf32>
  %0 = mo.gelu(%arg0) : !mo.tensor<4x5xf32>
  mo.output %0 : !mo.tensor<4x5xf32>
}

// CHECK: mo.graph @erf(%[[ARG0:.*]]: !mo.tensor<4x5xf32>) -> !mo.tensor<4x5xf32> {
mo.graph @erf(%arg0: !mo.tensor<4x5xf32>) -> (!mo.tensor<4x5xf32>) {
  // CHECK-NEXT: mo.erf(%[[ARG0]]) : !mo.tensor<4x5xf32>
  %0 = mo.erf(%arg0) : !mo.tensor<4x5xf32>
  mo.output %0 : !mo.tensor<4x5xf32>
}

// CHECK: mo.graph @tanh(%[[ARG0:.*]]: !mo.tensor<4x5xf32>) -> !mo.tensor<4x5xf32> {
mo.graph @tanh(%arg0: !mo.tensor<4x5xf32>) -> (!mo.tensor<4x5xf32>) {
  // CHECK-NEXT: mo.tanh(%[[ARG0]]) : !mo.tensor<4x5xf32>
  %0 = mo.tanh(%arg0) : !mo.tensor<4x5xf32>
  mo.output %0 : !mo.tensor<4x5xf32>
}

// CHECK: mo.graph @exp(%[[ARG0:.*]]: !mo.tensor<4x5xf32>) -> !mo.tensor<4x5xf32> {
mo.graph @exp(%arg0: !mo.tensor<4x5xf32>) -> (!mo.tensor<4x5xf32>) {
  // CHECK-NEXT: mo.exp(%[[ARG0]]) : !mo.tensor<4x5xf32>
  %0 = mo.exp(%arg0) : !mo.tensor<4x5xf32>
  mo.output %0 : !mo.tensor<4x5xf32>
}
