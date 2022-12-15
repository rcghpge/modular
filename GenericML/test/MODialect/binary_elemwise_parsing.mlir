// RUN: mo-opt %s | FileCheck %s

// CHECK: mo.graph @add(%[[ARG0:.*]]: !mo.tensor<2x3xf32>, %[[ARG1:.*]]: !mo.tensor<2x3xf32>) -> !mo.tensor<2x3xf32> {
mo.graph @add(%arg0: !mo.tensor<2x3xf32>, %arg1: !mo.tensor<2x3xf32>) -> (!mo.tensor<2x3xf32>) {
  // CHECK-NEXT: mo.add(%[[ARG0]], %[[ARG1]]) -> !mo.tensor<2x3xf32>
  %0 = mo.add(%arg0, %arg1) -> !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// CHECK: mo.graph @sub(%[[ARG0:.*]]: !mo.tensor<2x3xf32>, %[[ARG1:.*]]: !mo.tensor<2x3xf32>) -> !mo.tensor<2x3xf32> {
mo.graph @sub(%arg0: !mo.tensor<2x3xf32>, %arg1: !mo.tensor<2x3xf32>) -> (!mo.tensor<2x3xf32>) {
  // CHECK-NEXT: mo.sub(%[[ARG0]], %[[ARG1]]) -> !mo.tensor<2x3xf32>
  %0 = mo.sub(%arg0, %arg1) -> !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// CHECK: mo.graph @mul(%[[ARG0:.*]]: !mo.tensor<2x3xf32>, %[[ARG1:.*]]: !mo.tensor<2x3xf32>) -> !mo.tensor<2x3xf32> {
mo.graph @mul(%arg0: !mo.tensor<2x3xf32>, %arg1: !mo.tensor<2x3xf32>) -> (!mo.tensor<2x3xf32>) {
  // CHECK-NEXT: mo.mul(%[[ARG0]], %[[ARG1]]) -> !mo.tensor<2x3xf32>
  %0 = mo.mul(%arg0, %arg1) -> !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// CHECK: mo.graph @div(%[[ARG0:.*]]: !mo.tensor<2x3xf32>, %[[ARG1:.*]]: !mo.tensor<2x3xf32>) -> !mo.tensor<2x3xf32> {
mo.graph @div(%arg0: !mo.tensor<2x3xf32>, %arg1: !mo.tensor<2x3xf32>) -> (!mo.tensor<2x3xf32>) {
  // CHECK-NEXT: mo.div(%[[ARG0]], %[[ARG1]]) -> !mo.tensor<2x3xf32>
  %0 = mo.div(%arg0, %arg1) -> !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// CHECK: mo.graph @softmax(%[[ARG0:.*]]: !mo.tensor<2x3xf32>) -> !mo.tensor<2x3xf32> {
mo.graph @softmax(%arg0: !mo.tensor<2x3xf32>) -> (!mo.tensor<2x3xf32>) {
  // CHECK-NEXT: mo.softmax(%[[ARG0]]) : (!mo.tensor<2x3xf32>) -> !mo.tensor<2x3xf32>
  %0 = mo.softmax(%arg0) : (!mo.tensor<2x3xf32>) -> !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

