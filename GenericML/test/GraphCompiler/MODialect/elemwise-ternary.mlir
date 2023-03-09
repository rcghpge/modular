// RUN: mo-opt %s | FileCheck %s

// CHECK: mo.graph @select(%[[COND:.*]]: !mo.tensor<[2, 3], bool>, %[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @select(%cond: !mo.tensor<[2, 3], bool>, %x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.select(%[[COND]], %[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], bool>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %0 = mo.select(%cond, %x, %y) : (!mo.tensor<[2, 3], bool>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
