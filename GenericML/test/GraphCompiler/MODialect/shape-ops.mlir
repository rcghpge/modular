// RUN: mo-opt %s | mo-opt | FileCheck %s

// CHECK: mo.graph @broadcast_to(%[[FROM:.*]]: !mo.tensor<[3], f32>, %[[TO:.*]]: !mosh.ape) -> !mo.tensor<[2, 3], f32> {
mo.graph @broadcast_to(%from: !mo.tensor<[3], f32>, %to: !mosh.ape) -> !mo.tensor<[2, 3], f32> {
  // CHECK: [[RES:.*]] = mo.shape.broadcast_to(%[[FROM]], %[[TO]]) : (!mo.tensor<[3], f32>, !mosh.ape) -> !mo.tensor<[2, 3], f32>
  %0 = mo.shape.broadcast_to(%from, %to) : (!mo.tensor<[3], f32>, !mosh.ape) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @reshape(%[[FROM:.*]]: !mo.tensor<[3], f32>, %[[TO:.*]]: !mosh.ape) -> !mo.tensor<[2, 3], f32> {
mo.graph @reshape(%from: !mo.tensor<[3], f32>, %to: !mosh.ape) -> !mo.tensor<[2, 3], f32> {
  // CHECK: [[RES:.*]] = mo.shape.reshape(%[[FROM]], %[[TO]]) : (!mo.tensor<[3], f32>, !mosh.ape) -> !mo.tensor<[2, 3], f32>
  %0 = mo.shape.reshape(%from, %to) : (!mo.tensor<[3], f32>, !mosh.ape) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
