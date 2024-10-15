// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @all_reduce_sum(%[[IN0:.*]]: !mo.tensor<[2, 3], f32, cuda:0>, %[[IN1:.*]]: !mo.tensor<[2, 3], f32, cuda:1>, %[[IN2:.*]]: !mo.tensor<[2, 3], f32, cuda:2>) -> !mo.tensor<[2, 3], f32, cuda:0> {
mo.graph @all_reduce_sum(%in0: !mo.tensor<[2, 3], f32, cuda:0>, %in1: !mo.tensor<[2, 3], f32, cuda:1>, %in2: !mo.tensor<[2, 3], f32, cuda:2>) -> (!mo.tensor<[2, 3], f32, cuda:0>) {
  // CHECK-NEXT: mo.distributed.allreduce.sum(%[[IN0]], %[[IN1]], %[[IN2]]) : (!mo.tensor<[2, 3], f32, cuda:0>, !mo.tensor<[2, 3], f32, cuda:1>, !mo.tensor<[2, 3], f32, cuda:2>) -> (!mo.tensor<[2, 3], f32, cuda:0>, !mo.tensor<[2, 3], f32, cuda:1>, !mo.tensor<[2, 3], f32, cuda:2>)
  %out:3 = mo.distributed.allreduce.sum(%in0, %in1, %in2) : (!mo.tensor<[2, 3], f32, cuda:0>, !mo.tensor<[2, 3], f32, cuda:1>, !mo.tensor<[2, 3], f32, cuda:2>) -> (!mo.tensor<[2, 3], f32, cuda:0>, !mo.tensor<[2, 3], f32, cuda:1>, !mo.tensor<[2, 3], f32, cuda:2>)
  mo.output %out#0 : !mo.tensor<[2, 3], f32, cuda:0>
}

// CHECK: mo.graph @all_reduce_sum_dynamic(%[[IN0:.*]]: !mo.tensor<[2, ?], f32, cuda:0>, %[[IN1:.*]]: !mo.tensor<[2, ?], f32, cuda:1>, %[[IN2:.*]]: !mo.tensor<[2, ?], f32, cuda:2>) -> !mo.tensor<[2, ?], f32, cuda:0> {
mo.graph @all_reduce_sum_dynamic(%in0: !mo.tensor<[2, ?], f32, cuda:0>, %in1: !mo.tensor<[2, ?], f32, cuda:1>, %in2: !mo.tensor<[2, ?], f32, cuda:2>) -> (!mo.tensor<[2, ?], f32, cuda:0>) {
  // CHECK-NEXT: mo.distributed.allreduce.sum(%[[IN0]], %[[IN1]], %[[IN2]]) : (!mo.tensor<[2, ?], f32, cuda:0>, !mo.tensor<[2, ?], f32, cuda:1>, !mo.tensor<[2, ?], f32, cuda:2>) -> (!mo.tensor<[2, ?], f32, cuda:0>, !mo.tensor<[2, ?], f32, cuda:1>, !mo.tensor<[2, ?], f32, cuda:2>)
  %out:3 = mo.distributed.allreduce.sum(%in0, %in1, %in2) : (!mo.tensor<[2, ?], f32, cuda:0>, !mo.tensor<[2, ?], f32, cuda:1>, !mo.tensor<[2, ?], f32, cuda:2>) -> (!mo.tensor<[2, ?], f32, cuda:0>, !mo.tensor<[2, ?], f32, cuda:1>, !mo.tensor<[2, ?], f32, cuda:2>)
  mo.output %out#0 : !mo.tensor<[2, ?], f32, cuda:0>
}
