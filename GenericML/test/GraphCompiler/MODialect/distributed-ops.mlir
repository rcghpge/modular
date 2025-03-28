// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @all_reduce_sum(%[[IN0:.*]]: !mo.tensor<[2, 3], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[2, 3], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[2, 3], f32, gpu:2>
  mo.graph @all_reduce_sum(%in0: !mo.tensor<[2, 3], f32, gpu:0>, %in1: !mo.tensor<[2, 3], f32, gpu:1>, %in2: !mo.tensor<[2, 3], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[2, 3], f32, gpu:0>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.allreduce.sum([[CH:%.*]], %[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) : (!mo.chain, !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, {{.*}}) -> (!mo.chain, !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>)
  %outChain, %out:3 = mo.distributed.allreduce.sum(%ch, %in0, %in1, %in2, %sig0, %sig1, %sig2) : (!mo.chain, !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>) -> (!mo.chain, !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>)
  mo.output %out#0 : !mo.tensor<[2, 3], f32, gpu:0>
}

// CHECK: mo.graph @all_reduce_sum_dynamic(%[[IN0:.*]]: !mo.tensor<[2, ?], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[2, ?], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[2, ?], f32, gpu:2>
mo.graph @all_reduce_sum_dynamic(%in0: !mo.tensor<[2, ?], f32, gpu:0>, %in1: !mo.tensor<[2, ?], f32, gpu:1>, %in2: !mo.tensor<[2, ?], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[2, ?], f32, gpu:0>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.allreduce.sum([[CH:%.*]], %[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) : ({{.*}}, !mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, {{.*}}) -> (!mo.chain, !mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>)
  %outChain, %out:3 = mo.distributed.allreduce.sum(%ch, %in0, %in1, %in2, %sig0, %sig1, %sig2) : (!mo.chain, !mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>) -> (!mo.chain, !mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>)
  mo.output %out#0 : !mo.tensor<[2, ?], f32, gpu:0>
}
