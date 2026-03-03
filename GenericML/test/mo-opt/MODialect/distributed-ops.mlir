// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @all_reduce_sum(%[[IN0:.*]]: !mo.tensor<[2, 3], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[2, 3], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[2, 3], f32, gpu:2>
  mo.graph @all_reduce_sum(%in0: !mo.tensor<[2, 3], f32, gpu:0>, %in1: !mo.tensor<[2, 3], f32, gpu:1>, %in2: !mo.tensor<[2, 3], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.allreduce.sum(%[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, {{.*}}) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, !mo.chain)
  %out:3, %outChain = mo.distributed.allreduce.sum(%in0, %in1, %in2, %sig0, %sig1, %sig2, %ch) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, !mo.chain)
  mo.output %out#0, %out#1, %out#2 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>
}

// CHECK: mo.graph @all_reduce_sum_dynamic(%[[IN0:.*]]: !mo.tensor<[2, ?], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[2, ?], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[2, ?], f32, gpu:2>
mo.graph @all_reduce_sum_dynamic(%in0: !mo.tensor<[2, ?], f32, gpu:0>, %in1: !mo.tensor<[2, ?], f32, gpu:1>, %in2: !mo.tensor<[2, ?], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.allreduce.sum(%[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) : (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, {{.*}}) -> (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, !mo.chain)
  %out:3, %outChain = mo.distributed.allreduce.sum(%in0, %in1, %in2, %sig0, %sig1, %sig2, %ch) : (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain) -> (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, !mo.chain)
  mo.output %out#0, %out#1, %out#2 : !mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>
}

// CHECK-LABEL: mo.graph @all_reduce_folds
// CHECK: (%[[IN0:.*]]: !mo.tensor<[2, 17], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[CH:.*]]: !mo.chain)
mo.graph @all_reduce_folds(%in0: !mo.tensor<[2, 17], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %ch: !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain) {
  // CHECK-NOT: mo.distributed.allreduce.sum
  // CHECK: mo.output %[[IN0]], %[[CH]]
  %out, %outChain = mo.distributed.allreduce.sum(%in0, %sig0, %ch) : (!mo.tensor<[2, 17], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain)
  mo.output %out, %outChain : !mo.tensor<[2, 17], f32, gpu:0>, !mo.chain
}

// CHECK: mo.graph @reduce_scatter_sum(%[[IN0:.*]]: !mo.tensor<[3, 6], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[3, 6], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[3, 6], f32, gpu:2>
mo.graph @reduce_scatter_sum(%in0: !mo.tensor<[3, 6], f32, gpu:0>, %in1: !mo.tensor<[3, 6], f32, gpu:1>, %in2: !mo.tensor<[3, 6], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[3, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.reducescatter.sum(%[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) : (!mo.tensor<[3, 6], f32, gpu:0>, !mo.tensor<[3, 6], f32, gpu:1>, !mo.tensor<[3, 6], f32, gpu:2>, {{.*}}) -> (!mo.tensor<[3, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  %out:3, %outChain = mo.distributed.reducescatter.sum(%in0, %in1, %in2, %sig0, %sig1, %sig2, %ch) : (!mo.tensor<[3, 6], f32, gpu:0>, !mo.tensor<[3, 6], f32, gpu:1>, !mo.tensor<[3, 6], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain) -> (!mo.tensor<[3, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  mo.output %out#0, %out#1, %out#2 : !mo.tensor<[3, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>
}

// Ragged case: 7 elements / 3 GPUs = 2 base + 1 remainder. GPU 0 gets 3, others get 2.
// CHECK: mo.graph @reduce_scatter_sum_ragged(%[[IN0:.*]]: !mo.tensor<[3, 7], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[3, 7], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[3, 7], f32, gpu:2>
mo.graph @reduce_scatter_sum_ragged(%in0: !mo.tensor<[3, 7], f32, gpu:0>, %in1: !mo.tensor<[3, 7], f32, gpu:1>, %in2: !mo.tensor<[3, 7], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.reducescatter.sum(%[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) : (!mo.tensor<[3, 7], f32, gpu:0>, !mo.tensor<[3, 7], f32, gpu:1>, !mo.tensor<[3, 7], f32, gpu:2>, {{.*}}) -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  %out:3, %outChain = mo.distributed.reducescatter.sum(%in0, %in1, %in2, %sig0, %sig1, %sig2, %ch) : (!mo.tensor<[3, 7], f32, gpu:0>, !mo.tensor<[3, 7], f32, gpu:1>, !mo.tensor<[3, 7], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain) -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  mo.output %out#0, %out#1, %out#2 : !mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>
}

// CHECK: mo.graph @reduce_scatter_sum_dynamic(%[[IN0:.*]]: !mo.tensor<[3, ?], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[3, ?], f32, gpu:1>
mo.graph @reduce_scatter_sum_dynamic(%in0: !mo.tensor<[3, ?], f32, gpu:0>, %in1: !mo.tensor<[3, ?], f32, gpu:1>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.reducescatter.sum(%[[IN0]], %[[IN1]], {{.*}}) : (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>, {{.*}}) -> (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>, !mo.chain)
  %out:2, %outChain = mo.distributed.reducescatter.sum(%in0, %in1, %sig0, %sig1, %ch) : (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1 : !mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>
}

// CHECK-LABEL: mo.graph @reduce_scatter_folds
// CHECK: (%[[IN0:.*]]: !mo.tensor<[2, 17], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[CH:.*]]: !mo.chain)
mo.graph @reduce_scatter_folds(%in0: !mo.tensor<[2, 17], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %ch: !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain) {
  // CHECK-NOT: mo.distributed.reducescatter.sum
  // CHECK: mo.output %[[IN0]], %[[CH]]
  %out, %outChain = mo.distributed.reducescatter.sum(%in0, %sig0, %ch) : (!mo.tensor<[2, 17], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain)
  mo.output %out, %outChain : !mo.tensor<[2, 17], f32, gpu:0>, !mo.chain
}
