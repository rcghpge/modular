// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" | FileCheck %s

// Check that distributed allreduce kernels (MO_MultiDevice ops) are launched
// synchronously. These ops manage their own cross-device parallelism internally
// (via TaskGroup) and must complete before the graph runtime proceeds.
mo.graph @async_allreduce(%in0: !mo.tensor<[2], f32, gpu:0>, %in1: !mo.tensor<[2], f32, gpu:1>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>) {
  %chain = mo.chain.create()
  %out:2, %outChain = mo.distributed.allreduce.sum(%in0, %in1, %sig0, %sig1, %chain) : (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1 : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>
}

// CHECK-LABEL: mo.graph @async_allreduce
// CHECK: = mogg.kernel(
// CHECK: isAsync = false
