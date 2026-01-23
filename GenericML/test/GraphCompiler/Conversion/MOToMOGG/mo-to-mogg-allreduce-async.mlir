// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" | FileCheck %s

// Check that distributed allreduce kernels always launch asynchronously to
// prevent blocking worker threads.
mo.graph @async_allreduce(%in0: !mo.tensor<[2], f32, gpu:0>, %in1: !mo.tensor<[2], f32, gpu:1>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>) {
  %chain = mo.chain.create()
  %out0, %chain0 = mo.distributed.allreduce.sum(%in0, %in1, %sig0, %sig1, %chain) {device = #M.device_ref<"gpu", 0>} : (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2], f32, gpu:0>, !mo.chain)
  %out1, %chain1 = mo.distributed.allreduce.sum(%in0, %in1, %sig0, %sig1, %chain) {device = #M.device_ref<"gpu", 1>} : (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2], f32, gpu:1>, !mo.chain)
  mo.output %out0, %out1 : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>
}

// CHECK-LABEL: mo.graph @async_allreduce
// CHECK: = mogg.kernel(
// CHECK: isAsync = true
// CHECK: = mogg.kernel(
// CHECK: isAsync = true
