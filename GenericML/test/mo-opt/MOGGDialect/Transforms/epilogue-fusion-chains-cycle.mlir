// RUN: mo-opt %s --mo-to-mogg --mogg-fuse-prologues-epilogues | FileCheck %s

// No kernels get fused
// CHECK: mogg.kernel
// CHECK: mo.transfer
// CHECK: mogg.kernel
// CHECK: mogg.concat
// CHECK: mogg.kernel

mo.graph @fn(%arg0: !mo.tensor<[5, 64], bf16, gpu:0>, %arg1: !mo.tensor<[5, 64], bf16, gpu:1>, %arg2: !mo.tensor<[2, 64], bf16, gpu:0>, %arg3: !mo.tensor<[3, 64], bf16, gpu:0>, %arg4: !mo.buffer<[537919488], ui8, gpu:0>, %arg5: !mo.buffer<[537919488], ui8, gpu:1>) -> !mo.tensor<[5, 64], bf16, gpu:1> {
  %0 = mo.chain.create()
  %output:2, %outChain = mo.distributed.allreduce.sum(%arg0, %arg1, %arg4, %arg5, %0) : (!mo.tensor<[5, 64], bf16, gpu:0>, !mo.tensor<[5, 64], bf16, gpu:1>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[5, 64], bf16, gpu:0>, !mo.tensor<[5, 64], bf16, gpu:1>, !mo.chain)
  %result, %outChain_0 = mo.transfer[%outChain] %arg2 : !mo.tensor<[2, 64], bf16, gpu:0> to <"gpu", 1>
  %outputs:4, %outChain_1 = mo.distributed.allgather(%arg3, %result, %arg4, %arg5, %outChain_0) : (!mo.tensor<[3, 64], bf16, gpu:0>, !mo.tensor<[2, 64], bf16, gpu:1>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[3, 64], bf16, gpu:0>, !mo.tensor<[2, 64], bf16, gpu:0>, !mo.tensor<[3, 64], bf16, gpu:1>, !mo.tensor<[2, 64], bf16, gpu:1>, !mo.chain)
  %1 = mogg.concat(%outputs#2, %outputs#3) : (!mo.tensor<[3, 64], bf16, gpu:1>, !mo.tensor<[2, 64], bf16, gpu:1>) -> !mo.tensor<[5, 64], bf16, gpu:1>
  %2 = mo.add(%1, %output#1) : !mo.tensor<[5, 64], bf16, gpu:1>
  mo.output %2 : !mo.tensor<[5, 64], bf16, gpu:1>
}
