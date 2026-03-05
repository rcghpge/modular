// RUN: mo-opt %s --mo-to-mgp | FileCheck %s

// The allreduce.sum should be fused with the add operations
// CHECK-LABEL: mef.callable @allreduce_sum_with_epilogue$exec
// CHECK: Epilogue(mo.distributed.allreduce.sum, mo.add, mo.add, operandIndices=[0, 1])
// CHECK: mef.output
mo.graph @allreduce_sum_with_epilogue(
  %input0: !mo.tensor<[5, 64], bf16, gpu:0>,
  %input1: !mo.tensor<[5, 64], bf16, gpu:1>,
  %residual0: !mo.tensor<[5, 64], bf16, gpu:0>,
  %residual1: !mo.tensor<[5, 64], bf16, gpu:1>,
  %buf0: !mo.buffer<[537919488], ui8, gpu:0>,
  %buf1: !mo.buffer<[537919488], ui8, gpu:1>
) -> (!mo.tensor<[5, 64], bf16, gpu:0>, !mo.tensor<[5, 64], bf16, gpu:1>) {
  %chain = mo.chain.create()

  // AllReduce sum across GPUs
  %output:2, %outChain = mo.distributed.allreduce.sum(%input0, %input1, %buf0, %buf1, %chain)
    : (!mo.tensor<[5, 64], bf16, gpu:0>,
       !mo.tensor<[5, 64], bf16, gpu:1>,
       !mo.buffer<[537919488], ui8, gpu:0>,
       !mo.buffer<[537919488], ui8, gpu:1>,
       !mo.chain)
    -> (!mo.tensor<[5, 64], bf16, gpu:0>, !mo.tensor<[5, 64], bf16, gpu:1>, !mo.chain)

  // Epilogue: add residuals
  %result0 = mo.add(%output#0, %residual0) : !mo.tensor<[5, 64], bf16, gpu:0>
  %result1 = mo.add(%output#1, %residual1) : !mo.tensor<[5, 64], bf16, gpu:1>

  mo.output %result0, %result1 : !mo.tensor<[5, 64], bf16, gpu:0>, !mo.tensor<[5, 64], bf16, gpu:1>
}
