// TODO(GEX-2983): Fix the requirement of accelerators being present
// RUN: EMIT_MOJO=1 mo-opt %s --mo-to-mgp | FileCheck %s

// Test that epilogue fusion with distributed operations works correctly in the
// emit-mojo path. This tests the case where TensorFusedOp's user is not a
// CallExecuteOp after fusion transformations.

// CHECK-LABEL: mgp.model @matmul_allreduce_with_epilogue
// The matmul_allreduce should be fused with the add operations
// CHECK: mo.distributed.matmul_allreduce{{.*}}+mo.add
// CHECK: mef.output
mo.graph @matmul_allreduce_with_epilogue<total_seq_len>(
  %input0: !mo.tensor<[total_seq_len, 7168], bf16, gpu:0>,
  %input1: !mo.tensor<[total_seq_len, 7168], bf16, gpu:1>,
  %input2: !mo.tensor<[total_seq_len, 7168], bf16, gpu:2>,
  %input3: !mo.tensor<[total_seq_len, 7168], bf16, gpu:3>,
  %weight0: !mo.tensor<[8192, 7168], bf16, gpu:0>,
  %weight1: !mo.tensor<[8192, 7168], bf16, gpu:1>,
  %weight2: !mo.tensor<[8192, 7168], bf16, gpu:2>,
  %weight3: !mo.tensor<[8192, 7168], bf16, gpu:3>,
  %residual0: !mo.tensor<[total_seq_len, 8192], bf16, gpu:0>,
  %residual1: !mo.tensor<[total_seq_len, 8192], bf16, gpu:1>,
  %residual2: !mo.tensor<[total_seq_len, 8192], bf16, gpu:2>,
  %residual3: !mo.tensor<[total_seq_len, 8192], bf16, gpu:3>,
  %buf0: !mo.buffer<[537919488], ui8, gpu:0>,
  %buf1: !mo.buffer<[537919488], ui8, gpu:1>,
  %buf2: !mo.buffer<[537919488], ui8, gpu:2>,
  %buf3: !mo.buffer<[537919488], ui8, gpu:3>,
  %chain: !mo.chain
) -> (!mo.tensor<[total_seq_len, 8192], bf16, gpu:0>,
      !mo.tensor<[total_seq_len, 8192], bf16, gpu:1>,
      !mo.tensor<[total_seq_len, 8192], bf16, gpu:2>,
      !mo.tensor<[total_seq_len, 8192], bf16, gpu:3>,
      !mo.chain) {
  // matmul_allreduce produces 4 outputs (one per GPU)
  %outputs:4, %outChain = mo.distributed.matmul_allreduce(
    %input0, %input1, %input2, %input3,
    %weight0, %weight1, %weight2, %weight3,
    %buf0, %buf1, %buf2, %buf3, %chain
  ) : (!mo.tensor<[total_seq_len, 7168], bf16, gpu:0>,
       !mo.tensor<[total_seq_len, 7168], bf16, gpu:1>,
       !mo.tensor<[total_seq_len, 7168], bf16, gpu:2>,
       !mo.tensor<[total_seq_len, 7168], bf16, gpu:3>,
       !mo.tensor<[8192, 7168], bf16, gpu:0>,
       !mo.tensor<[8192, 7168], bf16, gpu:1>,
       !mo.tensor<[8192, 7168], bf16, gpu:2>,
       !mo.tensor<[8192, 7168], bf16, gpu:3>,
       !mo.buffer<[537919488], ui8, gpu:0>,
       !mo.buffer<[537919488], ui8, gpu:1>,
       !mo.buffer<[537919488], ui8, gpu:2>,
       !mo.buffer<[537919488], ui8, gpu:3>,
       !mo.chain)
    -> (!mo.tensor<[total_seq_len, 8192], bf16, gpu:0>,
        !mo.tensor<[total_seq_len, 8192], bf16, gpu:1>,
        !mo.tensor<[total_seq_len, 8192], bf16, gpu:2>,
        !mo.tensor<[total_seq_len, 8192], bf16, gpu:3>,
        !mo.chain)

  // Epilogue fusion: add residuals to each output
  %result0 = mo.add(%outputs#0, %residual0) : !mo.tensor<[total_seq_len, 8192], bf16, gpu:0>
  %result1 = mo.add(%outputs#1, %residual1) : !mo.tensor<[total_seq_len, 8192], bf16, gpu:1>
  %result2 = mo.add(%outputs#2, %residual2) : !mo.tensor<[total_seq_len, 8192], bf16, gpu:2>
  %result3 = mo.add(%outputs#3, %residual3) : !mo.tensor<[total_seq_len, 8192], bf16, gpu:3>

  mo.output %result0, %result1, %result2, %result3, %outChain
    : !mo.tensor<[total_seq_len, 8192], bf16, gpu:0>,
      !mo.tensor<[total_seq_len, 8192], bf16, gpu:1>,
      !mo.tensor<[total_seq_len, 8192], bf16, gpu:2>,
      !mo.tensor<[total_seq_len, 8192], bf16, gpu:3>,
      !mo.chain
}

// CHECK-LABEL: mgp.model @allreduce_sum_with_epilogue
// The allreduce.sum should be fused with the add operation
// CHECK: mo.distributed.allreduce.sum{{.*}}+mo.add
// CHECK: mef.output
mo.graph @allreduce_sum_with_epilogue(
  %input0: !mo.tensor<[5, 64], bf16, gpu:0>,
  %input1: !mo.tensor<[5, 64], bf16, gpu:1>,
  %residual: !mo.tensor<[5, 64], bf16, gpu:1>,
  %buf0: !mo.buffer<[537919488], ui8, gpu:0>,
  %buf1: !mo.buffer<[537919488], ui8, gpu:1>
) -> !mo.tensor<[5, 64], bf16, gpu:1> {
  %chain = mo.chain.create()

  // AllReduce sum across GPUs
  %output, %outChain = mo.distributed.allreduce.sum(%input0, %input1, %buf0, %buf1, %chain)
    {device = #M.device_ref<"gpu", 1>} : (!mo.tensor<[5, 64], bf16, gpu:0>,
                                !mo.tensor<[5, 64], bf16, gpu:1>,
                                !mo.buffer<[537919488], ui8, gpu:0>,
                                !mo.buffer<[537919488], ui8, gpu:1>,
                                !mo.chain)
    -> (!mo.tensor<[5, 64], bf16, gpu:1>, !mo.chain)

  // Epilogue: add residual
  %result = mo.add(%output, %residual) : !mo.tensor<[5, 64], bf16, gpu:1>

  mo.output %result : !mo.tensor<[5, 64], bf16, gpu:1>
}
