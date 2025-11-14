
// Check std:err
// RUN: mo-opt --mo-to-mgp %s --mlir-print-ir-before=jit-compile-kernels -o %t.mlir 2>&1 | FileCheck %s

// Under new api we expect the matmul add to be fused followed by sequence fusing the cumsum
// CHECK: summary = {{.*}}mo.matmul:{{[0-9]+}}+mo.add:{{[0-9]+}}
mo.graph @summary_string_simple(
    %arg0: !mo.tensor<[20, 30], f32>,
    %arg1: !mo.tensor<[30, 40], f32>,
    %arg2: !mo.tensor<[20, 40], f32>
) -> !mo.tensor<[20, 40], f32>{
    %matmul_res = mo.matmul(
        %arg0,
        %arg1
    ): (
        !mo.tensor<[20, 30], f32>,
        !mo.tensor<[30, 40], f32>
    ) -> !mo.tensor<[20, 40], f32>

    %bias_res = mo.add(%matmul_res, %arg2): !mo.tensor<[20, 40], f32>

    %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[], si32>
    %final_res = mo.cumsum(%bias_res, %axis) {exclusive = 1 : index, reverse = 0 : index}: (
        !mo.tensor<[20, 40], f32>,
        !mo.tensor<[], si32>
    ) -> !mo.tensor<[20, 40], f32>
    mo.output %final_res: !mo.tensor<[20, 40], f32>
}
