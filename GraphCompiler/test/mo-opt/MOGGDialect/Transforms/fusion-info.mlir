// RUN: mo-opt --gc-pipeline %s --mlir-print-ir-before=jit -o %t.mlir 2>&1 | FileCheck %s

// Check that fusionInfo is printed on kernel stubs after fusion passes.
// Under the new API we expect the matmul+add to be fused followed by sequence fusing the cumsum.
// CHECK: fusionInfo = Sequence({{.*}}Epilogue("mo.matmul":{{[0-9]+}}, "mo.add":{{[0-9]+}}, operandIndices=[0]))
mo.graph @fusion_info_simple(
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

    %final_res = mo.cumsum(%bias_res) {axis = 1 : index, exclusive = 1 : index, reverse = 0 : index}: (
        !mo.tensor<[20, 40], f32>
    ) -> !mo.tensor<[20, 40], f32>
    mo.output %final_res: !mo.tensor<[20, 40], f32>
}
