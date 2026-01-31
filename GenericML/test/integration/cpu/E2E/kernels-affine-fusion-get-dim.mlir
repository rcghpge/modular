
// RUN: mo-opt %s --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x2 --input-shapes=2x2 --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'matmul_merge_dim_with_param_rebind':
// CHECK: 'matmul_merge_dim_with_param_rebind' returned tensor<4xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]

mo.graph @matmul_merge_dim_with_param_rebind<A,B,C,AC>(%arg0: !mo.tensor<[A, B], f32>, %arg1: !mo.tensor<[B, C], f32>) -> !mo.tensor<[AC], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[A, B], f32>, !mo.tensor<[B, C], f32>) -> !mo.tensor<[A, C], f32>
  %1 = mo.merge_dim[0] (%0) : (!mo.tensor<[A, C], f32>) -> !mo.tensor<[mul_no_wrap(A, C)], f32>
  %2 = mo.rebind(%1) : !mo.tensor<[mul_no_wrap(A, C)], f32> -> !mo.tensor<[AC], f32>
  mo.output %2: !mo.tensor<[AC], f32>
}
