// RUN: mo-opt %s --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: mt --input-shapes=4x2 --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'reduce_split_dim_dyn':
// CHECK: 'reduce_split_dim_dyn' returned tensor<2x2x1xf32> [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
mo.graph @reduce_split_dim_dyn<A, B>(%arg0: !mo.tensor<[A, B], f32>) -> (!mo.tensor<[2, 2, 1], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
  %1 = mo.reduce.add(%arg0, %axis) : (!mo.tensor<[A, B], f32>, !mo.tensor<[], ui32>) -> !mo.tensor<[A, 1], f32>
  %2 = mo.split_dim[0](%1): (!mo.tensor<[A, 1], f32>) -> !mo.tensor<[2, 2, 1], f32>
  mo.output %2 : !mo.tensor<[2, 2, 1], f32>
}
