// RUN: mo-opt %s --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'reduce_split_dim':
// CHECK: 'reduce_split_dim' returned tensor<2x2x1xf32> [3.000000e+00, 7.000000e+00, 1.100000e+01, 1.500000e+01]
mo.graph @reduce_split_dim() -> (!mo.tensor<[2, 2, 1], f32>) {
  %0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<8xf32>} : !mo.tensor<[4, 2], f32>
  %axis = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
  %1 = mo.reduce.add(%0, %axis) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[], ui32>) -> !mo.tensor<[4, 1], f32>
  %2 = mo.split_dim[0](%1): (!mo.tensor<[4, 1], f32>) -> !mo.tensor<[2, 2, 1], f32>
  mo.output %2 : !mo.tensor<[2, 2, 1], f32>
}
