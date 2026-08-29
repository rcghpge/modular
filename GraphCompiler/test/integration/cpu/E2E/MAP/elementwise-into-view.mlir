// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x3 --input-shapes=2x3 --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x3 --input-shapes=2x3 --result-output-style=full %t.legacy.mlir | FileCheck %s

// ViewFuser fuses an elementwise producer into a view consumer of a
// different shape (e.g. mo.add feeding mo.transpose) -- the transpose's own
// index-transform, not the raw shared index, becomes the add's loop index.
// Verifying the actual transposed-and-added values (not just shape) proves
// this numerically, not just at the IR-shape level.
//
//   1 2 3        2 3 4        2 5
//   4 5 6  + 1 = 5 6 7  -> T = 3 6
//                             4 7

// CHECK-LABEL: Running 'add_into_transpose':
// CHECK: 'add_into_transpose' returned tensor<3x2xf32> [2.000000e+00, 5.000000e+00, 3.000000e+00, 6.000000e+00, 4.000000e+00, 7.000000e+00]
mo.graph @add_into_transpose() -> (!mo.tensor<[3, 2], f32>) {
  %x = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %y = mo.constant {value = #M.dense_array<1.0, 1.0, 1.0, 1.0, 1.0, 1.0> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %added = mo.add(%x, %y) : !mo.tensor<[2, 3], f32>
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.transpose(%added, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// Dynamic case: symbolic input shape (filled with all-ones by `mt`), static
// perm -- proving the fusion also runs correctly through the symbolic-dim
// path.

// CHECK-LABEL: Running 'add_into_transpose_dynamic':
// CHECK: 'add_into_transpose_dynamic' returned tensor<3x2xf32>
mo.graph @add_into_transpose_dynamic<N>(%x: !mo.tensor<[N, 3], f32>, %y: !mo.tensor<[N, 3], f32>) -> (!mo.tensor<[3, N], f32>) {
  %added = mo.add(%x, %y) : !mo.tensor<[N, 3], f32>
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.transpose(%added, %perm) : (!mo.tensor<[N, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, N], f32>
  mo.output %res : !mo.tensor<[3, N], f32>
}
