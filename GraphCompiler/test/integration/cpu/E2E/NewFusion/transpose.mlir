// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x64 --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x64 --result-output-style=full %t.legacy.mlir | FileCheck %s

// 2D swap: perm = [1, 0] is a plain matrix transpose, verifying the actual
// transposed values (not just shape).
//   1 2 3
//   4 5 6
// -> 1 4
//    2 5
//    3 6

// CHECK-LABEL: Running 'transpose_swap':
// CHECK: 'transpose_swap' returned tensor<3x2xf32> [1.000000e+00, 4.000000e+00, 2.000000e+00, 5.000000e+00, 3.000000e+00, 6.000000e+00]
mo.graph @transpose_swap() -> (!mo.tensor<[3, 2], f32>) {
  %input = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// 3D cyclic permutation: perm = [2, 0, 1], out.dim[i] = in.dim[perm[i]].
// Input is 1..24 (shape [2, 3, 4], row-major); verifying the full permuted
// output against its actual expected values.

// CHECK-LABEL: Running 'transpose_3d':
// CHECK: 'transpose_3d' returned tensor<4x2x3xf32> [1.000000e+00, 5.000000e+00, 9.000000e+00, 1.300000e+01, 1.700000e+01, 2.100000e+01, 2.000000e+00, 6.000000e+00, 1.000000e+01, 1.400000e+01, 1.800000e+01, 2.200000e+01, 3.000000e+00, 7.000000e+00, 1.100000e+01, 1.500000e+01, 1.900000e+01, 2.300000e+01, 4.000000e+00, 8.000000e+00, 1.200000e+01, 1.600000e+01, 2.000000e+01, 2.400000e+01]
mo.graph @transpose_3d() -> (!mo.tensor<[4, 2, 3], f32>) {
  %input = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0> : tensor<24xf32>} : !mo.tensor<[2, 3, 4], f32>
  %perm = mo.constant {value = #M.dense_array<2, 0, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[4, 2, 3], f32>
  mo.output %res : !mo.tensor<[4, 2, 3], f32>
}

// Dynamic case: symbolic input shape (filled with all-ones by `mt`), static
// perm -- proving the symbolic-dim path resolves correctly too.

// CHECK-LABEL: Running 'transpose_dynamic':
// CHECK: 'transpose_dynamic' returned tensor<64x2xf32>
mo.graph @transpose_dynamic<N>(%input: !mo.tensor<[N, 64], f32>) -> (!mo.tensor<[64, N], f32>) {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (
    !mo.tensor<[N, 64], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[64, N], f32>
  mo.output %res : !mo.tensor<[64, N], f32>
}
