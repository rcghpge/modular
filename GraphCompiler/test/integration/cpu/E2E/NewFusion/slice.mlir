// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4x4 --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=4x4 --result-output-style=full %t.legacy.mlir | FileCheck %s

// Static case: slice a known 4x4 input down to the 2x2 sub-block at
// rows/cols [1:3, 1:3), verifying the actual sliced values (not just shape).
//   1  2  3  4
//   5 [6  7] 8
//   9 [10 11] 12
//  13 14 15 16
// -> [6, 7, 10, 11]

// CHECK-LABEL: Running 'slice_static':
// CHECK: 'slice_static' returned tensor<2x2xf32> [6.000000e+00, 7.000000e+00, 1.000000e+01, 1.100000e+01]
mo.graph @slice_static() -> (!mo.tensor<[2, 2], f32>) {
  %input = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0> : tensor<16xf32>} : !mo.tensor<[4, 4], f32>
  %start = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[4, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// Dynamic case: symbolic input shape (filled with all-ones by `mt`), static
// slice bounds -- proving the dynamic-shape path runs correctly too.

// CHECK-LABEL: Running 'slice_dynamic':
// CHECK: 'slice_dynamic' returned tensor<2x2xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
mo.graph @slice_dynamic<M, N>(%input: !mo.tensor<[M, N], f32>) -> (!mo.tensor<[2, 2], f32>) {
  %start = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[M, N], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// Chained case: slicing the result of another slice, verifying the second
// slice's own view composes correctly against the first (not against the
// original, unsliced input).
//  1  2  3  4  5  6
//  7  8  9 10 11 12
// 13 14 15 16 17 18
// 19 20 21 22 23 24
// 25 26 27 28 29 30
// 31 32 33 34 35 36
//
// First slice [1:5, 1:5) -> 4x4:
//  8  9 10 11
// 14 15 16 17
// 20 21 22 23
// 26 27 28 29
//
// Second slice [1:3, 1:3) of that -> 2x2: [15, 16, 21, 22]

// CHECK-LABEL: Running 'slice_chained':
// CHECK: 'slice_chained' returned tensor<2x2xf32> [1.500000e+01, 1.600000e+01, 2.100000e+01, 2.200000e+01]
mo.graph @slice_chained() -> (!mo.tensor<[2, 2], f32>) {
  %input = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0> : tensor<36xf32>} : !mo.tensor<[6, 6], f32>
  %start0 = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop0 = mo.constant {value = #M.dense_array<5, 5> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step0 = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %mid = mo.slice(%input, %start0, %stop0, %step0) : (
    !mo.tensor<[6, 6], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[4, 4], f32>
  %start1 = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop1 = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step1 = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%mid, %start1, %stop1, %step1) : (
    !mo.tensor<[4, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}
