// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.legacy.mlir | FileCheck %s

// `start` is a graph argument (filled with all-1s by `mt`), not a
// `mo.constant`, so it isn't known at compile time -- same numbers as
// Conversion/MAPToMOGG's static case, since `start` happens to resolve to
// the same [1, 1].
//   1  2  3  4
//   5 [6  7] 8
//   9 [10 11] 12
//  13 14 15 16
// -> [6, 7, 10, 11]

// CHECK-LABEL: Running 'slice_start_argument':
// CHECK: 'slice_start_argument' returned tensor<2x2xf32> [6.000000e+00, 7.000000e+00, 1.000000e+01, 1.100000e+01]
mo.graph @slice_start_argument(%start: !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>) {
  %input = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0> : tensor<16xf32>} : !mo.tensor<[4, 4], f32>
  %stop = mo.constant {value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[4, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}
