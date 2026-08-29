// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-views --map-raise-back-contiguous-views | FileCheck %s

// A slice view kernel that ViewFuser already fused with another view (here,
// a chain of two slices collapsed into one kernel with two map.index.slice
// hops) is not the original, untouched single-view lowering anymore, so
// RaiseBackContiguousViews leaves it as a map.iter.foreach -- raising it back
// to mogg.slice (which represents only a single slice, not a composed
// chain) would be incorrect.

// CHECK-LABEL: mo.graph @fusedSliceNotRaised
// CHECK-NOT:     mogg.slice
// CHECK:         map.iter.foreach<[3, 64]>
// CHECK:         map.index.slice<#map.index.slice.spec<[6, 64] -> [3, 64]
// CHECK:         map.index.slice<#map.index.slice.spec<[10, 64] -> [6, 64]
mo.graph @fusedSliceNotRaised(%input: !mo.tensor<[10, 64], f32>) -> !mo.tensor<[3, 64], f32> {
  %start1 = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop1 = mo.constant {value = #M.dense_array<6, 64> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step1 = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %mid = mo.slice(%input, %start1, %stop1, %step1) : (
    !mo.tensor<[10, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[6, 64], f32>
  %start2 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop2 = mo.constant {value = #M.dense_array<4, 64> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step2 = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%mid, %start2, %stop2, %step2) : (
    !mo.tensor<[6, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[3, 64], f32>
  mo.output %res : !mo.tensor<[3, 64], f32>
}
