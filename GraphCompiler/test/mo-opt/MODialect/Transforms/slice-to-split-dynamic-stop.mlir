// Tests for SliceToSplitPattern<AllowDynamicStop=true>, which PatternFusion
// enables so adjacent slices with a shape-materialized (non-constant) stop
// still combine into a split (the dynamic-shape RoPE rotate-half case).
//
// The split results are routed to mo.output so the SplitToSlicePattern run at
// the end of mo-pattern-fusion leaves the split in place ("Split used in
// output"), making the fold observable.

// RUN: mo-opt %s --mo-pattern-fusion -split-input-file | FileCheck %s

// Dynamic stop, unit step on the sliced axis: two adjacent slices that together
// cover axis 1 are combined into a split even though their stop operands are
// graph arguments rather than constants. Each split size is taken from the
// statically known slice result extent (2 and 2).

// CHECK-LABEL: mo.graph @slice_to_split_dynamic_stop
// CHECK:       %[[SIZES:.*]] = mo.constant {value = #M.dense_array<2, 2>
// CHECK:       %[[SPLIT:.*]]:2 = mo.split(%arg0, %[[SIZES]]) {axis = 1 : index}
// CHECK:       mo.output %[[SPLIT]]#0, %[[SPLIT]]#1

mo.graph @slice_to_split_dynamic_stop(%arg0: !mo.tensor<[2, 4], f32>, %stop0: !mo.tensor<[2], si64>, %stop1: !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start0 = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start1 = mo.constant {value = #M.dense_array<0, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice0 = mo.slice(%arg0, %start0, %stop0, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %slice1 = mo.slice(%arg0, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %slice0, %slice1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// -----

// No conversion: dynamic stop with non-unit step on the sliced axis. With a
// dynamic stop the split sizes come from the result extents, which is only
// valid for unit step, so the pattern bails and the slices remain.

// CHECK-LABEL: mo.graph @no_conversion_dynamic_stop_step_not_one
// CHECK-NOT:   mo.split
// CHECK:       mo.slice
// CHECK:       mo.slice

mo.graph @no_conversion_dynamic_stop_step_not_one(%arg0: !mo.tensor<[2, 4], f32>, %stop0: !mo.tensor<[2], si64>, %stop1: !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %start0 = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start1 = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %slice0 = mo.slice(%arg0, %start0, %stop0, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %slice1 = mo.slice(%arg0, %start1, %stop1, %step) : (!mo.tensor<[2, 4], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %slice0, %slice1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
