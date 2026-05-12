// RUN: mo-opt %s --mogg-pattern-fusion | FileCheck %s

// Two parallel multi-use concat ops each consumed by a slice (and also output
// directly). ConcatSlicePattern fires first on each, producing two
// fused_concat_slice ops. DualConcatSlicePattern then pairs them into one
// dual_fused_concat_slice op.
// CHECK-LABEL: mo.graph @dual_fused_concat_slice_fusion
// CHECK: %[[C0:.*]], %[[S0:.*]], %[[C1:.*]], %[[S1:.*]] = mo.dual_fused_concat_slice
// CHECK-NOT: mo.fused_concat_slice
// CHECK-NOT: mo.concat
// CHECK-NOT: mo.slice
// CHECK: mo.output %[[C0]], %[[S0]], %[[C1]], %[[S1]]
mo.graph @dual_fused_concat_slice_fusion(
  %arg0: !mo.tensor<[32, 4096, 1536], f32>,
  %arg1: !mo.tensor<[32, 2048, 1536], f32>,
  %arg2: !mo.tensor<[32, 4096, 1536], f32>,
  %arg3: !mo.tensor<[32, 2048, 1536], f32>
) -> (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>,
      !mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %starts = mo.constant {value = #M.dense_array<0, -4096, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops = mo.constant {value = #M.dense_array<32, 6144, 1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>

  // Both concats have 2 uses each (the slice + the direct output), so
  // ConcatSlicePattern fires on each, then DualConcatSlicePattern pairs them.
  %concat0 = mo.concat[%axis : !mo.tensor<[], si64>] (%arg0, %arg1) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 2048, 1536], f32>) -> !mo.tensor<[32, 6144, 1536], f32>
  %concat1 = mo.concat[%axis : !mo.tensor<[], si64>] (%arg2, %arg3) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 2048, 1536], f32>) -> !mo.tensor<[32, 6144, 1536], f32>
  %slice0 = mo.slice(%concat0, %starts, %stops, %steps) : (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  %slice1 = mo.slice(%concat1, %starts, %stops, %steps) : (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  mo.output %concat0, %slice0, %concat1, %slice1 : !mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>
}

// Two parallel pairs with different numbers of concat inputs; each concat has
// an extra use so ConcatSlicePattern fires first.
// CHECK-LABEL: mo.graph @dual_fused_concat_slice_fusion_diff_num_inputs
// CHECK: %[[C0:.*]], %[[S0:.*]], %[[C1:.*]], %[[S1:.*]] = mo.dual_fused_concat_slice
// CHECK: mo.output %[[C0]], %[[S0]], %[[C1]], %[[S1]]
mo.graph @dual_fused_concat_slice_fusion_diff_num_inputs(
  %arg0: !mo.tensor<[32, 4096, 1536], f32>,
  %arg1: !mo.tensor<[32, 2048, 1536], f32>,
  %arg2: !mo.tensor<[32, 4096, 1536], f32>,
  %arg3: !mo.tensor<[32, 1024, 1536], f32>,
  %arg4: !mo.tensor<[32, 1024, 1536], f32>
) -> (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>,
      !mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %starts0 = mo.constant {value = #M.dense_array<0, -4096, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops0 = mo.constant {value = #M.dense_array<32, 6144, 1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %starts1 = mo.constant {value = #M.dense_array<0, -4096, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops1 = mo.constant {value = #M.dense_array<32, 6144, 1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>

  // concat0 has 2 inputs, concat1 has 3 inputs; both have 2 uses each.
  %concat0 = mo.concat[%axis : !mo.tensor<[], si64>] (%arg0, %arg1) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 2048, 1536], f32>) -> !mo.tensor<[32, 6144, 1536], f32>
  %concat1 = mo.concat[%axis : !mo.tensor<[], si64>] (%arg2, %arg3, %arg4) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 1024, 1536], f32>, !mo.tensor<[32, 1024, 1536], f32>) -> !mo.tensor<[32, 6144, 1536], f32>
  %slice0 = mo.slice(%concat0, %starts0, %stops0, %steps) : (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  %slice1 = mo.slice(%concat1, %starts1, %stops1, %steps) : (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  mo.output %concat0, %slice0, %concat1, %slice1 : !mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>
}

// Cannot dual-fuse: only one fused_concat_slice (no second partner).
// CHECK-LABEL: mo.graph @fail_dual_no_partner
// CHECK: mo.fused_concat_slice
// CHECK-NOT: mo.dual_fused_concat_slice
mo.graph @fail_dual_no_partner(
  %arg0: !mo.tensor<[32, 4096, 1536], f32>,
  %arg1: !mo.tensor<[32, 2048, 1536], f32>
) -> (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %starts = mo.constant {value = #M.dense_array<0, -4096, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops = mo.constant {value = #M.dense_array<32, 6144, 1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>

  // Only one concat with 2 uses → one fused_concat_slice, no pair for dual.
  %concat = mo.concat[%axis : !mo.tensor<[], si64>] (%arg0, %arg1) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 2048, 1536], f32>) -> !mo.tensor<[32, 6144, 1536], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  mo.output %concat, %slice : !mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>
}

// Cannot dual-fuse: fused_concat_slice ops have different concat axes.
// CHECK-LABEL: mo.graph @fail_dual_different_axes
// CHECK-NOT: mo.dual_fused_concat_slice
mo.graph @fail_dual_different_axes(
  %arg0: !mo.tensor<[32, 4096, 1536], f32>,
  %arg1: !mo.tensor<[32, 2048, 1536], f32>,
  %arg2: !mo.tensor<[32, 4096, 1536], f32>,
  %arg3: !mo.tensor<[32, 4096, 768], f32>
) -> (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>,
      !mo.tensor<[32, 4096, 2304], f32>, !mo.tensor<[32, 4096, 1536], f32>) {
  %axis1 = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %axis2 = mo.constant {value = #M.dense_array<2> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %starts0 = mo.constant {value = #M.dense_array<0, -4096, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops0 = mo.constant {value = #M.dense_array<32, 6144, 1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %starts1 = mo.constant {value = #M.dense_array<0, 0, -1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops1 = mo.constant {value = #M.dense_array<32, 4096, 2304> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>

  // Both concats have 2 uses, but their axes differ → no dual fusion.
  %concat0 = mo.concat[%axis1 : !mo.tensor<[], si64>] (%arg0, %arg1) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 2048, 1536], f32>) -> !mo.tensor<[32, 6144, 1536], f32>
  %concat1 = mo.concat[%axis2 : !mo.tensor<[], si64>] (%arg2, %arg3) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 4096, 768], f32>) -> !mo.tensor<[32, 4096, 2304], f32>
  %slice0 = mo.slice(%concat0, %starts0, %stops0, %steps) : (!mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  %slice1 = mo.slice(%concat1, %starts1, %stops1, %steps) : (!mo.tensor<[32, 4096, 2304], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  mo.output %concat0, %slice0, %concat1, %slice1 : !mo.tensor<[32, 6144, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 4096, 2304], f32>, !mo.tensor<[32, 4096, 1536], f32>
}
