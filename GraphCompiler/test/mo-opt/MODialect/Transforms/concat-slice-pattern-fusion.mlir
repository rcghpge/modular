// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// CHECK-LABEL: mo.graph @concat_slice_fusion
// CHECK: %[[OUT0:.*]], %[[OUT1:.*]] = mo.composite.concat_slice
// CHECK: mo.output %[[OUT0]], %[[OUT1]]
mo.graph @concat_slice_fusion(
  %arg0: !mo.tensor<[32, 4096, 1536], f32>,
  %arg1: !mo.tensor<[32, 1, 1536], f32>
) -> (!mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %starts = mo.constant {value = #M.dense_array<0, -4096, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops = mo.constant {value = #M.dense_array<32, 4097, 1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>

  %concat = mo.concat[1] (%arg0, %arg1) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 1, 1536], f32>) -> !mo.tensor<[32, 4097, 1536], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  mo.output %concat, %slice : !mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>
}


// Cannot fuse since the concat is only used by the slice.
// CHECK-LABEL: mo.graph @fail_concat_slice_fusion_single_use
// CHECK-NOT: mo.composite.concat_slice
mo.graph @fail_concat_slice_fusion_single_use(
  %arg0: !mo.tensor<[32, 4096, 1536], f32>,
  %arg1: !mo.tensor<[32, 1, 1536], f32>
) -> (!mo.tensor<[32, 4096, 1536], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %starts = mo.constant {value = #M.dense_array<0, -4096, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops = mo.constant {value = #M.dense_array<32, 4097, 1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>

  %concat = mo.concat[1] (%arg0, %arg1) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 1, 1536], f32>) -> !mo.tensor<[32, 4097, 1536], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  mo.output %slice : !mo.tensor<[32, 4096, 1536], f32>
}

// Cannot fuse since the slice doesn't need to be materialized as a kernel.
// CHECK-LABEL: mo.graph @fail_concat_slice_fusion_trivial
// CHECK-NOT: mo.composite.concat_slice
mo.graph @fail_concat_slice_fusion_trivial(
  %arg0: !mo.tensor<[1, 4096, 1536], f32>,
  %arg1: !mo.tensor<[1, 1, 1536], f32>
) -> (!mo.tensor<[1, 4097, 1536], f32>, !mo.tensor<[1, 4096, 1536], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %starts = mo.constant {value = #M.dense_array<0, -4096, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops = mo.constant {value = #M.dense_array<32, 4097, 1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>

  %concat = mo.concat[1] (%arg0, %arg1) : (!mo.tensor<[1, 4096, 1536], f32>, !mo.tensor<[1, 1, 1536], f32>) -> !mo.tensor<[1, 4097, 1536], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[1, 4097, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[1, 4096, 1536], f32>
  mo.output %concat, %slice : !mo.tensor<[1, 4097, 1536], f32>, !mo.tensor<[1, 4096, 1536], f32>
}

// Cannot fuse since the slice doesn't need to be materialized as a kernel.
// CHECK-LABEL: mo.graph @fail_concat_slice_fusion_inner_dim
// CHECK-NOT: mo.composite.concat_slice
mo.graph @fail_concat_slice_fusion_inner_dim(
  %arg0: !mo.tensor<[32, 4096, 1536], f32>,
  %arg1: !mo.tensor<[32, 1, 1536], f32>
) -> (!mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[32, 4097, 100], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %starts = mo.constant {value = #M.dense_array<0, 0, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops = mo.constant {value = #M.dense_array<32, 4097, 100> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>

  %concat = mo.concat[1] (%arg0, %arg1) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 1, 1536], f32>) -> !mo.tensor<[32, 4097, 1536], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4097, 100], f32>
  mo.output %concat, %slice : !mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[32, 4097, 100], f32>
}

// Cannot fuse since the slice has a negative step
// CHECK-LABEL: mo.graph @fail_concat_slice_fusion_neg_step
// CHECK-NOT: mo.composite.concat_slice
mo.graph @fail_concat_slice_fusion_neg_step(
  %arg0: !mo.tensor<[32, 4096, 1536], f32>,
  %arg1: !mo.tensor<[32, 1, 1536], f32>
) -> (!mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %starts = mo.constant {value = #M.dense_array<0, -4096, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %stops = mo.constant {value = #M.dense_array<32, 4097, 1536> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %steps = mo.constant {value = #M.dense_array<-1, 1, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>

  %concat = mo.concat[1] (%arg0, %arg1) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 1, 1536], f32>) -> !mo.tensor<[32, 4097, 1536], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  mo.output %concat, %slice : !mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>
}


// Cannot fuse since start / steps are not constants
// CHECK-LABEL: mo.graph @fail_concat_slice_fusion_dyn_operands
// CHECK-NOT: mo.composite.concat_slice
mo.graph @fail_concat_slice_fusion_dyn_operands(
  %arg0: !mo.tensor<[32, 4096, 1536], f32>,
  %arg1: !mo.tensor<[32, 1, 1536], f32>,
  %starts: !mo.tensor<[3], si64>,
  %stops: !mo.tensor<[3], si64>,
  %steps: !mo.tensor<[3], si64>
) -> (!mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %concat = mo.concat[1] (%arg0, %arg1) : (!mo.tensor<[32, 4096, 1536], f32>, !mo.tensor<[32, 1, 1536], f32>) -> !mo.tensor<[32, 4097, 1536], f32>
  %slice = mo.slice(%concat, %starts, %stops, %steps) : (!mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[32, 4096, 1536], f32>
  mo.output %concat, %slice : !mo.tensor<[32, 4097, 1536], f32>, !mo.tensor<[32, 4096, 1536], f32>
}
