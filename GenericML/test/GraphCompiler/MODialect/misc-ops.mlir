// RUN: mo-opt %s --canonicalize | mo-opt | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BroadcastShapeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @broadcast_shape(%[[ARG0:.*]]: !mo.tensor<[3], si64>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
mo.graph @broadcast_shape(%arg0: !mo.tensor<[3], si64>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64> {
  // CHECK: %[[RES:.*]] = mo.broadcast_shape(%[[ARG0:.*]], %[[ARG1:.*]]) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[3], si64>
  mo.output %res : !mo.tensor<[3], si64>
}

// CHECK: mo.graph @broadcast_shape_i32(%[[ARG0:.*]]: !mo.tensor<[3], si32>, %[[ARG1:.*]]: !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32>
mo.graph @broadcast_shape_i32(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32> {
  // CHECK: %[[RES:.*]] = mo.broadcast_shape(%[[ARG0:.*]], %[[ARG1:.*]]) : (!mo.tensor<[3], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32>
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[3], si32>
  mo.output %res : !mo.tensor<[3], si32>
}

// CHECK: mo.graph @broadcast_shape_fold() -> !mo.tensor<[3], si32> {
mo.graph @broadcast_shape_fold() -> !mo.tensor<[3], si32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.constant {value = #M.dense_array<5, 10, 42> : !M.array<3xsi32>} : !mo.tensor<[3], si32>
  %s0 = mo.constant {value = #M.dense_array<5, 10, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %s1 = mo.constant {value = #M.dense_array<10, 42> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %res = mo.broadcast_shape(%s0, %s1) : (!mo.tensor<[3], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[3], si32>
  mo.output %res : !mo.tensor<[3], si32>
}

// CHECK: mo.graph @broadcast_shape_partially_ranked<[[D0:.*]]>(%[[ARG0:.*]]: !mo.tensor<[3], si32>, %[[ARG1:.*]]: !mo.tensor<[[[D0]]], si32>) -> !mo.tensor<[4], si32>
mo.graph @broadcast_shape_partially_ranked<D0>(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[D0], si32>) -> !mo.tensor<[4], si32> {
  // CHECK: %[[RES:.*]] = mo.broadcast_shape(%[[ARG0:.*]], %[[ARG1:.*]]) : (!mo.tensor<[3], si32>, !mo.tensor<[[[D0]]], si32>) -> !mo.tensor<[4], si32>
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[D0], si32>) -> !mo.tensor<[4], si32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[4], si32>
  mo.output %res : !mo.tensor<[4], si32>
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @broadcast_same_rank(%[[FROM:.*]]: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @broadcast_same_rank(%from: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  // CHECK: %[[TO:.*]] = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %to  = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.broadcast_to(%[[FROM]], %[[TO]]) : (!mo.tensor<[1, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[1, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @broadcast_diff_rank(%[[FROM:.*]]: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 3], f32>
mo.graph @broadcast_diff_rank(%from: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 3], f32> {
  // CHECK: %[[TO:.*]] = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %to  = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.broadcast_to(%[[FROM]], %[[TO]]) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @concat(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32> {
mo.graph @concat(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32> {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK: mo.concat[%[[VAL0]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32>
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>) -> !mo.tensor<[2, 8], f32>
  mo.output %0 : !mo.tensor<[2, 8], f32>
}

// CHECK: mo.graph @concat_dyn_rank(%[[ARG0:.*]]: !mo.tensor<?, f32>, %[[ARG1:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
mo.graph @concat_dyn_rank(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %ax = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK: mo.concat[%[[VAL0]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// CHECK: mo.graph @concat_canonicalize(
mo.graph @concat_canonicalize(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2, ?], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 0], f32>, %[[AX:.*]]: !mo.tensor<[], si64>
  %arg0: !mo.tensor<[2, ?], f32>, %arg1: !mo.tensor<[2, 0], f32>, %ax: !mo.tensor<[], si64>
// CHECK-SAME: ) -> !mo.tensor<[2, ?], f32> {
) -> !mo.tensor<[2, ?], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.concat[%[[AX]] : !mo.tensor<[], si64>] (%[[ARG0]], %[[ARG0]]) : (
  // CHECK-SAME:   !mo.tensor<[2, ?], f32>, !mo.tensor<[2, ?], f32>) -> !mo.tensor<[2, ?], f32>
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg0, %arg1, %arg0) : (
    !mo.tensor<[2, ?], f32>, !mo.tensor<[2, 0], f32>, !mo.tensor<[2, ?], f32>) -> !mo.tensor<[2, ?], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, ?], f32>
  mo.output %0 : !mo.tensor<[2, ?], f32>
}

// CHECK: mo.graph @concat_canonicalize_to_rebind<[[N:.*]], [[K:.*]] -> [[M:.*]]>(
mo.graph @concat_canonicalize_to_rebind<N, K -> M>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[2, [[N]]], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 0], f32>, %[[AX:.*]]: !mo.tensor<[], si64>
  %arg0: !mo.tensor<[2, N], f32>, %arg1: !mo.tensor<[2, 0], f32>, %ax: !mo.tensor<[], si64>
// CHECK-SAME: ) -> !mo.tensor<[2, [[M]]], f32> {
) -> !mo.tensor<[2, M], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[2, N], f32> -> !mo.tensor<[2, K], f32>
  %0 = mo.concat[%ax: !mo.tensor<[], si64>](%arg1, %arg0, %arg1) : (
    !mo.tensor<[2, 0], f32>, !mo.tensor<[2, N], f32>, !mo.tensor<[2, 0], f32>) -> !mo.tensor<[2, K], f32>
  // CHECK-NEXT: mo.output<[[K]]> %[[RES]] : !mo.tensor<[2, [[K]]], f32>
  mo.output<K> %0 : !mo.tensor<[2, K], f32>
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @split_static(%[[ARG0:.*]]: !mo.tensor<[6], si32>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) {
mo.graph @split_static(%arg0: !mo.tensor<[6], si32>, %arg1: !mo.tensor<[2], si64>) -> (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %ax = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  // CHECK: %{{.*}} = mo.split[%[[VAL0]] : !mo.tensor<[], si32>] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[6], si32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>)
  %out1, %out2 = mo.split[%ax: !mo.tensor<[], si32>](%arg0, %arg1) : (!mo.tensor<[6], si32>, !mo.tensor<[2], si64>) -> (
    !mo.tensor<[?], si32>, !mo.tensor<[?], si32>
  )
  mo.output %out1, %out2 : !mo.tensor<[?], si32>, !mo.tensor<[?], si32>
}

// CHECK: mo.graph @split_negative_axis(%[[ARG0:.*]]: !mo.tensor<[6], si32>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) {
mo.graph @split_negative_axis(%arg0: !mo.tensor<[6], si32>, %arg1: !mo.tensor<[2], si64>) -> (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %ax = mo.constant {value = #M.dense_array<-1> : tensor<1xsi32>} : !mo.tensor<[], si32>
  // CHECK: %{{.*}} = mo.split[%[[VAL0]] : !mo.tensor<[], si32>] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[6], si32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>)
  %out1, %out2 = mo.split[%ax: !mo.tensor<[], si32>](%arg0, %arg1) : (!mo.tensor<[6], si32>, !mo.tensor<[2], si64>) -> (
    !mo.tensor<[?], si32>, !mo.tensor<[?], si32>
  )
  mo.output %out1, %out2 : !mo.tensor<[?], si32>, !mo.tensor<[?], si32>
}

// CHECK: mo.graph @split_dyn_rank(%[[ARG0:.*]]: !mo.tensor<?, si32>, %[[ARG1:.*]]: !mo.tensor<[2], si64>) -> (!mo.tensor<?, si32>, !mo.tensor<?, si32>) {
mo.graph @split_dyn_rank(%arg0: !mo.tensor<?, si32>, %arg1: !mo.tensor<[2], si64>) -> (!mo.tensor<?, si32>, !mo.tensor<?, si32>) {
  // CHECK: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %ax = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  // CHECK: %{{.*}} = mo.split[%[[VAL0]] : !mo.tensor<[], si32>] (%[[ARG0]], %[[ARG1]]) : (!mo.tensor<?, si32>, !mo.tensor<[2], si64>) -> (!mo.tensor<?, si32>, !mo.tensor<?, si32>)
  %out1, %out2 = mo.split[%ax: !mo.tensor<[], si32>](%arg0, %arg1) : (!mo.tensor<?, si32>, !mo.tensor<[2], si64>) -> (
    !mo.tensor<?, si32>, !mo.tensor<?, si32>
  )
  mo.output %out1, %out2 : !mo.tensor<?, si32>, !mo.tensor<?, si32>
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @gather(%[[INPUT:.*]]: !mo.tensor<[2, 2], f32>, %[[INDICES:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
mo.graph @gather(%input: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK: mo.gather(%[[INPUT]], %[[INDICES]], %[[AXIS]]) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @gather_dyn_rank_param(%[[INPUT:.*]]: !mo.tensor<?, f32>, %[[INDICES:.*]]: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
mo.graph @gather_dyn_rank_param(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK: mo.gather(%[[INPUT]], %[[INDICES]], %[[AXIS]]) : (!mo.tensor<?, f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  %0 = mo.gather(%input, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @gather_empty_input_folder
mo.graph @gather_empty_input_folder(%indices: !mo.tensor<?, si32>) -> !mo.tensor<[0], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.constant {value = #M.dense_array<> : tensor<0xsi64>} : !mo.tensor<[0], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %0 = mo.constant {value = #M.dense_array<> : !M.array<0xsi64>} : !mo.tensor<[0], si64>
  %1 = mo.gather(%0, %indices, %axis) : (!mo.tensor<[0], si64>, !mo.tensor<?, si32>, !mo.tensor<[], si64>) -> !mo.tensor<[0], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[0], si64>
  mo.output %1 : !mo.tensor<[0], si64>
}

// CHECK: mo.graph @gather_empty_input_index
mo.graph @gather_empty_input_index(%input: !mo.tensor<?, si32>) -> !mo.tensor<[0], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.constant {value = #M.dense_array<> : tensor<0xsi64>} : !mo.tensor<[0], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %indices = mo.constant {value = #M.dense_array<> : !M.array<0xsi64>} : !mo.tensor<[0], si64>
  %1 = mo.gather(%input, %indices, %axis) : (!mo.tensor<?, si32>, !mo.tensor<[0], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[0], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[0], si64>
  mo.output %1 : !mo.tensor<[0], si64>
}

// CHECK: mo.graph @gather_param_no_fold<[[SH0:.*]]: !mo.shape -> [[SH1:.*]]: !mo.shape>(
mo.graph @gather_param_no_fold<Sh0: !mo.shape -> Sh1: !mo.shape>(
    // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[SH0]], si32>) -> !mo.tensor<[[SH1]], si64> {
    %input: !mo.tensor<Sh0, si32>) -> !mo.tensor<Sh1, si64> {
  // CHECK-DAG: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK-DAG: %[[INDICES:.*]] = mo.constant {value = #M.dense_array<> : !M.array<0xsi64>} : !mo.tensor<[0], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %indices = mo.constant {value = #M.dense_array<> : !M.array<0xsi64>} : !mo.tensor<[0], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.gather<() -> [[SH2:.*]]: !mo.shape>(%[[INPUT]], %[[INDICES]], %[[AXIS]]) : (!mo.tensor<Sh0, si32>, !mo.tensor<[0], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, si64>
  %1 = mo.gather<() -> Sh2: !mo.shape>(%input, %indices, %axis) : (!mo.tensor<Sh0, si32>, !mo.tensor<[0], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, si64>
  // CHECK-NEXT: mo.output<:!mo.shape [[SH2]]> %[[RES]] : !mo.tensor<[[SH2]], si64>
  mo.output<:!mo.shape Sh2> %1 : !mo.tensor<Sh2, si64>
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @pad_static_output_shape(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 5], f32> {
mo.graph @pad_static_output_shape(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 5], f32> {
  // CHECK: %[[CONSTANT:.*]] = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %constant = mo.constant {value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
  // CHECK: %[[PADDINGS:.*]] = mo.constant {value = #M.dense_array<1, 0, 1, 1> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %paddings = mo.constant {value = #M.dense_array<1, 0, 1, 1> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  // CHECK: mo.pad(%[[INPUT]], %[[PADDINGS]], %[[CONSTANT]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[3, 5], f32>
  %res = mo.pad(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[3, 5], f32>
  mo.output %res : !mo.tensor<[3, 5], f32>
}

// CHECK: mo.graph @pad_dynamic_output_shape(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>, %[[PADDINGS:.*]]: !mo.tensor<[2, 2], si64>) -> !mo.tensor<[?, ?], f32> {
mo.graph @pad_dynamic_output_shape(%input: !mo.tensor<[2, 3], f32>, %paddings: !mo.tensor<[2, 2], si64>) -> !mo.tensor<[?, ?], f32> {
  // CHECK: %[[CONSTANT:.*]] = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} :
  %constant = mo.constant {value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
  // CHECK: mo.pad(%[[INPUT]], %[[PADDINGS]], %[[CONSTANT]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], f32>
  %res = mo.pad(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// CHECK: mo.graph @pad_unranked(%[[INPUT:.*]]: !mo.tensor<?, f32>, %[[PADDINGS:.*]]: !mo.tensor<[?, 2], si64>) -> !mo.tensor<?, f32> {
mo.graph @pad_unranked(%input: !mo.tensor<?, f32>, %paddings: !mo.tensor<[?, 2], si64>) -> !mo.tensor<?, f32> {
  // CHECK: %[[CONSTANT:.*]] = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<f32>} :
  %constant = mo.constant {value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
  // CHECK: mo.pad(%[[INPUT]], %[[PADDINGS]], %[[CONSTANT]]) : (!mo.tensor<?, f32>, !mo.tensor<[?, 2], si64>, !mo.tensor<[], f32>) -> !mo.tensor<?, f32>
  %res = mo.pad(%input, %paddings, %constant) : (!mo.tensor<?, f32>, !mo.tensor<[?, 2], si64>, !mo.tensor<[], f32>) -> !mo.tensor<?, f32>
  mo.output %res : !mo.tensor<?, f32>
}

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @range_float_dyn(%[[LIMIT:.*]]: !mo.tensor<[], f32>)
mo.graph @range_float_dyn(%limit: !mo.tensor<[], f32>) -> !mo.tensor<[?], f32> {
  // CHECK-DAG: %[[START:.*]] = mo.constant {value = #M.dense_array<0.{{0+}}e+00> : tensor<f32>} : !mo.tensor<[], f32>
  // CHECK-DAG: %[[STEP:.*]] = mo.constant {value = #M.dense_array<1.5{{0+}}e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %start = mo.constant {value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
  %step = mo.constant {value = #M.dense_array<1.5> : tensor<f32>} : !mo.tensor<[], f32>
  // CHECK: %[[RES:.*]] = mo.range(%[[START]], %[[LIMIT]], %[[STEP]]) : (
  %res = mo.range(%start, %limit, %step) : (
    // CHECK-SAME: !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?], f32>
    !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?], f32>
  mo.output %res : !mo.tensor<[?], f32>
}

// CHECK: mo.graph @range_int_static(%[[START:.*]]: !mo.tensor<[], si32>)
mo.graph @range_int_static(%startInt: !mo.tensor<[], si32>) -> !mo.tensor<[5], si32> {
  // CHECK-DAG: %[[STEP:.*]] = mo.constant {value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
  // CHECK-DAG: %[[LIMIT:.*]] = mo.constant {value = #M.dense_array<11> : tensor<si32>} : !mo.tensor<[], si32>
  %stepInt = mo.constant {value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
  %limitInt = mo.constant {value = #M.dense_array<11> : tensor<si32>} : !mo.tensor<[], si32>
  // CHECK: %[[RES:.*]] = mo.range(%[[START]], %[[LIMIT]], %[[STEP]]) : (
  %res = mo.range(%startInt, %limitInt, %stepInt) : (
    // CHECK-SAME: !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[5], si32>
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[5], si32>
  mo.output %res : !mo.tensor<[5], si32>
}

// CHECK: mo.graph @range_folder() -> (!mo.tensor<[5], ui32>, !mo.tensor<[4], si64>) {
mo.graph @range_folder() -> (!mo.tensor<[5], ui32>, !mo.tensor<[4], si64>) {
  // CHECK-DAG: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1, 3, 5, 7, 9> : tensor<5xui32>} : !mo.tensor<[5], ui32>
  %start = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
  %limit = mo.constant {value = #M.dense_array<11> : tensor<ui32>} : !mo.tensor<[], ui32>
  %step = mo.constant {value = #M.dense_array<2> : tensor<ui32>} : !mo.tensor<[], ui32>
  %0 = mo.range(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[5], ui32>
  // CHECK-DAG: %[[VAL1:.*]] = mo.constant {value = #M.dense_array<2, 1, 0, -1> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %startNeg = mo.constant {value = #M.dense_array<2> : tensor<si64>} : !mo.tensor<[], si64>
  %limitNeg = mo.constant {value = #M.dense_array<-2> : tensor<si64>} : !mo.tensor<[], si64>
  %stepNeg = mo.constant {value = #M.dense_array<-1> : tensor<si64>} : !mo.tensor<[], si64>
  %1 = mo.range(%startNeg, %limitNeg, %stepNeg) : (
    !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[4], si64>
  // CHECK-NEXT: %[[VAL0]], %[[VAL1]] : !mo.tensor<[5], ui32>, !mo.tensor<[4], si64>
  mo.output %0, %1 : !mo.tensor<[5], ui32>, !mo.tensor<[4], si64>
}

// CHECK: mo.graph @range_folder_unknown_output() -> !mo.tensor<[?], ui32> {
mo.graph @range_folder_unknown_output() -> !mo.tensor<[?], ui32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1, 3, 5, 7, 9> : tensor<5xui32>} : !mo.tensor<[5], ui32>
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[VAL0]]) : !mo.tensor<[5], ui32> -> !mo.tensor<[?], ui32>
  %start = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
  %limit = mo.constant {value = #M.dense_array<11> : tensor<ui32>} : !mo.tensor<[], ui32>
  %step = mo.constant {value = #M.dense_array<2> : tensor<ui32>} : !mo.tensor<[], ui32>
  %0 = mo.range(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[?], ui32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[?], ui32>
  mo.output %0 : !mo.tensor<[?], ui32>
}

// CHECK: mo.graph @range_folder_parametric_output<() -> [[D0:.*]]>() -> !mo.tensor<[[[D0]]], ui32> {
mo.graph @range_folder_parametric_output<() -> D0>() -> !mo.tensor<[D0], ui32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.constant {value = #M.dense_array<1, 3, 5, 7, 9> : tensor<5xui32>} : !mo.tensor<[5], ui32>
  // CHECK-DAG: kgen.param.declare [[D1:.*]] = <5>
  // CHECK-DAG: %[[RES:.*]] = mo.rebind(%[[VAL0]]) : !mo.tensor<[5], ui32> -> !mo.tensor<[[[D1]]], ui32>
  %start = mo.constant {value = #M.dense_array<1> : tensor<ui32>} : !mo.tensor<[], ui32>
  %limit = mo.constant {value = #M.dense_array<11> : tensor<ui32>} : !mo.tensor<[], ui32>
  %step = mo.constant {value = #M.dense_array<2> : tensor<ui32>} : !mo.tensor<[], ui32>
  %0 = mo.range<() -> D1>(%start, %limit, %step) : (
    !mo.tensor<[], ui32>, !mo.tensor<[], ui32>, !mo.tensor<[], ui32>) -> !mo.tensor<[D1], ui32>
  // CHECK-NEXT: mo.output<[[D1]]> %[[RES]] : !mo.tensor<[[[D1]]], ui32>
  mo.output<D1> %0 : !mo.tensor<[D1], ui32>
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @reshape_static_shape(%[[ARG:.*]]: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32>
mo.graph @reshape_static_shape(%arg: !mo.tensor<[4], f32>) -> !mo.tensor<[2, 2], f32> {
  // CHECK: %[[SHAPE:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %shape = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.reshape(%[[ARG]], %[[SHAPE]]) : (!mo.tensor<[4], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[4], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @reshape_dyn_shape(%[[ARG:.*]]: !mo.tensor<[?], f32>) -> !mo.tensor<[2, 2], f32>
mo.graph @reshape_dyn_shape(%arg: !mo.tensor<[?], f32>) -> !mo.tensor<[2, 2], f32> {
  // CHECK: %[[SHAPE:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %shape = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.reshape(%[[ARG]], %[[SHAPE]]) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %res = mo.reshape(%arg, %shape) : (!mo.tensor<[?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

// CHECK: mo.graph @reshape_fold_rank_1_to_scalar() -> !mo.tensor<[], si32>
mo.graph @reshape_fold_rank_1_to_scalar() -> !mo.tensor<[], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<42> : tensor<si32>} : !mo.tensor<[], si32>
  %input = mo.constant {value = #M.dense_array<42> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %shape = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[1], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[], si32>
  mo.output %res : !mo.tensor<[], si32>
}

// CHECK: mo.graph @reshape_fold_scalar_to_rank_1() -> !mo.tensor<[1], si32>
mo.graph @reshape_fold_scalar_to_rank_1() -> !mo.tensor<[1], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<42> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %input = mo.constant {value = #M.dense_array<42> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %shape = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[1], si64>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[1], si32>
  mo.output %res : !mo.tensor<[1], si32>
}

// CHECK: mo.graph @reshape_fold_rank_1_to_scalar_dyn_target_shape(%[[SHAPE:.*]]: !mo.tensor<[1], si64>) -> !mo.tensor<[], si32>
mo.graph @reshape_fold_rank_1_to_scalar_dyn_target_shape(%shape: !mo.tensor<[1], si64>) -> !mo.tensor<[], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<42> : tensor<si32>} : !mo.tensor<[], si32>
  %input = mo.constant {value = #M.dense_array<42> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[1], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[], si32>
  mo.output %res : !mo.tensor<[], si32>
}

// CHECK: mo.graph @reshape_fold_scalar_to_rank_1_dyn_target_shape<() -> [[D0:.*]]>(
mo.graph @reshape_fold_scalar_to_rank_1_dyn_target_shape<() -> D0>(
  // CHECK-SAME: %[[SHAPE:.*]]: !mo.tensor<[1], si64>
  %shape: !mo.tensor<[1], si64>
// CHECK-SAME: ) -> !mo.tensor<[D0], si32> {
) -> !mo.tensor<[D0], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<42> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %input = mo.constant {value = #M.dense_array<42> : tensor<1xsi32>} : !mo.tensor<[], si32>
  // CHECK-NEXT: %[[RES:.*]] = mo.rebind(%[[INPUT]]) : !mo.tensor<[1], si32> -> !mo.tensor<[[[D1]]], si32>
  %res = mo.reshape<() -> D1>(%input, %shape) : (!mo.tensor<[], si32>, !mo.tensor<[1], si64>) -> !mo.tensor<[D1], si32>
  // CHECK-NEXT: mo.output<[[D1]]> %[[RES]] : !mo.tensor<[[[D1]]], si32>
  mo.output<D1> %res : !mo.tensor<[D1], si32>
}

// CHECK: mo.graph @reshape_fold_multi_dims() -> !mo.tensor<[2, 2], si32>
mo.graph @reshape_fold_multi_dims() -> !mo.tensor<[2, 2], si32> {
  // CHECK-NEXT: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>} : !mo.tensor<[2, 2], si32>
  %input = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %shape = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[4], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], si32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[2, 2], si32>
  mo.output %res : !mo.tensor<[2, 2], si32>
}

// CHECK: mo.graph @reshape_no_fold_multi_dims() -> !mo.tensor<[2, 2], si32>
mo.graph @reshape_no_fold_multi_dims() -> !mo.tensor<[2, 2], si32> {
  // CHECK: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  // CHECK: %[[SHAPE:.*]] = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %input = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %shape = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[RES:.*]] = mo.reshape(%[[INPUT]], %[[SHAPE]]) : (!mo.tensor<[4], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], si32>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[4], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], si32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, 2], si32>
  mo.output %res : !mo.tensor<[2, 2], si32>
}

// CHECK: mo.graph @reshape_no_fold_output_neg_one_dim() -> !mo.tensor<[2, -1], si32>
mo.graph @reshape_no_fold_output_neg_one_dim() -> !mo.tensor<[2, -1], si32> {
  // CHECK: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  // CHECK: %[[SHAPE:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[-1], si64>
  %input = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %shape = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[-1], si64>
  // CHECK: %[[RES:.*]] = mo.reshape(%[[INPUT]], %[[SHAPE]]) : (!mo.tensor<[4], si32>, !mo.tensor<[-1], si64>) -> !mo.tensor<[2, -1], si32>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[4], si32>, !mo.tensor<[-1], si64>) -> !mo.tensor<[2, -1], si32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, -1], si32>
  mo.output %res : !mo.tensor<[2, -1], si32>
}

// CHECK: mo.graph @reshape_no_fold_target_shape_neg_one_dim() -> !mo.tensor<[2, -1], si32>
mo.graph @reshape_no_fold_target_shape_neg_one_dim() -> !mo.tensor<[2, -1], si32> {
  // CHECK: %[[INPUT:.*]] = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  // CHECK: %[[SHAPE:.*]] = mo.constant {value = #M.dense_array<2, -1> : tensor<2xsi64>} : !mo.tensor<[-1], si64>
  %input = mo.constant {value = #M.dense_array<1, 2, 3, 4> : tensor<4xsi32>} : !mo.tensor<[4], si32>
  %shape = mo.constant {value = #M.dense_array<2, -1> : tensor<2xsi64>} : !mo.tensor<[-1], si64>
  // CHECK: %[[RES:.*]] = mo.reshape(%[[INPUT]], %[[SHAPE]]) : (!mo.tensor<[4], si32>, !mo.tensor<[-1], si64>) -> !mo.tensor<[2, -1], si32>
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[4], si32>, !mo.tensor<[-1], si64>) -> !mo.tensor<[2, -1], si32>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2, -1], si32>
  mo.output %res : !mo.tensor<[2, -1], si32>
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @shape_of_static_rank(%[[ARG:.*]]: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
mo.graph @shape_of_static_rank(%arg: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64> {
  // CHECK: %[[RES:.*]] = mo.shape_of(%[[ARG]]) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[2], si64>
  mo.output %res : !mo.tensor<[2], si64>
}

// CHECK: mo.graph @shape_of_dyn_rank(%[[ARG:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
mo.graph @shape_of_dyn_rank(%arg: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64> {
  // CHECK: %[[RES:.*]] = mo.shape_of(%[[ARG]]) : (!mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
  %res = mo.shape_of(%arg) : (!mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
  // CHECK: mo.output %[[RES]] : !mo.tensor<[?], si64>
  mo.output %res : !mo.tensor<[?], si64>
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @slice_static_rank(%[[INPUT:.*]]: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[8, 5], f32> {
mo.graph @slice_static_rank(%input: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[8, 5], f32> {
  // CHECK-NEXT: %[[BEGIN:.*]] = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[END:.*]] = mo.constant {value = #M.dense_array<-1, 10> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %stop = mo.constant {value = #M.dense_array<-1, 10> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  // CHECK-NEXT: %[[STRIDES:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.slice(%[[INPUT]], %[[BEGIN]], %[[END]], %[[STRIDES]]) : (
  %res = mo.slice(%input, %start, %stop, %step) : (
    // CHECK-SAME: !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
    !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<[8, 5], f32>
  ) -> !mo.tensor<[8, 5], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// CHECK: mo.graph @slice_dyn_rank(%[[INPUT:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
mo.graph @slice_dyn_rank(%input: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
  // CHECK-NEXT: %[[BEGIN:.*]] = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %start = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[END:.*]] = mo.constant {value = #M.dense_array<-1, 10> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %stop = mo.constant {value = #M.dense_array<-1, 10> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  // CHECK-NEXT: %[[STRIDES:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.slice(%[[INPUT]], %[[BEGIN]], %[[END]], %[[STRIDES]]) : (
  %res = mo.slice(%input, %start, %stop, %step) : (
    // CHECK-SAME: !mo.tensor<[?, ?], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
    !mo.tensor<[?, ?], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?], f32>
  ) -> !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @transpose(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
mo.graph @transpose(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  // CHECK: %[[PERM:.*]] = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: mo.transpose(%[[INPUT]], %[[PERM]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// CHECK: mo.graph @transpose_identity_fold_no_decl(%[[INPUT:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
mo.graph @transpose_identity_fold_no_decl(%input: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
  %perm = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: mo.output %[[INPUT]] : !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// CHECK: mo.graph @transpose_identity_fold_with_decl<[[D0:.*]] -> [[D1:.*]], [[D2:.*]]>(
mo.graph @transpose_identity_fold_with_decl<D0 -> D1, D2>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 42], f32>
  %input: !mo.tensor<[D0, 42], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D1]], [[D2]]], f32> {
) -> !mo.tensor<[D1, D2], f32> {
  %perm = mo.constant {value = #M.dense_array<0, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: kgen.param.declare [[D3:.*]] = <[[D0]]>
  // CHECK-DAG: kgen.param.declare [[D4:.*]] = <42>
  // CHECK-DAG: %[[RES:.*]] = mo.rebind(%[[INPUT]]) : !mo.tensor<[[[D0]], 42], f32> -> !mo.tensor<[[[D3]], [[D4]]], f32>
  %res = mo.transpose<() -> D3, D4>(%input, %perm) : (!mo.tensor<[D0, 42], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D3, D4], f32>
  // CHECK-NEXT: mo.output<[[D3]], [[D4]]> %[[RES]] : !mo.tensor<[[[D3]], [[D4]]], f32>
  mo.output<D3, D4> %res : !mo.tensor<[D3, D4], f32>
}

// CHECK: mo.graph @transpose_consecutive_fold<[[D0:.*]] -> [[D1:.*]], [[D2:.*]], [[D3:.*]]>(
mo.graph @transpose_consecutive_fold<D0 -> D1, D2, D3>(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], 10, 20], f32>
  %input: !mo.tensor<[D0, 10, 20], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D1]], [[D2]], [[D3]]], f32> {
) -> !mo.tensor<[D1, D2, D3], f32> {
  // CHECK-NEXT: %[[NEW_PERM:.*]] = mo.constant {value = #M.dense_array<2, 1, 0> : !M.array<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[PERM1:.*]] = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %perm1 = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %perm2 = mo.constant {value = #M.dense_array<1, 2, 0> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[RES1:.*]] = mo.transpose<() -> [[D4:.*]], [[D5:.*]], [[D6:.*]]>(%[[INPUT]], %[[PERM1]])
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 10, 20], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[D4]], [[D5]], [[D6]]], f32>
  %res1 = mo.transpose<() -> D4, D5, D6>(%input, %perm1) : (!mo.tensor<[D0, 10, 20], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D4, D5, D6], f32>
  // CHECK-NEXT: %[[RES2:.*]] = mo.transpose<() -> [[D7:.*]], [[D8:.*]], [[D9:.*]]>(%[[INPUT]], %[[NEW_PERM]])
  // CHECK-SAME: : (!mo.tensor<[[[D0]], 10, 20], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
  %res2 = mo.transpose<() -> D7, D8, D9>(%res1, %perm2) : (!mo.tensor<[D4, D5, D6], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D7, D8, D9], f32>
  // CHECK-NEXT: mo.output<[[D7]], [[D8]], [[D9]]> %[[RES2]] : !mo.tensor<[[[D7]], [[D8]], [[D9]]], f32>
  mo.output<D7, D8, D9> %res2 : !mo.tensor<[D7, D8, D9], f32>
}

// CHECK: mo.graph @transpose_non_const_perm_no_fold(
mo.graph @transpose_non_const_perm_no_fold(
  // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[2, 2], f32>,
  %input: !mo.tensor<[2, 2], f32>,
  // CHECK-SAME: %[[PERM:.*]]: !mo.tensor<[2], si64>
  %perm: !mo.tensor<[2], si64>
// CHECK-SAME: ) -> !mo.tensor<[2, 2], f32> {
) -> !mo.tensor<[2, 2], f32> {
  // CHECK-NEXT: %[[RES:.*]] = mo.transpose(%[[INPUT]], %[[PERM]])
  // CHECK-SAME: : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 2], f32>
  mo.output %res : !mo.tensor<[2, 2], f32>
}

//===----------------------------------------------------------------------===//
// WhereOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @where_1d(%[[ARG:.*]]: !mo.tensor<[7], f32>) -> !mo.tensor<[?, 1], si64>
mo.graph @where_1d(%arg: !mo.tensor<[7], f32>) -> !mo.tensor<[?, 1], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.where(%[[ARG]]) : (!mo.tensor<[7], f32>) -> !mo.tensor<[?, 1], si64>
  %res = mo.where(%arg) : (!mo.tensor<[7], f32>) -> !mo.tensor<[?, 1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[?, 1], si64>
  mo.output %res : !mo.tensor<[?, 1], si64>
}

// CHECK: mo.graph @where_2d(%[[ARG:.*]]: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 2], si64>
mo.graph @where_2d(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 2], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.where(%[[ARG]]) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 2], si64>
  %res = mo.where(%arg) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 2], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[?, 2], si64>
  mo.output %res : !mo.tensor<[?, 2], si64>
}

// CHECK: mo.graph @where_2d_dynamic_shape(%[[ARG:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, 2], si64>
mo.graph @where_2d_dynamic_shape(%arg: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, 2], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.where(%[[ARG]]) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, 2], si64>
  %res = mo.where(%arg) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, 2], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[?, 2], si64>
  mo.output %res : !mo.tensor<[?, 2], si64>
}

// CHECK: mo.graph @where_dynamic_rank(%[[ARG:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<[?, ?], si64>
mo.graph @where_dynamic_rank(%arg: !mo.tensor<?, f32>) -> !mo.tensor<[?, ?], si64> {
  // CHECK-NEXT: %[[RES:.*]] = mo.where(%[[ARG]]) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, ?], si64>
  %res = mo.where(%arg) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, ?], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[?, ?], si64>
  mo.output %res : !mo.tensor<[?, ?], si64>
}

// CHECK: mo.graph @where_1d_fold() -> !mo.tensor<[?, 1], si64>
mo.graph @where_1d_fold() -> !mo.tensor<[?, 1], si64> {
  %input = mo.constant {value = #M.dense_array<1.0, 0.0, 1.0> : tensor<3xf32>} : !mo.tensor<[3], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.constant {value = #M.dense_array<0, 2> : tensor<2x1xsi64>} : !mo.tensor<[2, 1], si64>
  // CHECK-NEXT: %[[REB:.*]] = mo.rebind(%[[RES]]) : !mo.tensor<[2, 1], si64> -> !mo.tensor<[?, 1], si64>
  %res = mo.where(%input) : (!mo.tensor<[3], f32>) -> !mo.tensor<[?, 1], si64>
  // CHECK-NEXT: mo.output %[[REB]] : !mo.tensor<[?, 1], si64>
  mo.output %res : !mo.tensor<[?, 1], si64>
}

// CHECK: mo.graph @where_1d_parameterized_fold<() -> [[D0:.*]]>() -> !mo.tensor<[[[D0]], 1], si64>
mo.graph @where_1d_parameterized_fold<() -> D0>() -> !mo.tensor<[D0, 1], si64> {
  %input = mo.constant {value = #M.dense_array<1.0, 0.0, 1.0> : tensor<3xf32>} : !mo.tensor<[3], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.constant {value = #M.dense_array<0, 2> : tensor<2x1xsi64>} : !mo.tensor<[2, 1], si64>
  // CHECK-NEXT: kgen.param.declare [[D1:.*]] = <2>
  // CHECK-NEXT: %[[REB:.*]] = mo.rebind(%[[RES]]) : !mo.tensor<[2, 1], si64> -> !mo.tensor<[D1, 1], si64>
  %res = mo.where<() -> D1>(%input) : (!mo.tensor<[3], f32>) -> !mo.tensor<[D1, 1], si64>
  // CHECK-NEXT: mo.output<[[D1]]> %[[REB]] : !mo.tensor<[[[D1]], 1], si64>
  mo.output<D1> %res : !mo.tensor<[D1, 1], si64>
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @arg_max_2d_inner_axis(%[[ARG:.*]]: !mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64>
mo.graph @arg_max_2d_inner_axis(%arg: !mo.tensor<[7, 3], f32>) -> !mo.tensor<[7, 1], si64> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant
  %axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_max(%[[ARG]], %[[AXIS]]) : (!mo.tensor<[7, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[7, 1], si64>
  %res = mo.arg_max(%arg, %axis) : (!mo.tensor<[7, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[7, 1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[7, 1], si64>
  mo.output %res : !mo.tensor<[7, 1], si64>
}

// CHECK: mo.graph @arg_max_2d_inner_axis_dynamic_shape(%[[ARG:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, 1], si64>
mo.graph @arg_max_2d_inner_axis_dynamic_shape(%arg: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, 1], si64> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant
  %axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_max(%[[ARG]], %[[AXIS]]) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 1], si64>
  %res = mo.arg_max(%arg, %axis) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, 1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[?, 1], si64>
  mo.output %res : !mo.tensor<[?, 1], si64>
}

// CHECK: mo.graph @arg_max_2d_inner_axis_dynamic_reduced_dim(%[[ARG:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], si64>
mo.graph @arg_max_2d_inner_axis_dynamic_reduced_dim(%arg: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], si64> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant
  %axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_max(%[[ARG]], %[[AXIS]]) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], si64>
  %res = mo.arg_max(%arg, %axis) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[?, ?], si64>
  mo.output %res : !mo.tensor<[?, ?], si64>
}

// CHECK: mo.graph @arg_max_2d_inner_axis_dynamic_rank(%[[ARG:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, si64>
mo.graph @arg_max_2d_inner_axis_dynamic_rank(%arg: !mo.tensor<?, f32>) -> !mo.tensor<?, si64> {
  // CHECK-NEXT: %[[AXIS:.*]] = mo.constant
  %axis = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.arg_max(%[[ARG]], %[[AXIS]]) : (!mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, si64>
  %res = mo.arg_max(%arg, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<?, si64>
  mo.output %res : !mo.tensor<?, si64>
}
