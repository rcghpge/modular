// REQUIRES: disabled
// TODO(GEX-1573): Delete or convert to the new API
// RUN: mo-opt --mo-to-mogg --fuse-affine-views --canonicalize %s | FileCheck %s

// CHECK-LABEL: mo.graph @add_singleton_non_elementwise<seq_len>
// CHECK-NEXT: mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], f32>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg
// CHECK-NEXT: %[[ORIG_TENSOR:.*]] = mogg.call "reshape_contiguous_buffer"(%[[OUT_PH]], %[[SHAPE]])

// CHECK: mogg.lambda @{{.*}} = (%[[INDICES:.*]]: !mogg.indices
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "insert_index"(%[[INDICES]]) {axis = 0 : index, rank = 2 : index, value = 0 : index} : !mogg.indices -> !mogg.indices
// CHECK-NEXT: mogg.call "simd_store"(%[[OUT_PH]], %[[NEW_IND]]

// CHECK: mogg.call "mo.matmul" [@{{.*}}](%{{.*}}, %[[ORIG_TENSOR]])
// CHECK-NEXT: mogg.output %[[OUT_PH]] : !mo.tensor<[1, seq_len, 4096], f32>

mo.graph @add_singleton_non_elementwise<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) ->  !mo.tensor<[1, seq_len, 4096], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %1 = mo.add_singleton_dim[0 : index] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
  mo.output %1 : !mo.tensor<[1, seq_len, 4096], f32>
}

// CHECK-LABEL: mo.graph @fuse_kernel_in_true_block_of_if_statement<seq_len>
// CHECK-NEXT: [[RES:.*]] = mo.if
// CHECK-NEXT: [[KER_RES:.*]] = mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], f32>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg
// CHECK-NEXT: %[[ORIG_TENSOR:.*]] = mogg.call "reshape_contiguous_buffer"(%[[OUT_PH]], %[[SHAPE]])

// CHECK: mogg.lambda @{{.*}} = (%[[INDICES:.*]]: !mogg.indices
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "insert_index"(%[[INDICES]]) {axis = 0 : index, rank = 2 : index, value = 0 : index} : !mogg.indices -> !mogg.indices
// CHECK-NEXT: mogg.call "simd_store"(%[[OUT_PH]], %[[NEW_IND]]

// CHECK: mogg.call "mo.matmul" [@{{.*}}](%{{.*}}, %[[ORIG_TENSOR]])
// CHECK-NEXT: mogg.output %[[OUT_PH]] : !mo.tensor<[1, seq_len, 4096], f32>

// CHECK: mo.yield [[KER_RES:.*]]
// CHECK-NEXT: } else {
// CHECK: mo.output [[RES:.*]]

mo.graph @fuse_kernel_in_true_block_of_if_statement<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>, %arg2 : !mo.tensor<[1, seq_len, 4096], f32>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
    %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
    %1 = mo.add_singleton_dim[0 : index] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
    mo.yield %1 : !mo.tensor<[1, seq_len, 4096], f32>
  } else {
    mo.yield %arg2 : !mo.tensor<[1, seq_len, 4096], f32>
  }
  mo.output %res : !mo.tensor<[1, seq_len, 4096], f32>
}

// CHECK-LABEL: mo.graph @fuse_kernel_in_false_block_of_if_statement<seq_len>
// CHECK-NEXT: [[RES:.*]] = mo.if
// CHECK: } else {
// CHECK-NEXT: [[KER_RES:.*]] = mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], f32>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg
// CHECK-NEXT: %[[ORIG_TENSOR:.*]] = mogg.call "reshape_contiguous_buffer"(%[[OUT_PH]], %[[SHAPE]])

// CHECK: mogg.lambda @{{.*}} = (%[[INDICES:.*]]: !mogg.indices
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "insert_index"(%[[INDICES]]) {axis = 0 : index, rank = 2 : index, value = 0 : index} : !mogg.indices -> !mogg.indices
// CHECK-NEXT: mogg.call "simd_store"(%[[OUT_PH]], %[[NEW_IND]]

// CHECK: mogg.call "mo.matmul" [@{{.*}}](%{{.*}}, %[[ORIG_TENSOR]])
// CHECK-NEXT: mogg.output %[[OUT_PH]] : !mo.tensor<[1, seq_len, 4096], f32>

// CHECK: mo.yield [[KER_RES:.*]]
// CHECK: mo.output [[RES:.*]]

mo.graph @fuse_kernel_in_false_block_of_if_statement<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>, %arg2 : !mo.tensor<[1, seq_len, 4096], f32>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
    mo.yield %arg2 : !mo.tensor<[1, seq_len, 4096], f32>
  } else {
    %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
    %1 = mo.add_singleton_dim[0 : index] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
    mo.yield %1 : !mo.tensor<[1, seq_len, 4096], f32>
  }
  mo.output %res : !mo.tensor<[1, seq_len, 4096], f32>
}

// CHECK-LABEL: mo.graph @dont_fuse_kernel_across_blocks<seq_len>
// CHECK-NEXT: mogg.kernel
// CHECK: mo.if
// CHECK-NEXT: mo.add_singleton_dim
// CHECK: } else {

mo.graph @dont_fuse_kernel_across_blocks<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>, %arg2 : !mo.tensor<[1, seq_len, 4096], f32>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
    %1 = mo.add_singleton_dim[0 : index] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
    mo.yield %1 : !mo.tensor<[1, seq_len, 4096], f32>
  } else {
    mo.yield %arg2 : !mo.tensor<[1, seq_len, 4096], f32>
  }
  mo.output %res : !mo.tensor<[1, seq_len, 4096], f32>
}

// CHECK-LABEL: mo.graph @split_dim<seq_len, N, K>
// CHECK-NEXT: mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[N, K, 4096], f32>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg
// CHECK-NEXT: %[[ORIG_TENSOR:.*]] = mogg.call "reshape_contiguous_buffer"(%[[OUT_PH]], %[[SHAPE]])
// CHECK-NEXT: %[[NEW_DIM:.*]] = mogg.get_dim <K> -> !pop.scalar<si64>

// CHECK: mogg.lambda @{{.*}} = (%[[INDICES:.*]]: !mogg.indices
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "split_dim_indices"(%[[INDICES]], %[[NEW_DIM]]) {axis = 0 : index, rank = 2 : index} : !mogg.indices, !pop.scalar<si64> -> !mogg.indices
// CHECK-NEXT: mogg.call "simd_store"(%[[OUT_PH]], %[[NEW_IND]]

// CHECK: mogg.call "mo.matmul" [@{{.*}}](%{{.*}}, %[[ORIG_TENSOR]])
// CHECK-NEXT: mogg.output %[[OUT_PH]] : !mo.tensor<[N, K, 4096], f32>


mo.graph @split_dim<seq_len, N, K>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) ->  !mo.tensor<[N, K, 4096], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %1 = mo.split_dim[0 : index] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[N, K, 4096], f32>
  mo.output %1 : !mo.tensor<[N, K, 4096], f32>
}


// CHECK-LABEL: mo.graph @gather_custom
// CHECK:       mogg.lambda @input_0_fn_counter_{{.*}}
// CHECK-NEXT:    mogg.call "translate_gather_indices"
// CHECK-NEXT:    mogg.call "simd_load"
// CHECK:       summary = "mo.gather:{{.*}}+custom__rms_norm:{{.*}}"

mo.graph @gather_custom(%arg0: !mo.tensor<[], f32>, %arg1: !mo.tensor<[4], f32>, %arg2: !mo.tensor<[4, 4], f32>, %arg3: !mo.tensor<[4, 4], si32>, %arg4: !mo.tensor<[4, 4, 4], f32>) -> !mo.tensor<[4, 4, 4], f32> {
  %0 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %1 = mo.gather(%arg2, %arg3, %0) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[4, 4, 4], f32>
  %2 = mo.custom {symbol = "rms_norm"}(%1, %arg1, %arg0) : (!mo.tensor<[4, 4, 4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[4, 4, 4], f32>
  mo.output %2 : !mo.tensor<[4, 4, 4], f32>
}


// CHECK-LABEL: mo.graph @merge_dim
// CHECK-NEXT: mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[64], f32>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg
// CHECK-NEXT: %[[ORIG_TENSOR:.*]] = mogg.call "reshape_contiguous_buffer"(%[[OUT_PH]], %[[SHAPE]])

// CHECK: mogg.lambda @{{.*}} = (%[[INDICES:.*]]: !mogg.indices
// CHECK: %[[NEW_IND:.*]] = mogg.call "merge_dim_indices"(%[[INDICES]], %[[NEW_DIM:.*]]) {axis = 0 : index, rank = 2 : index} : !mogg.indices, !pop.scalar<si64> -> !mogg.indices
// CHECK-NEXT: mogg.call "simd_store"(%[[OUT_PH]], %[[NEW_IND]]

// CHECK: mogg.call "mo.matmul" [@{{.*}}](%{{.*}}, %[[ORIG_TENSOR]])
// CHECK-NEXT: mogg.output %[[OUT_PH]] : !mo.tensor<[64], f32>
mo.graph @merge_dim(%arg0: !mo.tensor<[8, 8], f32>, %arg1: !mo.tensor<[8, 8], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[64], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[8, 8], f32>
  %1 = mo.merge_dim[0 : index] (%0) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[64], f32>
  mo.output %1: !mo.tensor<[64], f32>
}
