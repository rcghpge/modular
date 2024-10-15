// RUN: mo-opt --mo-to-mogg --fuse-affine-views --canonicalize %s | FileCheck %s

// CHECK-LABEL: mo.graph @add_singleton_non_elementwise<seq_len>
// CHECK-NEXT: mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], f32>
// CHECK-NEXT: %[[SHAPE:.*]] = mogg.param_shape_to_value -> !mogg.shape<[seq_len, 4096]>
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
