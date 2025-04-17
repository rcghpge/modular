// REQUIRES: disabled
// TODO(GEX-1573): Delete or convert to the new API
// RUN: mo-opt --mo-to-mogg --fuse-affine-views --canonicalize --fuse-in-out-lambdas --canonicalize %s | FileCheck %s

// CHECK-LABEL: mo.graph @inout_fusion<seq_len>
// CHECK-NEXT: mogg.kernel

// CHECK: %[[D1:.*]] = kgen.param.constant: scalar<si64> = <4096>
// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], f32>
// CHECK: %[[DIM:.*]] = mogg.get_dim <seq_len> -> !pop.scalar<si64>
// CHECK: %[[ARR:.*]] = pop.array.create [%[[DIM]], %[[D1]]] : !pop.array<2, scalar<si64>>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg"(%[[ARR]]) : (!pop.array<2, scalar<si64>>) -> !mogg.shape<[seq_len, 4096]>

// CHECK-NEXT: %[[ORIG_TENSOR:.*]] = mogg.call "reshape_contiguous_buffer"(%[[OUT_PH]], %[[SHAPE]])

// CHECK: mogg.lambda @{{.*}} = (%[[INDICES:.*]]: !mogg.indices
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "insert_index"(%[[INDICES]]) {axis = 0 : index, rank = 2 : index, value = 0 : index} : !mogg.indices -> !mogg.indices

// CHECK-NEXT: %[[LD:.*]] = mogg.call "simd_load"(%{{.*}}, %[[NEW_IND]]) : !mo.tensor<[1, seq_len, 4096], f32>, !mogg.indices
// CHECK-NEXT: %[[VAL:.*]] = mogg.call "mo.add"
// CHECK-NEXT: mogg.call "simd_store"(%[[OUT_PH]], %[[NEW_IND]], %[[VAL]]

// CHECK: mogg.call "mo.matmul" [@{{.*}}](%{{.*}}, %[[ORIG_TENSOR]])
// CHECK-NEXT: mogg.output %[[OUT_PH]] : !mo.tensor<[1, seq_len, 4096], f32>

mo.graph @inout_fusion<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[1, seq_len, 4096], f32>) ->  !mo.tensor<[1, seq_len, 4096], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %1 = mo.add_singleton_dim[0 : index] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
  %out = mo.add(%1, %arg2) : !mo.tensor<[1, seq_len, 4096], f32>
  mo.output %out : !mo.tensor<[1, seq_len, 4096], f32>
}

// CHECK-LABEL: mo.graph @mixed_precision<seq_len>
// CHECK-NEXT: mogg.kernel
// CHECK: %[[D1:.*]] = kgen.param.constant: scalar<si64> = <4096>
// CHECK: %[[OUT_PH:.*]]:2 = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], si32>, !mo.tensor<[1, seq_len, 4096], f32>

// CHECK: %[[DIM:.*]] = mogg.get_dim <seq_len> -> !pop.scalar<si64>
// CHECK: %[[ARR:.*]] = pop.array.create [%[[DIM]], %[[D1]]] : !pop.array<2, scalar<si64>>
// CHECK: %[[SHAPE:.*]] = "mogg.to_mogg"(%[[ARR]]) : (!pop.array<2, scalar<si64>>) -> !mogg.shape<[seq_len, 4096]>

// CHECK-NEXT: %[[ORIG_TENSOR:.*]] = mogg.call "reshape_contiguous_buffer"(%[[OUT_PH]]#1, %[[SHAPE]])

// CHECK: mogg.lambda @{{.*}} = (%[[INDICES:.*]]: !mogg.indices
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "insert_index"(%[[INDICES]]) {axis = 0 : index, rank = 2 : index, value = 0 : index} : !mogg.indices -> !mogg.indices

// CHECK-NEXT: %[[VAL:.*]] = mogg.call "mo.cast"
// CHECK-NEXT: mogg.call "simd_store"(%[[OUT_PH]]#0, %[[NEW_IND]], %[[VAL]]

// CHECK: mogg.call "mo.matmul" [@{{.*}}](%{{.*}}, %[[ORIG_TENSOR]])
// CHECK-NEXT: mogg.output %[[OUT_PH]]#0, %[[OUT_PH]]#1 : !mo.tensor<[1, seq_len, 4096], si32>, !mo.tensor<[1, seq_len, 4096], f32>

mo.graph @mixed_precision<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) ->  !mo.tensor<[1, seq_len, 4096], si32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %1 = mo.add_singleton_dim[0 : index] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
  %out = mo.cast(%1) : (!mo.tensor<[1, seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], si32>
  mo.output %out : !mo.tensor<[1, seq_len, 4096], si32>
}
