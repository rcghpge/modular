// RUN: mo-opt --mo-to-mogg --fuse-affine-views --canonicalize %s | FileCheck %s

// CHECK-LABEL: mo.graph @add_singleton_non_elementwise<seq_len>
// CHECK-NEXT: mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], f32>
// CHECK: %[[BIND:.*]] = mogg.bind (%[[OUT_PH]]) output_lambda (%{{.*}}: f32, %[[INDICES:.*]]: !pop.array<2, index>) {
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "insert_index"(%[[INDICES]]) {axis = 0 : index, value = 0 : index} : !pop.array<2, index> -> !pop.array<3, index>
// CHECK-NEXT: mogg.output %{{.*}}, %[[NEW_IND]] : f32, !pop.array<3, index>

// CHECK: mogg.call.execute["mo.matmul"] inputs(%{{.*}}, %{{.*}} : {{.*}}) outputs(%[[BIND]] : {{.*}}) {isThrows, kernel_param = {packed_b = false, transpose_b = false}}
// CHECK: summary = "mo.matmul:{{.*}}+mo.add_singleton_dim:{{.*}}"

mo.graph @add_singleton_non_elementwise<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) ->  !mo.tensor<[1, seq_len, 4096], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %1 = mo.add_singleton_dim[0] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
  mo.output %1 : !mo.tensor<[1, seq_len, 4096], f32>
}

// CHECK-LABEL: mo.graph @fuse_kernel_in_true_block_of_if_statement<seq_len>
// CHECK-NEXT: [[RES:.*]] = mo.if
// CHECK-NEXT: [[KER_RES:.*]] = mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], f32>
// CHECK: %[[BIND:.*]] = mogg.bind (%[[OUT_PH]]) output_lambda (%{{.*}}: f32, %[[INDICES:.*]]: !pop.array<2, index>) {
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "insert_index"(%[[INDICES]]) {axis = 0 : index, value = 0 : index} : !pop.array<2, index> -> !pop.array<3, index>
// CHECK-NEXT: mogg.output %{{.*}}, %[[NEW_IND]] : f32, !pop.array<3, index>

// CHECK: mogg.call.execute["mo.matmul"] inputs(%{{.*}}, %{{.*}} : {{.*}}) outputs(%[[BIND]] : {{.*}}) {isThrows, kernel_param = {packed_b = false, transpose_b = false}}
// CHECK: summary = "mo.matmul:{{.*}}+mo.add_singleton_dim:{{.*}}"

// CHECK: mo.yield [[KER_RES:.*]]
// CHECK-NEXT: } else {
// CHECK: mo.output [[RES:.*]]

mo.graph @fuse_kernel_in_true_block_of_if_statement<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>, %arg2 : !mo.tensor<[1, seq_len, 4096], f32>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
    %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
    %1 = mo.add_singleton_dim[0] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
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
// CHECK: %[[BIND:.*]] = mogg.bind (%[[OUT_PH]]) output_lambda (%{{.*}}: f32, %[[INDICES:.*]]: !pop.array<2, index>) {
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "insert_index"(%[[INDICES]]) {axis = 0 : index, value = 0 : index} : !pop.array<2, index> -> !pop.array<3, index>
// CHECK-NEXT: mogg.output %{{.*}}, %[[NEW_IND]] : f32, !pop.array<3, index>

// CHECK: mogg.call.execute["mo.matmul"] inputs(%{{.*}}, %{{.*}} : {{.*}}) outputs(%[[BIND]] : {{.*}}) {isThrows, kernel_param = {packed_b = false, transpose_b = false}}
// CHECK: summary = "mo.matmul:{{.*}}+mo.add_singleton_dim:{{.*}}"

// CHECK: mo.yield [[KER_RES:.*]]
// CHECK: mo.output [[RES:.*]]

mo.graph @fuse_kernel_in_false_block_of_if_statement<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>, %arg2 : !mo.tensor<[1, seq_len, 4096], f32>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
    mo.yield %arg2 : !mo.tensor<[1, seq_len, 4096], f32>
  } else {
    %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
    %1 = mo.add_singleton_dim[0] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
    mo.yield %1 : !mo.tensor<[1, seq_len, 4096], f32>
  }
  mo.output %res : !mo.tensor<[1, seq_len, 4096], f32>
}

// CHECK-LABEL: mo.graph @dont_fuse_kernel_across_blocks<seq_len>
// CHECK-NEXT: mogg.kernel
// CHECK: summary = "mo.matmul:{{.*}}"
// CHECK: mo.if
// CHECK-NEXT: mo.add_singleton_dim
// CHECK: } else {

mo.graph @dont_fuse_kernel_across_blocks<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>, %arg2 : !mo.tensor<[1, seq_len, 4096], f32>, %cond: !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<[1, seq_len, 4096], f32> {
    %1 = mo.add_singleton_dim[0] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
    mo.yield %1 : !mo.tensor<[1, seq_len, 4096], f32>
  } else {
    mo.yield %arg2 : !mo.tensor<[1, seq_len, 4096], f32>
  }
  mo.output %res : !mo.tensor<[1, seq_len, 4096], f32>
}

// CHECK-LABEL: mo.graph @split_dim<seq_len, N, K>
// CHECK-NEXT: mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[N, K, 4096], f32>
// CHECK-NEXT: %[[NEW_DIM:.*]] = mogg.get_dim <K> -> !pop.scalar<si64>
// CHECK: %[[BIND:.*]] = mogg.bind (%[[OUT_PH]]) output_lambda (%{{.*}}: f32, %[[INDICES:.*]]: !pop.array<2, index>) {
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "split_dim_indices"(%[[INDICES]], %[[NEW_DIM]]) {axis = 0 : index} : !pop.array<2, index>, !pop.scalar<si64> -> !pop.array<3, index>
// CHECK-NEXT: mogg.output %{{.*}}, %[[NEW_IND]] : f32, !pop.array<3, index>

// CHECK: mogg.call.execute["mo.matmul"] inputs(%{{.*}}, %{{.*}} : {{.*}}) outputs(%[[BIND]] : {{.*}}) {isThrows, kernel_param = {packed_b = false, transpose_b = false}}
// CHECK: summary = "mo.matmul:{{.*}}+mo.split_dim:{{.*}}"


mo.graph @split_dim<seq_len, N, K>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) ->  !mo.tensor<[N, K, 4096], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %1 = mo.split_dim[0] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[N, K, 4096], f32>
  mo.output %1 : !mo.tensor<[N, K, 4096], f32>
}


// CHECK-LABEL: mo.graph @gather_custom
// CHECK: mogg.kernel
// CHECK: mogg.bind (%{{.*}}) input_lambda
// CHECK: mogg.bind (%{{.*}}) output_lambda
// CHECK: mogg.call.execute["mo.gather"]
// CHECK: summary = "mo.gather:{{.*}}"
// CHECK: mogg.kernel
// CHECK: mogg.bind (%{{.*}}) input_lambda
// CHECK: mogg.bind (%{{.*}}) output_lambda
// CHECK: mogg.call.execute["rms_norm"]
// CHECK: summary = "custom__rms_norm:{{.*}}"

mo.graph @gather_custom(%arg0: !mo.tensor<[], f32>, %arg1: !mo.tensor<[4], f32>, %arg2: !mo.tensor<[4, 4], f32>, %arg3: !mo.tensor<[4, 4], si32>, %arg4: !mo.tensor<[4, 4, 4], f32>) -> !mo.tensor<[4, 4, 4], f32> {
  %0 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %weight_offset = mo.constant {value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
  %1 = mo.gather(%arg2, %arg3, %0) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[4, 4, 4], f32>
  %2 = mo.custom {symbol = "rms_norm", device = #M.device_ref<"cpu", 0>}(%1, %arg1, %arg0, %weight_offset) : (!mo.tensor<[4, 4, 4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[4, 4, 4], f32>
  mo.output %2 : !mo.tensor<[4, 4, 4], f32>
}


// CHECK-LABEL: mo.graph @merge_dim
// CHECK-NEXT: mogg.kernel

// CHECK: %[[OUT_PH:.*]] = mogg.output_placeholders : !mo.tensor<[64], f32>
// CHECK: %[[SIMD:.*]] = mogg.constant.scalar {value = 8 : si64} : !mo.scalar<si64>
// CHECK: %[[BIND:.*]] = mogg.bind (%[[OUT_PH]]) output_lambda (%{{.*}}: f32, %[[INDICES:.*]]: !pop.array<2, index>) {
// CHECK-NEXT: %[[NEW_IND:.*]] = mogg.call "merge_dim_indices"(%[[INDICES]], %[[SIMD]]) {axis = 0 : index} : !pop.array<2, index>, !mo.scalar<si64> -> !pop.array<1, index>
// CHECK-NEXT: mogg.output %{{.*}}, %[[NEW_IND]] : f32, !pop.array<1, index>

// CHECK: mogg.call.execute["mo.matmul"] inputs(%{{.*}}, %{{.*}} : {{.*}}) outputs(%[[BIND]] : {{.*}}) {isThrows, kernel_param = {packed_b = false, transpose_b = false}}
// CHECK: summary = "mo.matmul:{{.*}}+mo.merge_dim:{{.*}}"
mo.graph @merge_dim(%arg0: !mo.tensor<[8, 8], f32>, %arg1: !mo.tensor<[8, 8], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[64], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[8, 8], f32>, !mo.tensor<[8, 8], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[8, 8], f32>
  %1 = mo.merge_dim[0] (%0) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[64], f32>
  mo.output %1: !mo.tensor<[64], f32>
}
