// RUN: mo-opt --mo-to-mogg --mogg-fuse-affine-views --canonicalize --mogg-fuse-prologues-epilogues --canonicalize %s | FileCheck %s

// CHECK-LABEL: mo.graph @inout_fusion<seq_len>
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[seq_len, 4096], f32>, %[[ARG1:.*]]: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>, %[[ARG2:.*]]: !mo.tensor<[1, seq_len, 4096], f32>)
// CHECK-SAME: -> !mo.tensor<[1, seq_len, 4096], f32>
// CHECK-NEXT: %[[KERNEL_OUT:.*]] = mogg.kernel(%[[ARG2]]: !mo.tensor<[1, seq_len, 4096], f32>, %[[ARG0]]: !mo.tensor<[seq_len, 4096], f32>, %[[ARG1]]: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[1, seq_len, 4096], f32>
// CHECK: %[[PLACEHOLDER:.*]] = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], f32>
// CHECK: %[[BIND:.*]] = mogg.bind (%[[PLACEHOLDER]]) output_lambda (%[[ARG6:.*]]: f32, %[[ARG7:.*]]: !pop.array<2, index>)
// CHECK: %[[INSERT_IDX:.*]] = mogg.call "insert_index"(%[[ARG7]]) {axis = 0 : index, value = 0 : index} : !pop.array<2, index> -> !pop.array<3, index>
// CHECK: %[[LOAD:.*]] = "mogg.tensor.load"(%[[ARG2]], %[[INSERT_IDX]]) : (!mo.tensor<[1, seq_len, 4096], f32>, !pop.array<3, index>) -> f32
// CHECK: %[[ADD:.*]] = mogg.call.elementwise["mo.add"] (%[[ARG6]], %[[LOAD]]) : f32, f32 -> f32
// CHECK: mogg.output %[[ADD]], %[[INSERT_IDX]] : f32, !pop.array<3, index>
// CHECK: }: (!mo.tensor<[1, seq_len, 4096], f32>) -> !mo.tensor<[seq_len, 4096], f32>
// CHECK: mogg.call.execute["mo.matmul"] inputs(%[[ARG0]], %[[ARG1]], {{.*}}
// CHECK: } fusionInfo = Epilogue(Affine(mo.matmul:{{[0-9]+}}, mo.add_singleton_dim:{{[0-9]+}}), mo.add:{{[0-9]+}}, operandIndices=[0])
// CHECK-SAME: hasReadEffect = false, hasWriteEffect = false, isAsync = true
// CHECK: mo.output %[[KERNEL_OUT]] : !mo.tensor<[1, seq_len, 4096], f32>

mo.graph @inout_fusion<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[1, seq_len, 4096], f32>) ->  !mo.tensor<[1, seq_len, 4096], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %1 = mo.add_singleton_dim[0] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
  %out = mo.add(%1, %arg2) : !mo.tensor<[1, seq_len, 4096], f32>
  mo.output %out : !mo.tensor<[1, seq_len, 4096], f32>
}

// CHECK-LABEL: mo.graph @mixed_precision<seq_len>
// CHECK-SAME: (%[[ARG0:.*]]: !mo.tensor<[seq_len, 4096], f32>, %[[ARG1:.*]]: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>)
// CHECK-SAME: -> !mo.tensor<[1, seq_len, 4096], si32>
// CHECK-NEXT: %[[KERNEL_OUT:.*]] = mogg.kernel(%[[ARG0]]: !mo.tensor<[seq_len, 4096], f32>, %[[ARG1]]: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[1, seq_len, 4096], si32>
// CHECK: %[[OUT_PLACEHOLDER:.*]] = mogg.output_placeholders : !mo.tensor<[1, seq_len, 4096], si32>
// CHECK: %[[SCRATCH:.*]] = mogg.scratch_placeholders : !mo.tensor<[seq_len, 4096], f32>
// CHECK: %[[BIND:.*]] = mogg.bind (%[[OUT_PLACEHOLDER]]) output_lambda (%[[ARG4:.*]]: f32, %[[ARG5:.*]]: !pop.array<2, index>)
// CHECK: %[[INSERT_IDX:.*]] = mogg.call "insert_index"(%[[ARG5]]) {axis = 0 : index, value = 0 : index} : !pop.array<2, index> -> !pop.array<3, index>
// CHECK: %[[CAST:.*]] = mogg.call.elementwise["mo.cast"] (%[[ARG4]]) : f32 -> si32
// CHECK: mogg.output %[[CAST]], %[[INSERT_IDX]] : si32, !pop.array<3, index>
// CHECK: }: (!mo.tensor<[1, seq_len, 4096], si32>) -> !mo.tensor<[seq_len, 4096], f32>
// CHECK: %[[CONCRETIZE:.*]] = mogg.tensor.fused(%[[BIND]], %[[SCRATCH]]) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[seq_len, 4096], f32>
// CHECK: mogg.call.execute["mo.matmul"] inputs(%[[ARG0]], %[[ARG1]], {{.*}}
// CHECK: } fusionInfo = Epilogue(Affine(mo.matmul:{{[0-9]+}}, mo.add_singleton_dim:{{[0-9]+}}), mo.cast:{{[0-9]+}}, operandIndices=[0])
// CHECK-SAME: hasReadEffect = false, hasWriteEffect = false, isAsync = true
// CHECK: mo.output %[[KERNEL_OUT]] : !mo.tensor<[1, seq_len, 4096], si32>

mo.graph @mixed_precision<seq_len>(%arg0: !mo.tensor<[seq_len, 4096], f32>, %arg1: !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) ->  !mo.tensor<[1, seq_len, 4096], si32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[seq_len, 4096], f32>, !mo.tensor<[4096, 4096], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[seq_len, 4096], f32>
  %1 = mo.add_singleton_dim[0] (%0) : (!mo.tensor<[seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], f32>
  %out = mo.cast(%1) : (!mo.tensor<[1, seq_len, 4096], f32>) -> !mo.tensor<[1, seq_len, 4096], si32>
  mo.output %out : !mo.tensor<[1, seq_len, 4096], si32>
}
