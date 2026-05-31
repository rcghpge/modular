// RUN: mo-opt %s --mogg-pattern-fusion=mock-arch-name=sm_100 | FileCheck %s

// CHECK-LABEL: mo.graph @matmul_add_2d
// CHECK: %[[OUT:.*]] = mo.fused_matmul_add
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_add_2d(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> !mo.tensor<[65536, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %1 : !mo.tensor<[65536, 1536], bf16, gpu:0>
}

// CHECK-LABEL: mo.graph @matmul_reshape_add_2d
// CHECK: %[[FUSED:.*]] = mo.fused_matmul_add{{.*}}transpose_b = true
// CHECK: %[[OUT:.*]] = mo.static.reshape(%[[FUSED]]) : !mo.tensor<[65536, 1536], bf16, gpu:0> -> !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_reshape_add_2d(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0>, %arg2: !mo.tensor<[32, 2048, 1536], bf16, gpu:0>) -> !mo.tensor<[32, 2048, 1536], bf16, gpu:0> {
  %0 = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %1 = mo.transpose(%arg1, %0) : (!mo.tensor<[1536, 1536], bf16, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>
  %2 = mo.matmul(%arg0, %1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %3 = mo.static.reshape(%2) : !mo.tensor<[65536, 1536], bf16, gpu:0> -> !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
  %4 = mo.add(%3, %arg2) : !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
  mo.output %4 : !mo.tensor<[32, 2048, 1536], bf16, gpu:0>
}

// CHECK-LABEL: mo.graph @matmul_add_1d
// CHECK-NOT: mo.static.broadcast_to
// CHECK: %[[OUT:.*]] = mo.fused_matmul_add
// CHECK: mo.output %[[OUT]]
mo.graph @matmul_add_1d(%arg0: !mo.tensor<[256, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[8192, 1536], bf16, gpu:0>, %arg2: !mo.tensor<[8192], bf16, gpu:0>) -> !mo.tensor<[256, 8192], bf16, gpu:0> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[8192, 1536], bf16, gpu:0> -> !mo.tensor<[8192, 1536], bf16, gpu:0, {layout = #mo.layout<NK>}>
  %1 = mo.matmul(%arg0, %0) {transpose_b = true} : (!mo.tensor<[256, 1536], bf16, gpu:0>, !mo.tensor<[8192, 1536], bf16, gpu:0, {layout = #mo.layout<NK>}>) -> !mo.tensor<[256, 8192], bf16, gpu:0>
  %2 = mo.static.broadcast_to(%arg2) : !mo.tensor<[8192], bf16, gpu:0> -> !mo.tensor<[256, 8192], bf16, gpu:0>
  %3 = mo.add(%1, %2) : !mo.tensor<[256, 8192], bf16, gpu:0>
  mo.output %3 : !mo.tensor<[256, 8192], bf16, gpu:0>
}

// Negative: matmul has multiple uses — output is used by the add AND returned directly.
// CHECK-LABEL: mo.graph @neg_matmul_multiple_uses
// CHECK: mo.matmul
// CHECK-NOT: mo.fused_matmul_add
// CHECK: mo.add
mo.graph @neg_matmul_multiple_uses(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[65536, 1536], bf16, gpu:0>) {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %0, %1 : !mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[65536, 1536], bf16, gpu:0>
}

// Negative: matmul is GEMV (M == 1).
// CHECK-LABEL: mo.graph @neg_gemv
// CHECK: mo.matmul
// CHECK-NOT: mo.fused_matmul_add
// CHECK: mo.add
mo.graph @neg_gemv(%arg0: !mo.tensor<[1, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[1, 1536], bf16, gpu:0>) -> !mo.tensor<[1, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[1, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[1, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[1, 1536], bf16, gpu:0>
  mo.output %1 : !mo.tensor<[1, 1536], bf16, gpu:0>
}

// Negative: matmul has a dynamic M dimension.
// CHECK-LABEL: mo.graph @neg_dynamic_m
// CHECK: mo.matmul
// CHECK-NOT: mo.fused_matmul_add
// CHECK: mo.add
mo.graph @neg_dynamic_m(%arg0: !mo.tensor<[dyn_m, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[dyn_m, 1536], bf16, gpu:0>) -> !mo.tensor<[dyn_m, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[dyn_m, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[dyn_m, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[dyn_m, 1536], bf16, gpu:0>
  mo.output %1 : !mo.tensor<[dyn_m, 1536], bf16, gpu:0>
}

// Negative: add's residual input is produced by an elementwise op.
// CHECK-LABEL: mo.graph @neg_elementwise_residual
// CHECK: mo.matmul
// CHECK-NOT: mo.fused_matmul_add
// CHECK: mo.add
mo.graph @neg_elementwise_residual(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> !mo.tensor<[65536, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %1 = mo.activation(%arg2) {activation = "relu"} : !mo.tensor<[65536, 1536], bf16, gpu:0>
  %2 = mo.add(%0, %1) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %2 : !mo.tensor<[65536, 1536], bf16, gpu:0>
}

// Negative: elementwise op after the add would miss epilogue fusion with fused_matmul_add.
// CHECK-LABEL: mo.graph @neg_downstream_elementwise
// CHECK: mo.matmul
// CHECK-NOT: mo.fused_matmul_add
// CHECK: mo.add
mo.graph @neg_downstream_elementwise(%arg0: !mo.tensor<[65536, 1536], bf16, gpu:0>, %arg1: !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>, %arg2: !mo.tensor<[65536, 1536], bf16, gpu:0>) -> !mo.tensor<[65536, 1536], bf16, gpu:0> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[65536, 1536], bf16, gpu:0>, !mo.tensor<[1536, 1536], bf16, gpu:0, {layout = #mo.layout<KN>}>) -> !mo.tensor<[65536, 1536], bf16, gpu:0>
  %1 = mo.add(%0, %arg2) : !mo.tensor<[65536, 1536], bf16, gpu:0>
  %2 = mo.activation(%1) {activation = "relu"} : !mo.tensor<[65536, 1536], bf16, gpu:0>
  mo.output %2 : !mo.tensor<[65536, 1536], bf16, gpu:0>
}
