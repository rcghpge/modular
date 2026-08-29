// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=1x4 --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=1x4 --result-output-style=full %t.legacy.mlir | FileCheck %s

// Rank-preserving case: in-shape dim 0 (size 1) broadcasts to 3, verifying
// the actual broadcast values (not just shape) -- row [1, 2, 3, 4] repeats.

// CHECK-LABEL: Running 'broadcast_same_rank':
// CHECK: 'broadcast_same_rank' returned tensor<3x4xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]
mo.graph @broadcast_same_rank() -> (!mo.tensor<[3, 4], f32>) {
  %input = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[1, 4], f32>
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[1, 4], f32> -> !mo.tensor<[3, 4], f32>
  mo.output %res : !mo.tensor<[3, 4], f32>
}

// Rank-expanding case (numpy semantics): in has one fewer dim than out, so
// the whole input aligns with out's trailing dims and every leading out
// dim is replicated. Same input/output values as the rank-preserving case
// above, proving the two paths agree numerically despite differing rank.

// CHECK-LABEL: Running 'broadcast_rank_expanding':
// CHECK: 'broadcast_rank_expanding' returned tensor<3x4xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]
mo.graph @broadcast_rank_expanding() -> (!mo.tensor<[3, 4], f32>) {
  %input = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[4], f32> -> !mo.tensor<[3, 4], f32>
  mo.output %res : !mo.tensor<[3, 4], f32>
}

// Dynamic case: symbolic input shape (filled with all-ones by `mt`), static
// broadcast factor -- proving the symbolic-dim path resolves correctly too.

// CHECK-LABEL: Running 'broadcast_dynamic':
// CHECK: 'broadcast_dynamic' returned tensor<3x4xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
mo.graph @broadcast_dynamic<M>(%input: !mo.tensor<[1, M], f32>) -> (!mo.tensor<[3, M], f32>) {
  %res = mo.static.broadcast_to(%input) : !mo.tensor<[1, M], f32> -> !mo.tensor<[3, M], f32>
  mo.output %res : !mo.tensor<[3, M], f32>
}
