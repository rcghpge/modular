// RUN: mo-opt %s --split-input-file --verify-diagnostics

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//

mo.graph @batch_matmul_too_low_rank(%arg0: !mo.tensor<[10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32> {
  // expected-error@+1 {{batch_matmul input tensors must be at least rank 3, got: 2}}
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// -----

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

mo.graph @rank_too_low(%arg0: !mo.tensor<[10], f32>, %arg1: !mo.tensor<[10], f32>) -> !mo.tensor<[10], f32> {
  // expected-error@+1 {{expected matmul-like input tensors to have at least rank 2, got 1}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10], f32>, !mo.tensor<[10], f32>) -> !mo.tensor<[10], f32>
  mo.output %0 : !mo.tensor<[10], f32>
}

// -----

mo.graph @different_input_ranks(%arg0: !mo.tensor<[10, 10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32> {
  // expected-error@+1 {{input and output ranks must match if static}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 10, 20], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// -----

mo.graph @K_mismatch(%arg0: !mo.tensor<[10, 10], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32> {
  // expected-error@+1 {{expected matmul-like input matrix shapes (M, K) x (K, N), got (10, 10) x (20, 5)}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// -----

mo.graph @output_wrong_shape(%arg0: !mo.tensor<[10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 20], f32> {
  // expected-error@+1 {{expected matmul-like inputs (M, K) x (K, N) to produce result matrix (M, N), got (10, 20) x (20, 5) -> (10, 20)}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 20], f32>
  mo.output %0 : !mo.tensor<[10, 20], f32>
}

// -----

mo.graph @output_wrong_rank(%arg0: !mo.tensor<[10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10], f32> {
  // expected-error@+1 {{input and output ranks must match if static}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10], f32>
  mo.output %0 : !mo.tensor<[10], f32>
}

// -----

mo.graph @matmul_too_high_rank(%arg0: !mo.tensor<[10, 10, 20], f32>, %arg1: !mo.tensor<[10, 20, 5], f32>) -> !mo.tensor<[10, 10, 5], f32> {
  // expected-error@+1 {{matmul input tensors must be rank 2, got: 3}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 10, 20], f32>, !mo.tensor<[10, 20, 5], f32>) -> !mo.tensor<[10, 10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 10, 5], f32>
}
