// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @rank_too_low(%arg0: !mo.tensor<10xf32>, %arg1: !mo.tensor<20xf32>) -> !mo.tensor<10x20xf32> {
  // expected-error@+1 {{expected matmul-like input tensors to have at least rank 2, got 1}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10xf32>, !mo.tensor<20xf32>) -> !mo.tensor<10x20xf32>
  mo.output %0 : !mo.tensor<10x20xf32>
}

// -----

mo.graph @different_ranks(%arg0: !mo.tensor<10x10x20xf32>, %arg1: !mo.tensor<20x5xf32>) -> !mo.tensor<10x5xf32> {
  // expected-error@+1 {{matmul-like input tensors are not the same rank, got 3 and 2}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x10x20xf32>, !mo.tensor<20x5xf32>) -> !mo.tensor<10x5xf32>
  mo.output %0 : !mo.tensor<10x5xf32>
}

// -----

mo.graph @too_many_ranks(%arg0: !mo.tensor<10x10x20xf32>, %arg1: !mo.tensor<10x20x5xf32>) -> !mo.tensor<10x10x5xf32> {
  // expected-error@+1 {{matmul input tensors must be rank 2, got: 3}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x10x20xf32>, !mo.tensor<10x20x5xf32>) -> !mo.tensor<10x10x5xf32>
  mo.output %0 : !mo.tensor<10x10x5xf32>
}

// -----

mo.graph @K_mismatch(%arg0: !mo.tensor<10x10xf32>, %arg1: !mo.tensor<20x5xf32>) -> !mo.tensor<10x5xf32> {
  // expected-error@+1 {{expected matmul-like input matrix shapes (M, K) x (K, N), got (10, 10) x (20, 5)}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x10xf32>, !mo.tensor<20x5xf32>) -> !mo.tensor<10x5xf32>
  mo.output %0 : !mo.tensor<10x5xf32>
}

// -----

mo.graph @output_wrong_shape(%arg0: !mo.tensor<10x20xf32>, %arg1: !mo.tensor<20x5xf32>) -> !mo.tensor<10x20xf32> {
  // expected-error@+1 {{expected matmul-like inputs (M, K) x (K, N) to produce result matrix (M, N), got (10, 20) x (20, 5) -> (10, 20)}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x20xf32>, !mo.tensor<20x5xf32>) -> !mo.tensor<10x20xf32>
  mo.output %0 : !mo.tensor<10x20xf32>
}

// -----

mo.graph @output_wrong_rank(%arg0: !mo.tensor<10x20xf32>, %arg1: !mo.tensor<20x5xf32>) -> !mo.tensor<10xf32> {
  // expected-error@+1 {{expected matmul-like result tensor to have the same rank as the input (2), got 1}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x20xf32>, !mo.tensor<20x5xf32>) -> !mo.tensor<10xf32>
  mo.output %0 : !mo.tensor<10xf32>
}
