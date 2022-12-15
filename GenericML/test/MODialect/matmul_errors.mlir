// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @different_ranks(%arg0: !mo.tensor<10x10x20xf32>, %arg1: !mo.tensor<20x5xf32>) -> (!mo.tensor<10x5xf32>) {
  // expected-error@+1 {{Matmul input tensors are not the same rank 3 vs 2}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x10x20xf32>, !mo.tensor<20x5xf32>) -> !mo.tensor<10x5xf32>
  mo.output %0 : !mo.tensor<10x5xf32>
}

// -----

mo.graph @too_many_ranks(%arg0: !mo.tensor<10x10x20xf32>, %arg1: !mo.tensor<10x20x5xf32>) -> (!mo.tensor<10x10x5xf32>) {
  // expected-error@+1 {{Matmul input tensors must be rank 2, got: 3}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x10x20xf32>, !mo.tensor<10x20x5xf32>) -> !mo.tensor<10x10x5xf32>
  mo.output %0 : !mo.tensor<10x10x5xf32>
}

// -----

mo.graph @K_mismatch(%arg0: !mo.tensor<10x10xf32>, %arg1: !mo.tensor<20x5xf32>) -> (!mo.tensor<10x5xf32>) {
  // expected-error@+1 {{Matmul input tensors are expected to be (M, K) * (K, N) but input shapes are (10, 10) * (20, 5)}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x10xf32>, !mo.tensor<20x5xf32>) -> !mo.tensor<10x5xf32>
  mo.output %0 : !mo.tensor<10x5xf32>
}

// -----

mo.graph @output_wrong_shape(%arg0: !mo.tensor<10x20xf32>, %arg1: !mo.tensor<20x5xf32>) -> (!mo.tensor<10x20xf32>) {
  // expected-error@+1 {{Matmul of (M, K) * (K, N) does not produce shape (M, N): (10, 20) * (20, 5) -> (10, 20)}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<10x20xf32>, !mo.tensor<20x5xf32>) -> !mo.tensor<10x20xf32>
  mo.output %0 : !mo.tensor<10x20xf32>
}
