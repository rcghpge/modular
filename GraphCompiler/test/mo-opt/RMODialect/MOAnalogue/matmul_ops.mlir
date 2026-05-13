// RUN: mo-opt %s --split-input-file

mo.graph @matmul_ok<M, K, N>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>
  mo.output %0 : !mo.tensor<[M, N], f32>
}

// -----

mo.graph @matmul_mixed_ok<M, K, N>(%arg0: !mo.tensor<[M, K], f16>, %arg1: !mo.tensor<[K, N], bf16>) -> !mo.tensor<[M, N], f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f16>, !mo.tensor<[K, N], bf16>) -> !mo.tensor<[M, N], f32>
  mo.output %0 : !mo.tensor<[M, N], f32>
}

// -----

mo.graph @batch_matmul_ok<B, M, K, N>(%arg0: !mo.tensor<[B, M, K], f32>, %arg1: !mo.tensor<[B, K, N], f32>) -> !mo.tensor<[B, M, N], f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[B, M, K], f32>, !mo.tensor<[B, K, N], f32>) -> !mo.tensor<[B, M, N], f32>
  mo.output %0 : !mo.tensor<[B, M, N], f32>
}

// -----

mo.graph @batch_matmul_mixed_ok<B, M, K, N>(%arg0: !mo.tensor<[B, M, K], f16>, %arg1: !mo.tensor<[B, K, N], bf16>) -> !mo.tensor<[B, M, N], f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[B, M, K], f16>, !mo.tensor<[B, K, N], bf16>) -> !mo.tensor<[B, M, N], f32>
  mo.output %0 : !mo.tensor<[B, M, N], f32>
}
