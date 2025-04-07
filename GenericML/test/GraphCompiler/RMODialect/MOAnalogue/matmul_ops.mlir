// RUN: mo-opt %s --split-input-file

mo.graph @matmul_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @matmul_mixed_ok(%arg0: !mo.tensor<?, f16>, %arg1: !mo.tensor<?, bf16>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<?, f16>, !mo.tensor<?, bf16>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @batch_matmul_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @batch_matmul_mixed_ok(%arg0: !mo.tensor<?, f16>, %arg1: !mo.tensor<?, bf16>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<?, f16>, !mo.tensor<?, bf16>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
