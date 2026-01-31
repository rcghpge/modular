// RUN: mo-opt %s --split-input-file

mo.graph @matmul_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @matmul_mixed_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f16>, %arg1: !mo.tensor<Sh1, bf16>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.matmul(%arg0, %arg1) : (!mo.tensor<Sh0, f16>, !mo.tensor<Sh1, bf16>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @batch_matmul_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @batch_matmul_mixed_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f16>, %arg1: !mo.tensor<Sh1, bf16>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<Sh0, f16>, !mo.tensor<Sh1, bf16>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}
