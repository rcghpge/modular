// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @squeeze_shape_ok<Sh0: !mosh.ape, Sh1: !mosh.ape>(%shape: !mo.tensor<Sh0, si64>, %indices: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, si64> {
  %1 = rmo.mo.squeeze_shape<() -> Sh2: !mosh.ape>(%shape, %indices) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, si64>
  mo.output %1: !mo.tensor<Sh2, si64>
}

// -----

mo.graph @squeeze_shape_bad_indices<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%shape: !mo.tensor<Sh0, si64>, %indices: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, si64> {
  // expected-error @+1 {{'rmo.mo.squeeze_shape' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<Sh1, f32>'}}
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, si64>
  mo.output %1: !mo.tensor<Sh2, si64>
}

// -----

mo.graph @squeeze_shape_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%shape: !mo.tensor<Sh0, si64>, %indices: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, si32> {
  // expected-error @+1 {{'rmo.mo.squeeze_shape' op failed to verify that all of {input_shape, result} have same dtype}}
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, si32>
  mo.output %1: !mo.tensor<Sh2, si32>
}
