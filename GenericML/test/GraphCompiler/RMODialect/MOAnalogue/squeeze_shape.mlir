// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @squeeze_shape_ok(%shape: !mo.tensor<?, si64>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, si64> {
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, si64>
  mo.output %1: !mo.tensor<?, si64>
}

// -----

mo.graph @squeeze_shape_bad_indices(%shape: !mo.tensor<?, si64>, %indices: !mo.tensor<?, f32>) -> !mo.tensor<?, si64> {
  // expected-error @+1 {{'rmo.mo.squeeze_shape' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<?, f32>'}}
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>) -> !mo.tensor<?, si64>
  mo.output %1: !mo.tensor<?, si64>
}

// -----

mo.graph @squeeze_shape_bad_result(%shape: !mo.tensor<?, si64>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, si32> {
  // expected-error @+1 {{'rmo.mo.squeeze_shape' op failed to verify that all of {input_shape, result} have same dtype}}
  %1 = rmo.mo.squeeze_shape(%shape, %indices) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<?, si32>
  mo.output %1: !mo.tensor<?, si32>
}
