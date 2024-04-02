// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @tile_ok(%arg0: !mo.tensor<?, f32>, %repeats: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @tile_bad_repeats(%arg0: !mo.tensor<?, f32>, %repeats: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.tile' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @tile_bad_result(%arg0: !mo.tensor<?, f32>, %repeats: !mo.tensor<?, si64>) -> !mo.tensor<?, f16> {
  // expected-error @+1 {{'rmo.mo.tile' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.tile(%arg0, %repeats) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f16>
  mo.output %0 : !mo.tensor<?, f16>
}
