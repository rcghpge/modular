// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @random_uniform_ok(%size: !mo.tensor<?, si64>, %lowerBound: !mo.tensor<?, f32>, %upperBound: !mo.tensor<?, f32>, %seed: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.random.uniform(%size, %lowerBound, %upperBound, %seed) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @random_uniform_bad_variance(%size: !mo.tensor<?, si64>, %lowerBound: !mo.tensor<?, f32>, %upperBound: !mo.tensor<?, f16>, %seed: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.random.uniform' op failed to verify that all of {lowerBound, upperBound, result} have same dtype}}
  %0 = rmo.mo.random.uniform(%size, %lowerBound, %upperBound, %seed) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f16>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @random_uniform_bad_size(%size: !mo.tensor<?, f32>, %lowerBound: !mo.tensor<?, f32>, %upperBound: !mo.tensor<?, f32>, %seed: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.random.uniform' op operand #0 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.random.uniform(%size, %lowerBound, %upperBound, %seed) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @random_uniform_bad_seed(%size: !mo.tensor<?, si64>, %lowerBound: !mo.tensor<?, f32>, %upperBound: !mo.tensor<?, f32>, %seed: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.random.uniform' op operand #3 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.random.uniform(%size, %lowerBound, %upperBound, %seed) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
