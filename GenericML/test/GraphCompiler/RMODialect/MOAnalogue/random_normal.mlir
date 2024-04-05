// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @random_normal_ok(%size: !mo.tensor<?, si64>, %mean: !mo.tensor<?, f32>, %variance: !mo.tensor<?, f32>, %seed: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @random_normal_bad_mean(%size: !mo.tensor<?, si64>, %mean: !mo.tensor<?, si32>, %variance: !mo.tensor<?, f32>, %seed: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op operand #1 must be rank 0 or unranked with float elements, but got '!mo.tensor<?, si32>'}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @random_normal_bad_variance(%size: !mo.tensor<?, si64>, %mean: !mo.tensor<?, f32>, %variance: !mo.tensor<?, f16>, %seed: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op failed to verify that all of {mean, variance} have same dtype}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f16>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @random_normal_bad_size(%size: !mo.tensor<?, f32>, %mean: !mo.tensor<?, f32>, %variance: !mo.tensor<?, f32>, %seed: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op operand #0 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @random_normal_bad_seed(%size: !mo.tensor<?, si64>, %mean: !mo.tensor<?, f32>, %variance: !mo.tensor<?, f32>, %seed: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op operand #3 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
