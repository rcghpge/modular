// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @random_normal_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%size: !mo.tensor<Sh0, si64>, %mean: !mo.tensor<Sh1, f32>, %variance: !mo.tensor<Sh2, f32>, %seed: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  %0 = rmo.mo.random.normal<() -> Sh4: !mosh.ape>(%size, %mean, %variance, %seed) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @random_normal_bad_mean<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%size: !mo.tensor<Sh0, si64>, %mean: !mo.tensor<Sh1, si32>, %variance: !mo.tensor<Sh2, f32>, %seed: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op operand #1 must be rank 0 or unranked with float elements, but got '!mo.tensor<Sh1, si32>'}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si32>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @random_normal_bad_variance<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%size: !mo.tensor<Sh0, si64>, %mean: !mo.tensor<Sh1, f32>, %variance: !mo.tensor<Sh2, f16>, %seed: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op failed to verify that all of {mean, variance} have same dtype}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f16>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @random_normal_bad_size<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%size: !mo.tensor<Sh0, f32>, %mean: !mo.tensor<Sh1, f32>, %variance: !mo.tensor<Sh2, f32>, %seed: !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op operand #0 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<Sh0, f32>'}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, si64>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}

// -----

mo.graph @random_normal_bad_seed<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%size: !mo.tensor<Sh0, si64>, %mean: !mo.tensor<Sh1, f32>, %variance: !mo.tensor<Sh2, f32>, %seed: !mo.tensor<Sh3, f32>) -> !mo.tensor<Sh4, f32> {
  // expected-error @+1 {{'rmo.mo.random.normal' op operand #3 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<Sh3, f32>'}}
  %0 = rmo.mo.random.normal(%size, %mean, %variance, %seed) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, f32>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}
