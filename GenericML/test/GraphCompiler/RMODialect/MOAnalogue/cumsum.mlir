// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @cumsum_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, si64>) -> (!mo.tensor<Sh2, f32>) {
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @cumsum_bad_axis<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, f32>) -> (!mo.tensor<Sh2, f32>) {
  // expected-error @+1 {{'rmo.mo.cumsum' op operand #1 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @cumsum_bad_output<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, si64>) -> (!mo.tensor<Sh2, f16>) {
  // expected-error @+1 {{'rmo.mo.cumsum' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f16>
  mo.output %0 : !mo.tensor<Sh2, f16>
}
