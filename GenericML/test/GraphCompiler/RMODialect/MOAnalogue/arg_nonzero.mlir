// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @argnonzero_ok<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, si64> {
  %0 = rmo.mo.arg_nonzero<() -> Sh2: !mosh.ape>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh2, si64>
  mo.output<:!mosh.ape Sh2> %0 : !mo.tensor<Sh2, si64>
}

// -----

mo.graph @argnonzero_bad_output<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32> {
  // expected-error @+1 {{'rmo.mo.arg_nonzero' op result #0 must be signed 64 integer element type, but got '!mo.tensor<Sh1, f32>'}}
  %0 = rmo.mo.arg_nonzero(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32>
  mo.output %0 : !mo.tensor<Sh1, f32>
}
