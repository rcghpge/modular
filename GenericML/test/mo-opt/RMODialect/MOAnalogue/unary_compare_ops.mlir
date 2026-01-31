// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @is_nan_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, bool> {
  %1 = rmo.mo.is_nan<() -> Sh1: !mosh.ape>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, bool>
  mo.output %1 : !mo.tensor<Sh1, bool>
}

// -----

mo.graph @is_nan_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32> {
  // expected-error @+1 {{'rmo.mo.is_nan' op result #0 must be boolean element type, but got '!mo.tensor<Sh1, f32>'}}
  %1 = rmo.mo.is_nan(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, f32>
  mo.output %1 : !mo.tensor<Sh1, f32>
}

// -----

mo.graph @is_inf_ok<Sh0: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, bool> {
  %1 = rmo.mo.is_inf<() -> Sh1: !mosh.ape>(%arg0) : (!mo.tensor<Sh0, f32>) -> !mo.tensor<Sh1, bool>
  mo.output %1 : !mo.tensor<Sh1, bool>
}
