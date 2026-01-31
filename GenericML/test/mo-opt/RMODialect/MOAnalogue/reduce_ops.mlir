// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @mean_ok<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.mean<() -> Sh2: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @mean_bad_axis<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  // expected-error@+1 {{'rmo.mo.mean' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %1 = rmo.mo.mean(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @mean_bad_output<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f16> {
  // expected-error@+1 {{'rmo.mo.mean' op failed to verify that all of {input, result} have same dtype}}
  %1 = rmo.mo.mean(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f16>
  mo.output %1 : !mo.tensor<Sh2, f16>
}

// -----

mo.graph @argmin_ok<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, si64> {
  %1 = rmo.mo.arg_min<() -> Sh2: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, si64>
  mo.output %1 : !mo.tensor<Sh2, si64>
}

// -----

mo.graph @argmin_bad_output<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32> {
  // expected-error @+1 {{'rmo.mo.arg_min' op result #0 must be tensor with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %1 = rmo.mo.arg_min(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @mean_param_shapes<Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh1, f32>, %arg1: !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.mean<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @mean_param_shapes_bad_output<Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh1, f32>, %arg1: !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f16> {
  // expected-error@+1 {{'rmo.mo.mean' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.mean<() -> Sh3: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f16>
  mo.output %0 : !mo.tensor<Sh3, f16>
}
