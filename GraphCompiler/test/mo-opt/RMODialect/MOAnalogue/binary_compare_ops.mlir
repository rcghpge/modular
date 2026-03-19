// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @equal_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool>
  mo.output %1 : !mo.tensor<Sh2, bool>
}

// -----

mo.graph @equal_bad_operand<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, bool> {
  // expected-error @+1 {{'rmo.mo.equal' op failed to verify that all of {input_x, input_y} have same dtype}}
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, bool>
  mo.output %1 : !mo.tensor<Sh2, bool>
}

// -----

mo.graph @equal_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  // expected-error @+1 {{'rmo.mo.equal' op result #0 must be boolean element type, but got '!mo.tensor<Sh2, f32>'}}
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @greater_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.greater(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool>
  mo.output %1 : !mo.tensor<Sh2, bool>
}

// -----

mo.graph @greater_equal_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.greater_equal(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool>
  mo.output %1 : !mo.tensor<Sh2, bool>
}

// -----

mo.graph @not_equal_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.not_equal(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool>
  mo.output %1 : !mo.tensor<Sh2, bool>
}
