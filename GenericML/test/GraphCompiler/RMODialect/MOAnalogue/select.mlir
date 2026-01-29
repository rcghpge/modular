// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @select_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, f32>, %arg2: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32> {
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32>
  mo.output %1 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @select_bad_condition<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>, %arg2: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32> {
  // expected-error@below {{'rmo.mo.select' op operand #0 must be boolean element type, but got '!mo.tensor<Sh0, f32>'}}
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32>
  mo.output %1 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @select_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, f32>, %arg2: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f16> {
  // expected-error@below {{'rmo.mo.select' op failed to verify that all of {x, y, result} have same dtype}}
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f16>
  mo.output %1 : !mo.tensor<Sh3, f16>
}
