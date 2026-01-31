// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @and_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool>
  mo.output %1 : !mo.tensor<Sh2, bool>
}

// -----

mo.graph @and_bad_operand<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool> {
  // expected-error @+1 {{'rmo.mo.and' op operand #1 must be boolean element type, but got '!mo.tensor<Sh1, f32>'}}
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, bool>
  mo.output %1 : !mo.tensor<Sh2, bool>
}

// -----

mo.graph @and_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, f32> {
  // expected-error @+1 {{'rmo.mo.and' op result #0 must be boolean element type, but got '!mo.tensor<Sh2, f32>'}}
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @or_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.or(%arg0, %arg1) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool>
  mo.output %1 : !mo.tensor<Sh2, bool>
}

// -----

mo.graph @xor_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, bool>, %arg1: !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool> {
  %1 = rmo.mo.xor(%arg0, %arg1) : (!mo.tensor<Sh0, bool>, !mo.tensor<Sh1, bool>) -> !mo.tensor<Sh2, bool>
  mo.output %1 : !mo.tensor<Sh2, bool>
}
