// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @and_ok(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<?, bool>, !mo.tensor<?, bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}

// -----

mo.graph @and_bad_operand(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  // expected-error @+1 {{'rmo.mo.and' op operand #1 must be boolean element type, but got '!mo.tensor<?, f32>'}}
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<?, bool>, !mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}

// -----

mo.graph @and_bad_result(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, bool>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.and' op result #0 must be boolean element type, but got '!mo.tensor<?, f32>'}}
  %1 = rmo.mo.and(%arg0, %arg1) : (!mo.tensor<?, bool>, !mo.tensor<?, bool>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @or_ok(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.or(%arg0, %arg1) : (!mo.tensor<?, bool>, !mo.tensor<?, bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}

// -----

mo.graph @xor_ok(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, bool>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.xor(%arg0, %arg1) : (!mo.tensor<?, bool>, !mo.tensor<?, bool>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
