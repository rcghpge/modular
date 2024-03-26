// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @equal_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}

// -----

mo.graph @equal_bad_operand(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, si32>) -> !mo.tensor<?, bool> {
  // expected-error @+1 {{'rmo.mo.equal' op failed to verify that all of {input_x, input_y} have same dtype}}
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, si32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}

// -----

mo.graph @equal_bad_result(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.equal' op result #0 must be boolean element type, but got '!mo.tensor<?, f32>'}}
  %1 = rmo.mo.equal(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @greater_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.greater(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}

// -----

mo.graph @greater_equal_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.greater_equal(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}

// -----

mo.graph @not_equal_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.not_equal(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
