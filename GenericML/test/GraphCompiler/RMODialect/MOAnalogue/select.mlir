// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @select_ok(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, f32>, %arg2: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<?, bool>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @select_bad_cond(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>, %arg2: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.select' op operand #0 must be boolean element type, but got '!mo.tensor<?, f32>'}}
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @select_bad_operand(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, f32>, %arg2: !mo.tensor<?, f16>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.select' op failed to verify that all of {x, y, result} have same dtype}}
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<?, bool>, !mo.tensor<?, f32>, !mo.tensor<?, f16>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @select_bad_result(%arg0: !mo.tensor<?, bool>, %arg1: !mo.tensor<?, f32>, %arg2: !mo.tensor<?, f32>) -> !mo.tensor<?, f16> {
  // expected-error @+1 {{'rmo.mo.select' op failed to verify that all of {x, y, result} have same dtype}}
  %1 = rmo.mo.select(%arg0, %arg1, %arg2) : (!mo.tensor<?, bool>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f16>
  mo.output %1 : !mo.tensor<?, f16>
}
