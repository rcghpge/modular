// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @is_nan_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.is_nan(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}

// -----

mo.graph @is_nan_bad_result(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.is_nan' op result #0 must be boolean element type, but got '!mo.tensor<?, f32>'}}
  %1 = rmo.mo.is_nan(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @is_inf_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, bool> {
  %1 = rmo.mo.is_inf(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, bool>
  mo.output %1 : !mo.tensor<?, bool>
}
