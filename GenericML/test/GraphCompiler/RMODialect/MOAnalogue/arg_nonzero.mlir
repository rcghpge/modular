// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @argnonzero_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, si64> {
  %0 = rmo.mo.arg_nonzero(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}

// -----

mo.graph @argnonzero_bad_output(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.arg_nonzero' op result #0 must be signed 64 integer element type, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.arg_nonzero(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
