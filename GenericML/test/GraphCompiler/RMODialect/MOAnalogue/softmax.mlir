// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @softmax_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.softmax(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @softmax_bad_result(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<?, f16> {
  // expected-error @+1 {{op failed to verify that all of {input, result} have same dtype}}
  %1 = rmo.mo.softmax(%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f16>
  mo.output %1 : !mo.tensor<?, f16>
}
