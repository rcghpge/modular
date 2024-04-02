// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @transpose_ok(%arg0: !mo.tensor<?, f32>, %perm: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.transpose(%arg0, %perm) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @transpose_bad_perm(%arg0: !mo.tensor<?, f32>, %perm: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.transpose' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.transpose(%arg0, %perm) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @transpose_bad_result(%arg0: !mo.tensor<?, f32>, %perm: !mo.tensor<?, si64>) -> !mo.tensor<?, f16> {
  // expected-error @+1 {{'rmo.mo.transpose' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.transpose(%arg0, %perm) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f16>
  mo.output %0 : !mo.tensor<?, f16>
}
