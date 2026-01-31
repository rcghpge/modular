// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @softmax_ok<Sh0: !mosh.ape -> Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f32> {
  %1 = rmo.mo.softmax<() -> Sh2: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, f32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @softmax_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f16> {
  // expected-error @+1 {{op failed to verify that all of {input, result} have same dtype}}
  %1 = rmo.mo.softmax(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f16>
  mo.output %1 : !mo.tensor<Sh1, f16>
}
