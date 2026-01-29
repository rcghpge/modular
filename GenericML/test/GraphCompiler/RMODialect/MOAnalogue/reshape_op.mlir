// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @reshape_ok<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %shape: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %1 = rmo.mo.reshape<() -> Sh3: !mosh.ape>(%arg0, %shape) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %1 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @reshape_not_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %shape: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  // expected-error@below {{'rmo.mo.reshape' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<Sh1, f32>'}}
  %1 = rmo.mo.reshape(%arg0, %shape) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %1 : !mo.tensor<Sh2, f32>
}
