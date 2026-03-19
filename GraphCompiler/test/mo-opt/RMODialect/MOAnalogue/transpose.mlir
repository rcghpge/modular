// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @transpose_ok<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %perm: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.transpose<() -> Sh3: !mosh.ape>(%arg0, %perm) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh3, f32>
  mo.output<:!mosh.ape Sh3> %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @transpose_bad_perm<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %perm: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  // expected-error @+1 {{'rmo.mo.transpose' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<Sh1, f32>'}}
  %0 = rmo.mo.transpose(%arg0, %perm) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @transpose_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %perm: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f16> {
  // expected-error @+1 {{'rmo.mo.transpose' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.transpose(%arg0, %perm) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f16>
  mo.output %0 : !mo.tensor<Sh2, f16>
}
