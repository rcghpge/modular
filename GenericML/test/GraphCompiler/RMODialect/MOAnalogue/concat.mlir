// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @concat_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>, %axis: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.concat<() -> Sh4: !mosh.ape>[%axis: !mo.tensor<Sh2, si64>](%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> (!mo.tensor<Sh4, f32>)
  mo.output<:!mosh.ape Sh4> %0: !mo.tensor<Sh4, f32>
}

// -----

mo.graph @concat_bax_axis<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>, %axis: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32> {
  // expected-error @+1 {{'rmo.mo.concat' op operand #0 must be tensor with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %0 = rmo.mo.concat[%axis: !mo.tensor<Sh2, f32>](%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> (!mo.tensor<Sh3, f32>)
  mo.output %0: !mo.tensor<Sh3, f32>
}
