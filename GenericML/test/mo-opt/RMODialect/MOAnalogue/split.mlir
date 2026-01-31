// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @split_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, si64>, %sizes: !mo.tensor<Sh2, si64>) -> (!mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>) {
  %0:2 = rmo.mo.split<() -> Sh3: !mosh.ape, Sh4: !mosh.ape>[%axis: !mo.tensor<Sh1, si64>](%arg0, %sizes) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh2, si64>) -> (!mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>)
  mo.output %0#0, %0#1 : !mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>
}

// -----

mo.graph @split_bad_axis<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, f32>, %sizes: !mo.tensor<Sh2, si64>) -> (!mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>) {
  // expected-error @+1 {{'rmo.mo.split' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %0:2 = rmo.mo.split[%axis: !mo.tensor<Sh1, f32>](%arg0, %sizes) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh2, si64>) -> (!mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>)
  mo.output %0#0, %0#1 : !mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>
}

// -----

mo.graph @split_bad_sizes<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %axis: !mo.tensor<Sh1, si64>, %sizes: !mo.tensor<Sh2, f32>) -> (!mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>) {
  // expected-error @+1 {{'rmo.mo.split' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %0:2 = rmo.mo.split[%axis: !mo.tensor<Sh1, si64>](%arg0, %sizes) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh2, f32>) -> (!mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>)
  mo.output %0#0, %0#1 : !mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>
}
