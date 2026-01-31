// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @gather_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si32>, %axis: !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.gather<() -> Sh3: !mosh.ape>(%input, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>, !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @gather_bad_axis<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si32>, %axis: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32> {
  // expected-error @+1 {{'rmo.mo.gather' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @gather_bad_indices<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, f32>, %axis: !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f32> {
  // expected-error @+1 {{'rmo.mo.gather' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @gather_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si32>, %axis: !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f16> {
  // expected-error @+1 {{'rmo.mo.gather' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>, !mo.tensor<Sh2, si32>) -> !mo.tensor<Sh3, f16>
  mo.output %0 : !mo.tensor<Sh3, f16>
}

// -----

mo.graph @gather_nd_ok<Sh0: !mosh.ape, Sh1: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.gather_nd<() -> Sh2: !mosh.ape>(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @gather_nd_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f16> {
  // expected-error @+1 {{'rmo.mo.gather_nd' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si32>) -> !mo.tensor<Sh2, f16>
  mo.output %0 : !mo.tensor<Sh2, f16>
}

// -----

mo.graph @gather_nd_bad_indices<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %indices: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  // expected-error @+1 {{'rmo.mo.gather_nd' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}
