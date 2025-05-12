// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @gather_ok(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si32>, %axis: !mo.tensor<?, si32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, si32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @gather_bad_axis(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si32>, %axis: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.gather' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, si32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @gather_bad_indices(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.gather' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @gather_bad_result(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si32>, %axis: !mo.tensor<?, si32>) -> !mo.tensor<?, f16> {
  // expected-error @+1 {{'rmo.mo.gather' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.gather(%input, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, si32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f16>
  mo.output %0 : !mo.tensor<?, f16>
}

// -----

mo.graph @gather_nd_ok(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<?, f32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @gather_nd_bad_result(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si32>) -> !mo.tensor<?, f16> {
  // expected-error @+1 {{'rmo.mo.gather_nd' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<?, f32>, !mo.tensor<?, si32>) -> !mo.tensor<?, f16>
  mo.output %0 : !mo.tensor<?, f16>
}

// -----

mo.graph @gather_nd_bad_indices(%input: !mo.tensor<?, f32>, %indices: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.gather_nd' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.gather_nd(%input, %indices) {batchDims = 1 : index} : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
