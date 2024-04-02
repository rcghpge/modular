// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @split_ok(%arg0: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>, %sizes: !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>) {
  %0:2 = rmo.mo.split[%axis: !mo.tensor<?, si64>](%arg0, %sizes) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>)
  mo.output %0#0, %0#1 : !mo.tensor<?, f32>, !mo.tensor<?, f32>
}

// -----

mo.graph @split_bad_axis(%arg0: !mo.tensor<?, f32>, %axis: !mo.tensor<?, f32>, %sizes: !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>) {
  // expected-error @+1 {{'rmo.mo.split' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0:2 = rmo.mo.split[%axis: !mo.tensor<?, f32>](%arg0, %sizes) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>)
  mo.output %0#0, %0#1 : !mo.tensor<?, f32>, !mo.tensor<?, f32>
}

// -----

mo.graph @split_bad_sizes(%arg0: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>, %sizes: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>) {
  // expected-error @+1 {{'rmo.mo.split' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0:2 = rmo.mo.split[%axis: !mo.tensor<?, si64>](%arg0, %sizes) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>)
  mo.output %0#0, %0#1 : !mo.tensor<?, f32>, !mo.tensor<?, f32>
}
