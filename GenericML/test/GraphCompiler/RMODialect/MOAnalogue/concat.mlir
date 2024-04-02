// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @concat_ok(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.concat[%axis: !mo.tensor<?, si64>](%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>)
  mo.output %0: !mo.tensor<?, f32>
}

// -----

mo.graph @concat_bax_axis(%arg0: !mo.tensor<?, f32>, %arg1: !mo.tensor<?, f32>, %axis: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.concat' op operand #0 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.concat[%axis: !mo.tensor<?, f32>](%arg0, %arg1) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>)
  mo.output %0: !mo.tensor<?, f32>
}
