// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @conv_ok(%input : !mo.tensor<?, f32>, %filter : !mo.tensor<?, f32>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>, %ng: !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>) {
  %0 = rmo.mo.conv(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @conv_bad_operands(%input : !mo.tensor<?, f32>, %filter : !mo.tensor<?, f32>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, f32>, %pa: !mo.tensor<?, si64>, %ng: !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>) {
  // expected-error @+1{{'rmo.mo.conv' op operand #3 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.conv(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
