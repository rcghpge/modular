// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @conv_transpose_ok(%input : !mo.tensor<?, f32>, %filter : !mo.tensor<?, f32>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, si64>, %pa: !mo.tensor<?, si64>, %op: !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>) {
  %0 = rmo.mo.conv_transpose(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @conv_transpose_bad_operand(%input : !mo.tensor<?, f32>, %filter : !mo.tensor<?, f32>, %st: !mo.tensor<?, si64>, %di: !mo.tensor<?, f32>, %pa: !mo.tensor<?, si64>, %op: !mo.tensor<?, si64>) -> (!mo.tensor<?, f32>) {
  // expected-error@+1 {{'rmo.mo.conv_transpose' op operand #3 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.conv_transpose(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>
  ) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
