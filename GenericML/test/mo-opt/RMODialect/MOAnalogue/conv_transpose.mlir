// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @conv_transpose_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape -> Sh6: !mosh.ape>(%input : !mo.tensor<Sh0, f32>, %filter : !mo.tensor<Sh1, f32>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, si64>, %pa: !mo.tensor<Sh4, si64>, %op: !mo.tensor<Sh5, si64>) -> (!mo.tensor<Sh6, f32>) {
  %0 = rmo.mo.conv_transpose<() -> Sh7: !mosh.ape>(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>, !mo.tensor<Sh5, si64>
  ) -> !mo.tensor<Sh7, f32>
  mo.output<:!mosh.ape Sh7> %0 : !mo.tensor<Sh7, f32>
}

// -----

mo.graph @conv_transpose_bad_operand<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape, Sh6: !mosh.ape>(%input : !mo.tensor<Sh0, f32>, %filter : !mo.tensor<Sh1, f32>, %st: !mo.tensor<Sh2, si64>, %di: !mo.tensor<Sh3, f32>, %pa: !mo.tensor<Sh4, si64>, %op: !mo.tensor<Sh5, si64>) -> (!mo.tensor<Sh6, f32>) {
  // expected-error@+1 {{'rmo.mo.conv_transpose' op operand #3 must be tensor with integer or index elements, but got '!mo.tensor<Sh3, f32>'}}
  %0 = rmo.mo.conv_transpose(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, f32>, !mo.tensor<Sh4, si64>, !mo.tensor<Sh5, si64>
  ) -> !mo.tensor<Sh6, f32>
  mo.output %0 : !mo.tensor<Sh6, f32>
}
