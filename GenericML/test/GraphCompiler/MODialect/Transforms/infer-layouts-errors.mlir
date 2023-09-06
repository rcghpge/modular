// RUN: mo-opt %s --infer-layouts --split-input-file --verify-diagnostics

mo.graph @conv_same_input_and_filter(
  %input: !mo.tensor<[?, ?, ?, ?], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>,
  %paddings: !mo.tensor<[4], si64>
) -> !mo.tensor<[?, ?, ?, ?], f32> {

  // expected-error @below {{attempting to constrain value to layout: #mo.layout<RSCF> when it's already constrained to layout: #mo.layout<NHWC>}}
  %conv1 = mo.conv(%input, %input) [strides = %strides, dilations = %dilations, paddings = %paddings] : (
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[4], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  mo.output %conv1 : !mo.tensor<[?, ?, ?, ?], f32>
}
