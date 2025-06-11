// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @resize_linear_ok(%arg0: !mo.tensor<?, f32>, %size: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.resize.linear(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @resize_linear_bad_operands(%arg0: !mo.tensor<?, f32>, %size: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.resize.linear' op operand #1 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.resize.linear(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @resize_nearest_ok(%arg0: !mo.tensor<?, f32>, %size: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.resize.nearest(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode half_pixel>, round_mode = 0}: (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @resize_bicubic_ok(%arg0: !mo.tensor<?, f32>, %size: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.resize.bicubic(%arg0, %size) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
