// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @resize_linear_ok<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %size: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.resize.linear<() -> Sh2: !mosh.ape>(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @resize_linear_bad_operands<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %size: !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32> {
  // expected-error @+1 {{'rmo.mo.resize.linear' op operand #1 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %0 = rmo.mo.resize.linear(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = false}: (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @resize_nearest_ok<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %size: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.resize.nearest<() -> Sh2: !mosh.ape>(%arg0, %size) {coordinate_transform_mode = #mo<coord_trans_mode half_pixel>, round_mode = 0}: (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}

// -----

mo.graph @resize_bicubic_ok<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %size: !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32> {
  %0 = rmo.mo.resize.bicubic<() -> Sh2: !mosh.ape>(%arg0, %size) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<Sh2, f32>
  mo.output %0 : !mo.tensor<Sh2, f32>
}
