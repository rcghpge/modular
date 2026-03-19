// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @roi_align_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %rois: !mo.tensor<Sh1, f32>, %output_height: !mo.tensor<Sh2, si64>, %output_width: !mo.tensor<Sh3, si32>, %spatial_scale: !mo.tensor<Sh4, f32>, %sampling_ratio: !mo.tensor<Sh5, f32>) -> !mo.tensor<Sh6, f32> {
  %0 = rmo.mo.roi_align<() -> Sh6: !mosh.ape>(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si32>, !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, f32>) -> !mo.tensor<Sh6, f32>
  mo.output %0 : !mo.tensor<Sh6, f32>
}

// -----

mo.graph @roi_align_bad_height<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape, Sh6: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %rois: !mo.tensor<Sh1, f32>, %output_height: !mo.tensor<Sh2, f32>, %output_width: !mo.tensor<Sh3, si32>, %spatial_scale: !mo.tensor<Sh4, f32>, %sampling_ratio: !mo.tensor<Sh5, f32>) -> !mo.tensor<Sh6, f32> {
  // expected-error @+1 {{'rmo.mo.roi_align' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, si32>, !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, f32>) -> !mo.tensor<Sh6, f32>
  mo.output %0 : !mo.tensor<Sh6, f32>
}

// -----

mo.graph @roi_align_bad_scale<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape, Sh6: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %rois: !mo.tensor<Sh1, f32>, %output_height: !mo.tensor<Sh2, si64>, %output_width: !mo.tensor<Sh3, si32>, %spatial_scale: !mo.tensor<Sh4, si32>, %sampling_ratio: !mo.tensor<Sh5, f32>) -> !mo.tensor<Sh6, f32> {
  // expected-error @+1 {{'rmo.mo.roi_align' op operand #4 must be rank 0 or unranked with float elements, but got '!mo.tensor<Sh4, si32>'}}
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si32>, !mo.tensor<Sh4, si32>, !mo.tensor<Sh5, f32>) -> !mo.tensor<Sh6, f32>
  mo.output %0 : !mo.tensor<Sh6, f32>
}
