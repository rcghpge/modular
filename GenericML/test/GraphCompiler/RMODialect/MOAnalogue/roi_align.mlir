// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @roi_align_ok(%input: !mo.tensor<?, f32>, %rois: !mo.tensor<?, f32>, %output_height: !mo.tensor<?, si64>, %output_width: !mo.tensor<?, si32>, %spatial_scale: !mo.tensor<?, f32>, %sampling_ratio: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @roi_align_bad_height(%input: !mo.tensor<?, f32>, %rois: !mo.tensor<?, f32>, %output_height: !mo.tensor<?, f32>, %output_width: !mo.tensor<?, si32>, %spatial_scale: !mo.tensor<?, f32>, %sampling_ratio: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.roi_align' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @roi_align_bad_scale(%input: !mo.tensor<?, f32>, %rois: !mo.tensor<?, f32>, %output_height: !mo.tensor<?, si64>, %output_width: !mo.tensor<?, si32>, %spatial_scale: !mo.tensor<?, si32>, %sampling_ratio: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.roi_align' op operand #4 must be rank 0 or unranked with float elements, but got '!mo.tensor<?, si32>'}}
  %0 = rmo.mo.roi_align(%input, %rois, %output_height, %output_width, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, si32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
