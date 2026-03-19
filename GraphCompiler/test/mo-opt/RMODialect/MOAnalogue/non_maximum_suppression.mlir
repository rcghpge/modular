// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @mms_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape>(%boxes: !mo.tensor<Sh0, f32>, %scores: !mo.tensor<Sh1, f32>, %maxb: !mo.tensor<Sh2, si64>, %iou_thres: !mo.tensor<Sh3, f32>, %scores_thres: !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh5, si64> {
  %0 = rmo.mo.non_maximum_suppression<() -> Sh5: !mosh.ape>(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh5, si64>
  mo.output %0 : !mo.tensor<Sh5, si64>
}

// -----

mo.graph @mms_bax_max_out_boxes<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%boxes: !mo.tensor<Sh0, f32>, %scores: !mo.tensor<Sh1, f32>, %maxb: !mo.tensor<Sh2, f32>, %iou_thres: !mo.tensor<Sh3, f32>, %scores_thres: !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh5, si64> {
  // expected-error @+1 {{'rmo.mo.non_maximum_suppression' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh5, si64>
  mo.output %0 : !mo.tensor<Sh5, si64>
}

// -----

mo.graph @mms_bad_threshold<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%boxes: !mo.tensor<Sh0, f32>, %scores: !mo.tensor<Sh1, f32>, %maxb: !mo.tensor<Sh2, si64>, %iou_thres: !mo.tensor<Sh3, si64>, %scores_thres: !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh5, si64> {
  // expected-error @+1 {{'rmo.mo.non_maximum_suppression' op operand #3 must be rank 0 or unranked with float elements, but got '!mo.tensor<Sh3, si64>'}}
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh5, si64>
  mo.output %0 : !mo.tensor<Sh5, si64>
}

// -----

mo.graph @mms_bad_output<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%boxes: !mo.tensor<Sh0, f32>, %scores: !mo.tensor<Sh1, f32>, %maxb: !mo.tensor<Sh2, si64>, %iou_thres: !mo.tensor<Sh3, f32>, %scores_thres: !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh5, f32> {
  // expected-error @+1 {{'rmo.mo.non_maximum_suppression' op result #0 must be tensor with integer or index elements, but got '!mo.tensor<Sh5, f32>'}}
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, f32>, !mo.tensor<Sh4, f32>) -> !mo.tensor<Sh5, f32>
  mo.output %0 : !mo.tensor<Sh5, f32>
}
