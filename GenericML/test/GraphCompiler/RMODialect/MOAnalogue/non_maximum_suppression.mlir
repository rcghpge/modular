// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @mms_ok(%boxes: !mo.tensor<?, f32>, %scores: !mo.tensor<?, f32>, %maxb: !mo.tensor<?, si64>, %iou_thres: !mo.tensor<?, f32>, %scores_thres: !mo.tensor<?, f32>) -> !mo.tensor<?, si64> {
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}

// -----

mo.graph @mms_bax_max_out_boxes(%boxes: !mo.tensor<?, f32>, %scores: !mo.tensor<?, f32>, %maxb: !mo.tensor<?, f32>, %iou_thres: !mo.tensor<?, f32>, %scores_thres: !mo.tensor<?, f32>) -> !mo.tensor<?, si64> {
  // expected-error @+1 {{'rmo.mo.non_maximum_suppression' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}

// -----

mo.graph @mms_bad_threshold(%boxes: !mo.tensor<?, f32>, %scores: !mo.tensor<?, f32>, %maxb: !mo.tensor<?, si64>, %iou_thres: !mo.tensor<?, si64>, %scores_thres: !mo.tensor<?, f32>) -> !mo.tensor<?, si64> {
  // expected-error @+1 {{'rmo.mo.non_maximum_suppression' op operand #3 must be rank 0 or unranked with float elements, but got '!mo.tensor<?, si64>'}}
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, f32>) -> !mo.tensor<?, si64>
  mo.output %0 : !mo.tensor<?, si64>
}

// -----

mo.graph @mms_bad_output(%boxes: !mo.tensor<?, f32>, %scores: !mo.tensor<?, f32>, %maxb: !mo.tensor<?, si64>, %iou_thres: !mo.tensor<?, f32>, %scores_thres: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.non_maximum_suppression' op result #0 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxb, %iou_thres, %scores_thres) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
