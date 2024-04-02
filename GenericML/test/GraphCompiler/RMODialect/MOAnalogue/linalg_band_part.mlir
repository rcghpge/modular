// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @band_part_ok(%arg0: !mo.tensor<?, f32>, %num_lower: !mo.tensor<?, si64>, %num_upper: !mo.tensor<?, si64>, %exclude: !mo.tensor<?, bool>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @band_part_bad_num_lower(%arg0: !mo.tensor<?, f32>, %num_lower: !mo.tensor<?, f32>, %num_upper: !mo.tensor<?, si64>, %exclude: !mo.tensor<?, bool>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.linalg.band_part' op operand #1 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, bool>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @band_part_bad_num_upper(%arg0: !mo.tensor<?, f32>, %num_lower: !mo.tensor<?, si64>, %num_upper: !mo.tensor<?, f32>, %exclude: !mo.tensor<?, bool>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.linalg.band_part' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, bool>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @band_part_bad_exclude(%arg0: !mo.tensor<?, f32>, %num_lower: !mo.tensor<?, si64>, %num_upper: !mo.tensor<?, si64>, %exclude: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.linalg.band_part' op operand #3 must be rank 0 or unranked with boolean element type, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @band_part_bad_result(%arg0: !mo.tensor<?, f32>, %num_lower: !mo.tensor<?, si64>, %num_upper: !mo.tensor<?, si64>, %exclude: !mo.tensor<?, bool>) -> !mo.tensor<?, f16> {
  // expected-error @+1 {{'rmo.mo.linalg.band_part' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.linalg.band_part(%arg0, %num_lower, %num_upper, %exclude) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>) -> !mo.tensor<?, f16>
  mo.output %0 : !mo.tensor<?, f16>
}
