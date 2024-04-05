// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @slice_ok(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @slice_bad_start(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<?, f32>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.slice' op operand #1 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @slice_bad_step(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, f32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.slice' op operand #2 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @slice_bad_stop(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.slice' op operand #3 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @slice_bad_result(%arg0: !mo.tensor<?, f32>, %start: !mo.tensor<?, si64>, %stop: !mo.tensor<?, si32>, %step: !mo.tensor<?, si64>) -> !mo.tensor<?, f16> {
  // expected-error @+1 {{'rmo.mo.slice' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f16>
  mo.output %0 : !mo.tensor<?, f16>
}
