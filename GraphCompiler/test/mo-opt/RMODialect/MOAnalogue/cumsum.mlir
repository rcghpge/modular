// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @cumsum_ok<D0, D1>(%input: !mo.tensor<[D0, D1], f32>, %axis: !mo.tensor<[], si64>) -> (!mo.tensor<[D0, D1], f32>) {
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @cumsum_bad_axis<D0, D1>(%input: !mo.tensor<[D0, D1], f32>, %axis: !mo.tensor<[], f32>) -> (!mo.tensor<[D0, D1], f32>) {
  // expected-error @+1 {{'rmo.mo.cumsum' op operand #1 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<[], f32>'}}
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[D0, D1], f32>
  mo.output %0 : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @cumsum_bad_output<D0, D1>(%input: !mo.tensor<[D0, D1], f32>, %axis: !mo.tensor<[], si64>) -> (!mo.tensor<[D0, D1], f16>) {
  // expected-error @+1 {{'rmo.mo.cumsum' op failed to verify that all of {input, result} have same dtype}}
  %0 = rmo.mo.cumsum(%input, %axis) {reverse = 0 : index, exclusive = 0 : index} : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[D0, D1], f16>
  mo.output %0 : !mo.tensor<[D0, D1], f16>
}
