// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @concat_ok<D0, D1, D2 -> D3, D4>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D1], f32>, %axis: !mo.tensor<[], si64>) -> !mo.tensor<[D3, D4], f32> {
  %0 = rmo.mo.concat<() -> D5, D6>[%axis: !mo.tensor<[], si64>](%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D1], f32>) -> (!mo.tensor<[D5, D6], f32>)
  mo.output<D5, D6> %0: !mo.tensor<[D5, D6], f32>
}

// -----

mo.graph @concat_bax_axis<D0, D1, D2>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D1], f32>, %axis: !mo.tensor<[], f32>) -> !mo.tensor<[D0, D1], f32> {
  // expected-error @+1 {{'rmo.mo.concat' op operand #0 must be tensor with integer or index elements, but got '!mo.tensor<[], f32>'}}
  %0 = rmo.mo.concat[%axis: !mo.tensor<[], f32>](%arg0, %arg1) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D1], f32>) -> (!mo.tensor<[D0, D1], f32>)
  mo.output %0: !mo.tensor<[D0, D1], f32>
}
