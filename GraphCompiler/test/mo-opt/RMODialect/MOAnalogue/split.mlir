// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @split_ok<D0, D1 -> D2, D3, D4, D5>(%arg0: !mo.tensor<[D0, D1], f32>, %axis: !mo.tensor<[], si64>, %sizes: !mo.tensor<[2], si64>) -> (!mo.tensor<[D2, D3], f32>, !mo.tensor<[D4, D5], f32>) {
  %0:2 = rmo.mo.split<() -> D6, D7, D8, D9>[%axis: !mo.tensor<[], si64>](%arg0, %sizes) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[D6, D7], f32>, !mo.tensor<[D8, D9], f32>)
  mo.output<D6, D7, D8, D9> %0#0, %0#1 : !mo.tensor<[D6, D7], f32>, !mo.tensor<[D8, D9], f32>
}

// -----

mo.graph @split_bad_axis<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %axis: !mo.tensor<[], f32>, %sizes: !mo.tensor<[2], si64>) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) {
  // expected-error @+1 {{'rmo.mo.split' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<[], f32>'}}
  %0:2 = rmo.mo.split[%axis: !mo.tensor<[], f32>](%arg0, %sizes) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>)
  mo.output %0#0, %0#1 : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @split_bad_sizes<D0, D1>(%arg0: !mo.tensor<[D0, D1], f32>, %axis: !mo.tensor<[], si64>, %sizes: !mo.tensor<[2], f32>) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) {
  // expected-error @+1 {{'rmo.mo.split' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[2], f32>'}}
  %0:2 = rmo.mo.split[%axis: !mo.tensor<[], si64>](%arg0, %sizes) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2], f32>) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>)
  mo.output %0#0, %0#1 : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>
}
