// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @topk_ok<D0, D1 -> V0, V1, I0, I1>(%input: !mo.tensor<[D0, D1], f32>, %k: !mo.tensor<[], si64>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<[V0, V1], f32>, !mo.tensor<[I0, I1], si64>) {
  %values, %indices = rmo.mo.top_k<() -> X0, X1, Y0, Y1>(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[X0, X1], f32>, !mo.tensor<[Y0, Y1], si64>
      )
  mo.output<X0, X1, Y0, Y1> %values, %indices : !mo.tensor<[X0, X1], f32>, !mo.tensor<[Y0, Y1], si64>
}

// -----

mo.graph @topk_bad_k<D0, D1>(%input: !mo.tensor<[D0, D1], f32>, %k: !mo.tensor<[], f32>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op operand #1 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<[], f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si64>
      )
  mo.output %values, %indices : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si64>
}

// -----

mo.graph @topk_bad_axis<D0, D1>(%input: !mo.tensor<[D0, D1], f32>, %k: !mo.tensor<[], si64>, %axis: !mo.tensor<[], f32>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<[], f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si64>
      )
  mo.output %values, %indices : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si64>
}

// -----

mo.graph @topk_bad_sorted<D0, D1>(%input: !mo.tensor<[D0, D1], f32>, %k: !mo.tensor<[], si64>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], f32>) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op operand #3 must be rank 0 or unranked with boolean element type, but got '!mo.tensor<[], f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>
      ) -> (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si64>
      )
  mo.output %values, %indices : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], si64>
}

// -----

mo.graph @topk_bad_values<D0, D1>(%input: !mo.tensor<[D0, D1], f32>, %k: !mo.tensor<[], si64>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<[D0, D1], f16>, !mo.tensor<[D0, D1], si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op failed to verify that all of {input, values} have same dtype}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[D0, D1], f16>, !mo.tensor<[D0, D1], si64>
      )
  mo.output %values, %indices : !mo.tensor<[D0, D1], f16>, !mo.tensor<[D0, D1], si64>
}

// -----

mo.graph @topk_bad_indices<D0, D1>(%input: !mo.tensor<[D0, D1], f32>, %k: !mo.tensor<[], si64>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>) {
  // expected-error @+1 {{'rmo.mo.top_k' op result #1 must be signed 64 integer element type, but got '!mo.tensor<[D0, D1], f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>
      )
  mo.output %values, %indices : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @bottomk_ok<D0, D1 -> V0, V1, I0, I1>(%input: !mo.tensor<[D0, D1], f32>, %k: !mo.tensor<[], si64>, %axis: !mo.tensor<[], si64>, %sorted: !mo.tensor<[], bool>) -> (!mo.tensor<[V0, V1], f32>, !mo.tensor<[I0, I1], si64>) {
  %values, %indices = rmo.mo.bottom_k<() -> X0, X1, Y0, Y1>(%input, %k, %axis, %sorted) : (
        !mo.tensor<[D0, D1], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[X0, X1], f32>, !mo.tensor<[Y0, Y1], si64>
      )
  mo.output<X0, X1, Y0, Y1> %values, %indices : !mo.tensor<[X0, X1], f32>, !mo.tensor<[Y0, Y1], si64>
}
