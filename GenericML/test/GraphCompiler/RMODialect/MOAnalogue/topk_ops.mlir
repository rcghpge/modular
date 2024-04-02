// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @topk_ok(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<?, f32>, !mo.tensor<?, si64>) {
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<?, si64>
}

// -----

mo.graph @topk_bad_k(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, f32>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<?, f32>, !mo.tensor<?, si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op operand #1 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<?, si64>
}

// -----

mo.graph @topk_bad_axis(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, f32>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<?, f32>, !mo.tensor<?, si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, f32>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<?, si64>
}

// -----

mo.graph @topk_bad_sorted(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>, !mo.tensor<?, si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op operand #3 must be rank 0 or unranked with boolean element type, but got '!mo.tensor<?, f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, f32>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<?, si64>
}

// -----

mo.graph @topk_bad_values(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<?, f16>, !mo.tensor<?, si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op failed to verify that all of {input, values} have same dtype}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<?, f16>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f16>, !mo.tensor<?, si64>
}

// -----

mo.graph @topk_bad_indices(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<?, f32>, !mo.tensor<?, f32>) {
  // expected-error @+1 {{'rmo.mo.top_k' op result #1 must be signed 64 integer element type, but got '!mo.tensor<?, f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<?, f32>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<?, f32>
}

// -----

mo.graph @bottomk_ok(%input: !mo.tensor<?, f32>, %k: !mo.tensor<?, si64>, %axis: !mo.tensor<?, si64>, %sorted: !mo.tensor<?, bool>) -> (!mo.tensor<?, f32>, !mo.tensor<?, si64>) {
  %values, %indices = rmo.mo.bottom_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<?, si64>, !mo.tensor<?, bool>
      ) -> (
        !mo.tensor<?, f32>, !mo.tensor<?, si64>
      )
  mo.output %values, %indices : !mo.tensor<?, f32>, !mo.tensor<?, si64>
}
