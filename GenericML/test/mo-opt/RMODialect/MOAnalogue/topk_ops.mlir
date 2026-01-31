// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @topk_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, si64>, %axis: !mo.tensor<Sh2, si64>, %sorted: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>) {
  %values, %indices = rmo.mo.top_k<() -> Sh4: !mosh.ape, Sh5: !mosh.ape>(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>
      ) -> (
        !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
      )
  mo.output %values, %indices : !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
}

// -----

mo.graph @topk_bad_k<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, f32>, %axis: !mo.tensor<Sh2, si64>, %sorted: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op operand #1 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<Sh1, f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>
      ) -> (
        !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
      )
  mo.output %values, %indices : !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
}

// -----

mo.graph @topk_bad_axis<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, si64>, %axis: !mo.tensor<Sh2, f32>, %sorted: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op operand #2 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, f32>, !mo.tensor<Sh3, bool>
      ) -> (
        !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
      )
  mo.output %values, %indices : !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
}

// -----

mo.graph @topk_bad_sorted<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, si64>, %axis: !mo.tensor<Sh2, si64>, %sorted: !mo.tensor<Sh3, f32>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op operand #3 must be rank 0 or unranked with boolean element type, but got '!mo.tensor<Sh3, f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, f32>
      ) -> (
        !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
      )
  mo.output %values, %indices : !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
}

// -----

mo.graph @topk_bad_values<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, si64>, %axis: !mo.tensor<Sh2, si64>, %sorted: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f16>, !mo.tensor<Sh5, si64>) {
  // expected-error @+1 {{'rmo.mo.top_k' op failed to verify that all of {input, values} have same dtype}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>
      ) -> (
        !mo.tensor<Sh4, f16>, !mo.tensor<Sh5, si64>
      )
  mo.output %values, %indices : !mo.tensor<Sh4, f16>, !mo.tensor<Sh5, si64>
}

// -----

mo.graph @topk_bad_indices<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, si64>, %axis: !mo.tensor<Sh2, si64>, %sorted: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, f32>) {
  // expected-error @+1 {{'rmo.mo.top_k' op result #1 must be signed 64 integer element type, but got '!mo.tensor<Sh5, f32>'}}
  %values, %indices = rmo.mo.top_k(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>
      ) -> (
        !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, f32>
      )
  mo.output %values, %indices : !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, f32>
}

// -----

mo.graph @bottomk_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%input: !mo.tensor<Sh0, f32>, %k: !mo.tensor<Sh1, si64>, %axis: !mo.tensor<Sh2, si64>, %sorted: !mo.tensor<Sh3, bool>) -> (!mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>) {
  %values, %indices = rmo.mo.bottom_k<() -> Sh4: !mosh.ape, Sh5: !mosh.ape>(%input, %k, %axis, %sorted) : (
        !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, bool>
      ) -> (
        !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
      )
  mo.output %values, %indices : !mo.tensor<Sh4, f32>, !mo.tensor<Sh5, si64>
}
