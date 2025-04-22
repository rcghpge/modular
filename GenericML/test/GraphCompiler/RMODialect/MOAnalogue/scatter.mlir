// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @scatter_ok(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----
mo.graph @scatter_bad_axis(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %axis = mo.constant { value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
  // expected-error@+1 {{'rmo.mo.scatter' op operand #3 must be rank 0 or unranked with integer or index elements, but got '!mo.tensor<[], f32>'}}
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @scatter_bad_input_dtype(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f16>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error@+1 {{'rmo.mo.scatter' op failed to verify that all of {input, updates, result} have same dtype}}
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f16>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}

// -----

mo.graph @scatter_bad_result_dtype(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f16> {
  %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error@+1 {{'rmo.mo.scatter' op failed to verify that all of {input, updates, result} have same dtype}}
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>, !mo.tensor<[], si64>) -> !mo.tensor<?, f16>
  mo.output %1 : !mo.tensor<?, f16>
}

// -----

mo.graph @scatter_bad_indices_dtype(%input: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  %axis = mo.constant { value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error@+1 {{'rmo.mo.scatter' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %1 = rmo.mo.scatter(%input, %updates, %indices, %axis) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
