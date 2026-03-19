// RUN: mo-opt %s --split-input-file --verify-diagnostics

// COM: These tests are specific to ops, but verify properties that apply to all reduction ops.

mo.graph @non_scalar_axis(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32> {
  %axis = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2, 1], si64>
  // expected-error @below {{'mo.mean' op operand #1 must be ranked scalar index tensor, but got '!mo.tensor<[2, 1], si64>'}}
  %output = mo.mean(%input, %axis) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 1], si64>) -> !mo.tensor<[2, 1], f32>
  mo.output %output : !mo.tensor<[2, 1], f32>
}

// -----

mo.graph @non_int_axis(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32> {
  %axis = mo.constant {value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
  // expected-error @below {{'mo.reduce.max' op operand #1 must be ranked scalar index tensor, but got '!mo.tensor<[], f32>'}}
  %output = mo.reduce.max(%input, %axis) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[2, 1], f32>
  mo.output %output : !mo.tensor<[2, 1], f32>
}

// -----

mo.graph @input_static_res_dyn_rank<D0, D1, Sh0: !mosh.ape>(%input: !mo.tensor<[D0, D1, 5], f32>) -> !mo.tensor<Sh0, f32> {
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{input and output must either both have dynamic ranks or both have the same size static ranks}}
  %output = mo.reduce.min(%input, %axis) : (!mo.tensor<[D0, D1, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh0, f32>
  mo.output %output : !mo.tensor<Sh0, f32>
}

// -----

mo.graph @input_dyn_res_static_rank<Sh0: !mosh.ape, D0, D1>(%input: !mo.tensor<Sh0, f32>) -> !mo.tensor<[D0, D1], f32> {
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{input and output must either both have dynamic ranks or both have the same size static ranks}}
  %output = mo.reduce.mul(%input, %axis) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<[D0, D1], f32>
  mo.output %output : !mo.tensor<[D0, D1], f32>
}

// -----

mo.graph @input_res_rank_mismatch(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{input and output must either both have dynamic ranks or both have the same size static ranks}}
  %output = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @axis_too_big(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<3> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{'mo.reduce.add' op axis must be between [-3, 2] found 3}}
  %output = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @axis_too_small(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<-4> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{'mo.reduce.add' op axis must be between [-3, 2] found -4}}
  %output = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @mixed_dtypes_float_to_int(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], si32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{'mo.reduce.add' op failed to verify that all of {input, result} have same dtype}}
  %output = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, ?], si32>
  mo.output %output : !mo.tensor<[?, ?, ?], si32>
}

// -----

mo.graph @mixed_dtypes_float_to_float(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f16> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{'mo.mean' op failed to verify that all of {input, result} have same dtype}}
  %output = mo.mean(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, ?], f16>
  mo.output %output : !mo.tensor<[?, ?, ?], f16>
}
