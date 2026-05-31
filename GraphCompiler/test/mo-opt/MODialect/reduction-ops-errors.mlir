// RUN: mo-opt %s --split-input-file --verify-diagnostics

// COM: These tests are specific to ops, but verify properties that apply to all reduction ops.

mo.graph @input_res_rank_mismatch(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?], f32> {
  // expected-error @below {{input and output must either both have dynamic ranks or both have the same size static ranks}}
  %output = mo.reduce.add(%input) {axis = 2 : index} : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @axis_too_big(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  // expected-error @below {{'mo.reduce.add' op axis must be between [-3, 2] found 3}}
  %output = mo.reduce.add(%input) {axis = 3 : index} : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @axis_too_small(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  // expected-error @below {{'mo.reduce.add' op axis must be between [-3, 2] found -4}}
  %output = mo.reduce.add(%input) {axis = -4 : index} : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @mixed_dtypes_float_to_int(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], si32> {
  // expected-error @below {{'mo.reduce.add' op failed to verify that all of {input, result} have same dtype}}
  %output = mo.reduce.add(%input) {axis = 0 : index} : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], si32>
  mo.output %output : !mo.tensor<[?, ?, ?], si32>
}

// -----

mo.graph @mixed_dtypes_float_to_float(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f16> {
  // expected-error @below {{'mo.reduce.mean' op failed to verify that all of {input, result} have same dtype}}
  %output = mo.reduce.mean(%input) {axis = 0 : index} : (!mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, ?], f16>
  mo.output %output : !mo.tensor<[?, ?, ?], f16>
}
