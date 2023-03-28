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

mo.graph @input_static_res_dyn_rank(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<?, f32> {
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{input and ranks ranks must be both dynamic, or both static and match}}
  %output = mo.reduce.min(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<?, f32>
  mo.output %output : !mo.tensor<?, f32>
}

// -----

mo.graph @input_dyn_res_static_rank(%input: !mo.tensor<?, f32>) -> !mo.tensor<[?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{input and ranks ranks must be both dynamic, or both static and match}}
  %output = mo.reduce.mul(%input, %axis) : (!mo.tensor<?, f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @input_res_rank_mismatch(%input: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // expected-error @below {{input and ranks ranks must be both dynamic, or both static and match}}
  %output = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %output : !mo.tensor<[?, ?], f32>
}
