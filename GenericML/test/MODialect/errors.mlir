// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @constant_illegal_attr() -> !mo.tensor<4xf32> {
  // expected-error@+1 {{'mo.constant' op attribute 'value' failed to satisfy constraint}}
  %0 = mo.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : !mo.tensor<4xf32>
  mo.output %0 : !mo.tensor<4xf32>
}

// -----

mo.graph @concat_no_args(%arg0: !mo.tensor<7xf32>) -> !mo.tensor<7xf32> {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<si64>
  // expected-error@+1 {{concat must have at least 2 input tensors}}
  %0 = mo.concat[%ax: !mo.tensor<si64>](%arg0) : (!mo.tensor<7xf32>) -> !mo.tensor<7xf32>
  mo.output %0 : !mo.tensor<7xf32>
}

// -----

mo.graph @concat_non_scalar_axis(%arg0: !mo.tensor<7xf32>, %arg1: !mo.tensor<2xf32>) -> !mo.tensor<9xf32> {
  %ax = mo.constant {value = #M.dense_array<0, 2> : tensor<2xsi64>} : !mo.tensor<2xsi64>
  // expected-error@+1 {{concat axis must be a scalar tensor}}
  %0 = mo.concat[%ax: !mo.tensor<2xsi64>](%arg0, %arg1) : (!mo.tensor<7xf32>, !mo.tensor<2xf32>) -> !mo.tensor<9xf32>
  mo.output %0 : !mo.tensor<9xf32>
}

// -----

mo.graph @concat_non_matching_input_ranks(%arg0: !mo.tensor<7xf32>, %arg1: !mo.tensor<2x3xf32>) -> !mo.tensor<9xf32> {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<si64>
  // expected-error@+1 {{concat inputs with known rank must match}}
  %0 = mo.concat[%ax: !mo.tensor<si64>](%arg0, %arg1) : (!mo.tensor<7xf32>, !mo.tensor<2x3xf32>) -> !mo.tensor<9xf32>
  mo.output %0 : !mo.tensor<9xf32>
}

// -----

mo.graph @concat_non_matching_output_rank(%arg0: !mo.tensor<2x2xf32>, %arg1: !mo.tensor<2x3xf32>) -> !mo.tensor<9xf32> {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<si64>
  // expected-error@+1 {{concat output with known rank must match inputs}}
  %0 = mo.concat[%ax: !mo.tensor<si64>](%arg0, %arg1) : (!mo.tensor<2x2xf32>, !mo.tensor<2x3xf32>) -> !mo.tensor<9xf32>
  mo.output %0 : !mo.tensor<9xf32>
}

// -----

mo.graph @gather_scalar_param(%params: !mo.tensor<f32>, %indices: !mo.tensor<2xsi64>) -> !mo.tensor<2x2xf32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<si64>
  // expected-error@+1 {{'mo.gather' op operand #0 must be ranked non-scalar tensor, but got '!mo.tensor<f32>'}}
  %0 = mo.gather(%params, %indices, %axis) : (!mo.tensor<f32>, !mo.tensor<2xsi64>, !mo.tensor<si64>) -> !mo.tensor<2x2xf32>
  mo.output %0 : !mo.tensor<2x2xf32>
}

// -----

mo.graph @gather_float_indices(%params: !mo.tensor<2x2xf32>, %indices: !mo.tensor<2xf32>) -> !mo.tensor<2x2xf32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<si64>
  // expected-error@+1 {{'mo.gather' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<2xf32>'}}
  %0 = mo.gather(%params, %indices, %axis) : (!mo.tensor<2x2xf32>, !mo.tensor<2xf32>, !mo.tensor<si64>) -> !mo.tensor<2x2xf32>
  mo.output %0 : !mo.tensor<2x2xf32>
}

// -----

mo.graph @gather_non_scalar_axis(%params: !mo.tensor<2x2xf32>, %indices: !mo.tensor<2xsi64>) -> !mo.tensor<2x2xf32> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<2xsi64>
  // expected-error@+1 {{'mo.gather' op operand #2 must be ranked scalar tensor, but got '!mo.tensor<2xsi64>'}}
  %0 = mo.gather(%params, %indices, %axis) : (!mo.tensor<2x2xf32>, !mo.tensor<2xsi64>, !mo.tensor<2xsi64>) -> !mo.tensor<2x2xf32>
  mo.output %0 : !mo.tensor<2x2xf32>
}
