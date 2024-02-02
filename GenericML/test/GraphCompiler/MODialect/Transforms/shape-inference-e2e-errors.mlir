// RUN: mo-opt --shape-inference --verify-diagnostics

// TODO make first input dimension dynamic dynamic, which would require folding
// `eq(mul(D0, 10), mul(D0, 8))` in order to emit compile time error.
mo.graph @reshape_number_of_elems(
  %input: !mo.tensor<[10, 10], f32>
) -> !mo.tensor<?, f32> {

  %axis = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %zero = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[1], si32>
  %one = mo.constant {value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[1], si32>

  %input_shape = mo.shape_of(%input) : (!mo.tensor<[10, 10], f32>) -> !mo.tensor<[2], si32>
  %d0 = mo.slice(%input_shape, %zero, %one, %one) : (
    !mo.tensor<[2], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>
  ) -> !mo.tensor<[?], si32>

  %dims = mo.constant {value = #M.dense_array<2, 4> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %target_shape = mo.concat[%axis : !mo.tensor<[], si32>](%d0, %dims) : (
    !mo.tensor<[?], si32>, !mo.tensor<[2], si32>
  ) -> !mo.tensor<[?], si32>

  // expected-error@below {{[reshape] input and output number of elements must match}}
  %result = mo.reshape(%input, %target_shape) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[?], si32>
  ) -> !mo.tensor<?, f32>

  mo.output %result : !mo.tensor<?, f32>
}
