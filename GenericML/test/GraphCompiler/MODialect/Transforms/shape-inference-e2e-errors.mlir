// RUN: mo-opt %s --shape-inference --split-input-file --verify-diagnostics

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

// -----

// This was adopted from an incorrect lowering of TF slice, which went silent
// without error. This is a minimized version of the repro test.
mo.graph @reshape_bad_annotated_rank(%arg0: !mo.tensor<[?, 320], f32>) -> !mo.tensor<[?, ?], f32> {
  %0 = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[1], si32>
  %1 = mo.constant {value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
  %2 = mo.constant {value = #M.dense_array<1> : !M.array<1xsi32>} : !mo.tensor<[], si32>
  %3 = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %4 = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %5 = mo.constant {value = #M.dense_array<1, 1> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  %6 = mo.constant {value = #M.dense_array<0, 0> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  %7 = mo.constant {value = #M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>} : !mo.tensor<[2], si64>
  %8 = mo.slice(%arg0, %6, %7, %5) : (!mo.tensor<[?, 320], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  %9 = mo.shape_of(%8) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[2], si64>
  %10 = mo.shape_of(%9) : (!mo.tensor<[2], si64>) -> !mo.tensor<[1], si32>
  %11 = mo.rebind(%0) : !mo.tensor<[1], si32> -> !mo.tensor<[?], si32>
  %12 = mo.static.reshape(%10) : !mo.tensor<[1], si32> -> !mo.tensor<[], si32>
  %13 = mo.range(%2, %12, %1) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si32>
  %14 = mo.gather(%9, %11, %3) : (!mo.tensor<[2], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %15 = mo.gather(%9, %13, %3) : (!mo.tensor<[2], si64>, !mo.tensor<[?], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], si64>
  %16 = mo.concat[%3 : !mo.tensor<[], si32>] (%14, %4, %15) : (!mo.tensor<[?], si64>, !mo.tensor<[1], si64>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], si64>
  // expected-error@below {{shape function returned shape of rank 3 but the type annotation has shape of rank 2}}
  %17 = mo.reshape(%8, %16) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %17 : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @reshape_parametric_shape_inferred_neg_one_empty<D0 -> D2, D3, D4>(
  %input: !mo.tensor<[D0, 0], f32>
) -> !mo.tensor<[D2, D3, D4], f32> {

  %param_shape = mosh.param.to_value: !mosh.ape = <[-1, 4, 0]>
  %target_shape = mo.shape.to_tensor(%param_shape) -> !mo.tensor<[3], si32>

  // expected-error@below {{[reshape] cannot infer dimension when a specified output dimension is 0}}
  %res = mo.reshape<() -> D5, D6, D7>(%input, %target_shape) : (
    !mo.tensor<[D0, 0], f32>, !mo.tensor<[3], si32>
  ) -> !mo.tensor<[D5, D6, D7], f32>

  mo.output<D5, D6, D7> %res : !mo.tensor<[D5, D6, D7], f32>
}

// -----

mo.graph @concat_axis_out_of_bound(
  %input1: !mo.tensor<[?, 1], f32>,
  %input2: !mo.tensor<[?, 2], f32>
) -> !mo.tensor<[?, ?], f32> {
  %axis = mo.constant {value = #M.dense_array<2> : tensor<si64>} : !mo.tensor<[], si64>
  // expected-error@below {{[concat] normalized axis must be within range [0, input_rank)}}
  %res = mo.concat[%axis : !mo.tensor<[], si64>](%input1, %input2) : (
    !mo.tensor<[?, 1], f32>, !mo.tensor<[?, 2], f32>
  ) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}
