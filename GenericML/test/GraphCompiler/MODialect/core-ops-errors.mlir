// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @tensor_incorrect_syntax(%arg0: !mo.tensor<[], f32>) {
  // expected-error@+2 {{expected string}}
  // expected-error@+1 {{couldn't parse shape parameter: expected `?`, shape reference, or literal shape reference (i.e. `[...]`)}}
  %0 = mo.rebind(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<*, f32>
  mo.output
}

// -----

mo.graph @constant_illegal_attr() -> !mo.tensor<[4], f32> {
  // expected-error@+1 {{'mo.constant' op attribute 'value' failed to satisfy constraint}}
  %0 = mo.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

mo.graph @rebind_different_static_ranks<D0, D1, D2>(
  %arg1: !mo.tensor<[D0], f32>,
  %arg2: !mo.tensor<[D1, D2], f32>
) -> !mo.tensor<[D0], f32> {
  // expected-error @below {{input and output static ranks must match}}
  %0 = mo.rebind(%arg1) : !mo.tensor<[D0], f32> -> !mo.tensor<[D1, D2], f32>
  mo.output %arg1: !mo.tensor<[D0], f32>
}

// -----

mo.graph @rebind_different_static_dimensions<D0, D1>(
  %arg1: !mo.tensor<[D0, 42, D1], f32>
) -> !mo.tensor<[D0, 42, D1], f32> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[D0, 42, D1], f32> -> !mo.tensor<[D1, 42, D0], f32>
  // expected-error @below {{statically known dimensions must match}}
  %1 = mo.rebind(%arg1) : !mo.tensor<[D0, 42, D1], f32> -> !mo.tensor<[D1, 24, D0], f32>
  mo.output %arg1: !mo.tensor<[D0, 42, D1], f32>
}

// -----

mo.graph @rebind_same_in_param_bind_to_different_ints<D0>(
  %arg1: !mo.tensor<[D0, D0], f32>
) -> !mo.tensor<[D0, D0], f32> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[D0, D0], f32> -> !mo.tensor<[42, 42], f32>
  // expected-error @below {{one dimension parameter cannot bind to different static dimensions}}
  %1 = mo.rebind(%arg1) : !mo.tensor<[D0, D0], f32> -> !mo.tensor<[42, 24], f32>
  mo.output %arg1: !mo.tensor<[D0, D0], f32>
}

// -----

mo.graph @rebind_same_out_param_bind_to_different_ints<D0>(
  %arg1: !mo.tensor<[42, 42], f32>,
  %arg2: !mo.tensor<[42, 24], f32>,
  %arg3: !mo.tensor<[D0], f32>
) -> !mo.tensor<[D0], f32> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[42, 42], f32> -> !mo.tensor<[D0, D0], f32>
  // expected-error @below {{one dimension parameter cannot bind to different static dimensions}}
  %1 = mo.rebind(%arg2) : !mo.tensor<[42, 24], f32> -> !mo.tensor<[D0, D0], f32>
  mo.output %arg3: !mo.tensor<[D0], f32>
}
