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
