// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @constant_illegal_attr() -> !mo.tensor<[4], f32> {
  // expected-error@+1 {{'mo.constant' op attribute 'value' failed to satisfy constraint}}
  %0 = mo.constant {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}
