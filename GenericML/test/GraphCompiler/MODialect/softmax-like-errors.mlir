// RUN: mo-opt %s --split-input-file --verify-diagnostics

// CHECK: mo.graph @softmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 4], f32> {
mo.graph @softmax(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 4], f32>) {
  // expected-error@+1 {{op requires the same type for all operands and results}}
  %0 = mo.softmax(%arg0) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 4], f32>
  mo.output %0 : !mo.tensor<[2, 4], f32>
}

// -----

// CHECK: mo.graph @softmax_dtype(%[[ARG0:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f64> {
mo.graph @softmax_dtype(%arg0: !mo.tensor<?, f32>) -> (!mo.tensor<?, f64>) {
  // expected-error@+1 {{op requires the same type for all operands and results}}
  %0 = mo.softmax(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f64>
  mo.output %0 : !mo.tensor<?, f64>
}

// -----

// CHECK: mo.graph @logsoftmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 4], f32> {
mo.graph @logsoftmax(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 4], f32>) {
  // expected-error@+1 {{op requires the same type for all operands and results}}
  %0 = mo.logsoftmax(%arg0) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 4], f32>
  mo.output %0 : !mo.tensor<[2, 4], f32>
}

// -----

// CHECK: mo.graph @logsoftmax_dtype(%[[ARG0:.*]]: !mo.tensor<?, f32>) -> !mo.tensor<?, f64> {
mo.graph @logsoftmax_dtype(%arg0: !mo.tensor<?, f32>) -> (!mo.tensor<?, f64>) {
  // expected-error@+1 {{op requires the same type for all operands and results}}
  %0 = mo.logsoftmax(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<?, f64>
  mo.output %0 : !mo.tensor<?, f64>
}
