// RUN: mo-opt %s --split-input-file --verify-diagnostics

// CHECK: mo.graph @softmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[2, 4], f32> {
mo.graph @softmax(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<[2, 4], f32>) {
  // expected-error@+1 {{op failed to verify that all of {input, result} have same shapeAttr}}
  %0 = mo.softmax(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 4], f32>
  mo.output %0 : !mo.tensor<[2, 4], f32>
}

// -----

// CHECK: mo.graph @softmax_dtype<[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f64> {
mo.graph @softmax_dtype<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<Sh1, f64>) {
  // expected-error@+1 {{op failed to verify that all of {input, result} have same dtype}}
  %0 = mo.softmax(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f64>
  mo.output %0 : !mo.tensor<Sh1, f64>
}

// -----

// CHECK: mo.graph @logsoftmax(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[2, 4], f32> {
mo.graph @logsoftmax(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<[2, 4], f32>) {
  // expected-error@+1 {{op failed to verify that all of {input, result} have same shapeAttr}}
  %0 = mo.logsoftmax(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 4], f32>
  mo.output %0 : !mo.tensor<[2, 4], f32>
}

// -----

// CHECK: mo.graph @logsoftmax_dtype<[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH1]], f64> {
mo.graph @logsoftmax_dtype<Sh0: !mosh.ape, Sh1: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[], si64>) -> (!mo.tensor<Sh1, f64>) {
  // expected-error@+1 {{op failed to verify that all of {input, result} have same dtype}}
  %0 = mo.logsoftmax(%arg0, %arg1) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si64>) -> !mo.tensor<Sh1, f64>
  mo.output %0 : !mo.tensor<Sh1, f64>
}
