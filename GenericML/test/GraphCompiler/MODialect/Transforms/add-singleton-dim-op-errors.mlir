// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @nonaxis_dims_notequal(%arg0: !mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 1, 9, 25], f32> {
    // expected-error@+1 {{'mo.add_singleton_dim' op Input and output dims do not match around axis.}}
    %out = mo.add_singleton_dim[1](%arg0): (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 1, 9, 25], f32>
    mo.output %out: !mo.tensor<[4, 1, 9, 25], f32>
}

// -----

mo.graph @axis_not_one(%arg0: !mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 9, 1, 16], f32> {
    // expected-error@+1 {{'mo.add_singleton_dim' op Output dim does not have 1 for `axis`}}
    %out = mo.add_singleton_dim[1](%arg0): (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 9, 1, 16], f32>
    mo.output %out: !mo.tensor<[4, 9, 1, 16], f32>
}

// -----

mo.graph @wrong_ranks(%arg0: !mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 9, 16], f32> {
    // expected-error@+1 {{'mo.add_singleton_dim' op Output rank must be one greater than input}}
    %out = mo.add_singleton_dim[1](%arg0): (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 9, 16], f32>
    mo.output %out: !mo.tensor<[4, 9, 16], f32>
}

// -----

mo.graph @wrong_dtype(%arg0: !mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 1, 9, 16], f64> {
    // expected-error@+1 {{'mo.add_singleton_dim' op More than shape / layout has changed between input and output.}}
    %out = mo.add_singleton_dim[1](%arg0): (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 1, 9, 16], f64>
    mo.output %out: !mo.tensor<[4, 1, 9, 16], f64>
}
