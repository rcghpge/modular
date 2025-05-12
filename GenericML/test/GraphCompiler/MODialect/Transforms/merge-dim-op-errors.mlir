// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @nonaxis_dims_notequal(%arg0: !mo.tensor<[4, 3, 3, 25], f32>) -> !mo.tensor<[4, 9, 16], f32> {
    // expected-error@+1 {{mo.merge_dim' op Input and output dims do not match around axis}}
    %out = mo.merge_dim[1](%arg0): (!mo.tensor<[4, 3, 3, 25], f32>) -> !mo.tensor<[4, 9, 16], f32>
    mo.output %out: !mo.tensor<[4, 9, 16], f32>
}

// -----

mo.graph @axis_split_notequal(%arg0: !mo.tensor<[4, 3, 4, 16], f32>) -> !mo.tensor<[4, 9, 16], f32> {
    // expected-error@+1 {{'mo.merge_dim' op Merge dimensions do not multiply to final}}
    %out = mo.merge_dim[1](%arg0): (!mo.tensor<[4, 3, 4, 16], f32>) -> !mo.tensor<[4, 9, 16], f32>
    mo.output %out: !mo.tensor<[4, 9, 16], f32>
}

// -----

mo.graph @wrong_ranks(%arg0: !mo.tensor<[4, 3, 4], f32>) -> !mo.tensor<[4, 9, 16], f32> {
    // expected-error@+1 {{'mo.merge_dim' op Output rank must be one less than input}}
    %out = mo.merge_dim[1](%arg0): (!mo.tensor<[4, 3, 4], f32>) -> !mo.tensor<[4, 9, 16], f32>
    mo.output %out: !mo.tensor<[4, 9, 16], f32>
}

// -----

mo.graph @wrong_dtype(%arg0: !mo.tensor<[4, 3, 3, 16], f32>) -> !mo.tensor<[4, 9, 16], f64> {
    // expected-error@+1 {{'mo.merge_dim' op More than shape / layout has changed between input and output.}}
    %out = mo.merge_dim[1](%arg0): (!mo.tensor<[4, 3, 3, 16], f32>) -> !mo.tensor<[4, 9, 16], f64>
    mo.output %out: !mo.tensor<[4, 9, 16], f64>
}
