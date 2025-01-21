// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @split_dim_ok(%arg0: !mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}> {
    %out = mo.split_dim[1: i64](%arg0): (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}>
    mo.output %out: !mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}>
}

// -----

mo.graph @merge_dim_ok(%arg0: !mo.tensor<[4, 3, 3, 16, 1], f32>) -> !mo.tensor<[4, 9, 16, 1], f32, {layout = #mo.layout<NHWC>}> {
    %out = mo.merge_dim[1: i64](%arg0): (!mo.tensor<[4, 3, 3, 16, 1], f32>) -> !mo.tensor<[4, 9, 16, 1], f32, {layout = #mo.layout<NHWC>}>
    mo.output %out: !mo.tensor<[4, 9, 16, 1], f32, {layout = #mo.layout<NHWC>}>
}

// -----

mo.graph @add_singleton_dim_ok(%arg0: !mo.tensor<[4, 3, 16], f32>) -> !mo.tensor<[4, 3, 1, 16], f32, {layout = #mo.layout<NHWC>}> {
    %out = mo.add_singleton_dim[2: i64](%arg0): (!mo.tensor<[4, 3, 16], f32>) -> !mo.tensor<[4, 3, 1, 16], f32, {layout = #mo.layout<NHWC>}>
    mo.output %out: !mo.tensor<[4, 3, 1, 16], f32, {layout = #mo.layout<NHWC>}>
}
