// RUN: mo-opt %s --split-input-file

mo.graph @split_dim_add_layout_ok(%arg0: !mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}> {
    %out = mo.split_dim[1](%arg0): (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}>
    mo.output %out: !mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}>
}

// -----

mo.graph @split_dim_remove_layout_ok(%arg0: !mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}>) -> !mo.tensor<[4, 3, 3, 4, 4], f32> {
    %out = mo.split_dim[3](%arg0): (!mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}>) -> !mo.tensor<[4, 3, 3, 4, 4], f32>
    mo.output %out: !mo.tensor<[4, 3, 3, 4, 4], f32>
}

// -----

mo.graph @merge_dim_add_layout_ok(%arg0: !mo.tensor<[4, 3, 3, 16, 1], f32>) -> !mo.tensor<[4, 9, 16, 1], f32, {layout = #mo.layout<NHWC>}> {
    %out = mo.merge_dim[1](%arg0): (!mo.tensor<[4, 3, 3, 16, 1], f32>) -> !mo.tensor<[4, 9, 16, 1], f32, {layout = #mo.layout<NHWC>}>
    mo.output %out: !mo.tensor<[4, 9, 16, 1], f32, {layout = #mo.layout<NHWC>}>
}

// -----

mo.graph @merge_dim_remove_layout_ok(%arg0: !mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}>) -> !mo.tensor<[4, 9, 16], f32> {
    %out = mo.merge_dim[1](%arg0): (!mo.tensor<[4, 3, 3, 16], f32, {layout = #mo.layout<NHWC>}>) -> !mo.tensor<[4, 9, 16], f32>
    mo.output %out: !mo.tensor<[4, 9, 16], f32>
}

// -----

mo.graph @add_singleton_dim_add_layout_ok(%arg0: !mo.tensor<[4, 3, 16], f32>) -> !mo.tensor<[4, 3, 1, 16], f32, {layout = #mo.layout<NHWC>}> {
    %out = mo.add_singleton_dim[2](%arg0): (!mo.tensor<[4, 3, 16], f32>) -> !mo.tensor<[4, 3, 1, 16], f32, {layout = #mo.layout<NHWC>}>
    mo.output %out: !mo.tensor<[4, 3, 1, 16], f32, {layout = #mo.layout<NHWC>}>
}

// -----

mo.graph @add_singleton_dim_remove_layout_ok(%arg0: !mo.tensor<[4, 3, 16, 1], f32, {layout = #mo.layout<NHWC>}>) -> !mo.tensor<[4, 3, 1, 16, 1], f32> {
    %out = mo.add_singleton_dim[2](%arg0): (!mo.tensor<[4, 3, 16, 1], f32, {layout = #mo.layout<NHWC>}>) -> !mo.tensor<[4, 3, 1, 16, 1], f32>
    mo.output %out: !mo.tensor<[4, 3, 1, 16, 1], f32>
}
