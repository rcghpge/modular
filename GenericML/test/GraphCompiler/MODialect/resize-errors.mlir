//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @resize_nearest(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    %size = mo.constant {value = #M.dense_array<1.0, 1.0, 7.0, 8.0> : tensor<4xf32>} : !mo.tensor<[4], f32>
    // expected-error@+1 {{'mo.resize.nearest' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[4], f32>'}}
    %output = mo.resize.nearest(%input, %size) {coordinate_transform_mode = #mo<coord_trans_mode half_pixel>, round_mode = 2}: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// -----


mo.graph @resize_linear(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    %size = mo.constant {value = #M.dense_array<1.0, 1.0, 7.0, 8.0> : tensor<4xf32>} : !mo.tensor<[4], f32>
    // expected-error@+1 {{'mo.resize.linear' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[4], f32>'}}
    %output = mo.resize.linear(%input, %size) {coordinate_transform_mode = #mo<coord_trans_mode align_corners>, antialias = true} : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}
