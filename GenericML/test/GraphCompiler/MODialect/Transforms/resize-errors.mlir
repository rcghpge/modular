//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @resize(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    %size = mo.constant {value = #M.dense_array<1.0, 1.0, 7.0, 8.0> : tensor<4xf32>} : !mo.tensor<[4], f32>
    // expected-error@+1 {{'mo.resize.nearest' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[4], f32>'}}
    %output = mo.resize.nearest(%input, %size) {coordinate_transform_mode = 0, round_mode = 2}: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// -----

mo.graph @resize(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    %size = mo.constant {value = #M.dense_array<1.0, 1.0, 7.0, 8.0> : tensor<4xf32>} : !mo.tensor<[4], si64>
    // expected-error@+1 {{'mo.resize.nearest' op round_mode attribute must be between 0 and 3}}
    %output = mo.resize.nearest(%input, %size) {coordinate_transform_mode = 0, round_mode = 4}: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// -----

mo.graph @resize(%input : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    %size = mo.constant {value = #M.dense_array<1.0, 1.0, 7.0, 8.0> : tensor<4xf32>} : !mo.tensor<[4], si64>
    // expected-error@+1 {{'mo.resize.nearest' op coordinate_transform_mode attribute must be between 0 and 2}}
    %output = mo.resize.nearest(%input, %size) {coordinate_transform_mode = -1, round_mode = 3}: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}
