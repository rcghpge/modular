//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK:   mo.graph @resize.nearest(%[[ARG0:.*]]: !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> no_inline {
mo.graph @resize.nearest(%arg0 : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    //  CHECK-NEXT: %[[ARG1:.*]] = mo.constant {value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %size = mo.constant {value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    // CHECK-NEXT: %1 = mo.resize.nearest(%[[ARG0]], %[[ARG1]]) {coordinate_transform_mode = 1 : i64, round_mode = 2 : i64} : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    %output = mo.resize.nearest(%arg0, %size) {coordinate_transform_mode = 1, round_mode = 2}: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// CHECK:   mo.graph @resize.nearest2(%[[ARG0:.*]]: !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> no_inline {
mo.graph @resize.nearest2(%arg0 : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    //  CHECK-NEXT: %[[ARG1:.*]] = mo.constant {value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %size = mo.constant {value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    // CHECK-NEXT: %1 = mo.resize.nearest(%[[ARG0]], %[[ARG1]]) {round_mode = 2 : i64} : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    %output = mo.resize.nearest(%arg0, %size) {round_mode = 2}: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}

// CHECK:   mo.graph @resize.nearest3(%[[ARG0:.*]]: !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> no_inline {
mo.graph @resize.nearest3(%arg0 : !mo.tensor<[1, 1, 2, 2], f32>) -> !mo.tensor<[1, 1, 7, 8], f32> {
    //  CHECK-NEXT: %[[ARG1:.*]] = mo.constant {value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %size = mo.constant {value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    // CHECK-NEXT: %1 = mo.resize.nearest(%[[ARG0]], %[[ARG1]]) {coordinate_transform_mode = 1 : i64} : (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    %output = mo.resize.nearest(%arg0, %size) {coordinate_transform_mode = 1 : i64}: (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    mo.output %output : !mo.tensor<[1, 1, 7, 8], f32>
}
