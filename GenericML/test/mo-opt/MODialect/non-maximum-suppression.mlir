//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @non_maximum_suppression(%[[ARG0:.*]]: !mo.tensor<[1, 6, 4], f32>, %[[ARG1:.*]]: !mo.tensor<[1, 1, 6], f32>, %[[ARG2:.*]]: !mo.tensor<[], si64>, %[[ARG3:.*]]: !mo.tensor<[], f32>, %[[ARG4:.*]]: !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64> {
mo.graph @non_maximum_suppression(%arg0 : !mo.tensor<[1, 6, 4], f32>, %arg1 : !mo.tensor<[1, 1, 6], f32>, %arg2 : !mo.tensor<[], si64>, %arg3 : !mo.tensor<[], f32>, %arg4 : !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64> {
    // CHECK-NEXT: %0 = mo.non_maximum_suppression(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]]) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
    %output = mo.non_maximum_suppression(%arg0, %arg1, %arg2, %arg3, %arg4) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
    mo.output %output : !mo.tensor<[?, ?], si64>
}
