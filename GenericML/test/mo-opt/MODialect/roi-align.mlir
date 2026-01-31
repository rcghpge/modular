//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK:   mo.graph @roi_align(%[[ARG0:.*]]: !mo.tensor<[1, 10, 10, 1], f32>, %[[ROIS:.*]]: !mo.tensor<[1, 5], f32>) -> !mo.tensor<[1, 5, 5, 1], f32> {
mo.graph @roi_align(%inp: !mo.tensor<[1, 10, 10, 1], f32>, %rois: !mo.tensor<[1, 5], f32>) -> !mo.tensor<[1, 5, 5, 1], f32> {
    %output_height = mo.constant {value = #M.dense_array<5> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %spatial_scale = mo.constant {value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
    %sampling_ratio = mo.constant {value = #M.dense_array<2.0> : tensor<1xf32>} : !mo.tensor<[], f32>

    %output = mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio) {aligned = false,  mode = "AVG"} : (!mo.tensor<[1, 10, 10, 1], f32>, !mo.tensor<[1, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
    mo.output %output : !mo.tensor<[1, 5, 5, 1], f32>
}
