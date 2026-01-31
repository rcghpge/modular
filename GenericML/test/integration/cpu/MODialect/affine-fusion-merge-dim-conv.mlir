// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones %t.mlir | FileCheck %s

// CHECK-LABEL: 'conv_merge_dim_rank3_merge_1' returned tensor<1x4x3xf32>
// CHECK-SAME: 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
// CHECK-SAME: 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01
mo.graph @conv_merge_dim_rank3_merge_1(%input : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 4, 3], f32>) {
    %bias = "mo.constant"() {value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.> : tensor<12xf32>} : () -> !mo.tensor<[1, 4, 3], f32>
    %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
    %ng = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
    %filter = "mo.constant"() {value = #M.dense_array<1.0, 1.0, 1.0> : tensor<1x1x1x3xf32>} : () -> !mo.tensor<[1, 1, 1, 3], f32>
    %0 = mo.add(%filter, %filter) : !mo.tensor<[1, 1, 1, 3], f32>
    %conv = mo.conv(%input, %0) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[1, 2, 2, 3], f32>, !mo.tensor<[1, 1, 1, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
    ) -> !mo.tensor<[1, 2, 2, 3], f32>
    %merged = mo.merge_dim[1](%conv): (!mo.tensor<[1, 2, 2, 3], f32>) -> !mo.tensor<[1, 4, 3], f32>
    %final = mo.add(%merged, %bias): !mo.tensor<[1, 4, 3], f32>
    mo.output %final : !mo.tensor<[1, 4, 3], f32>
}

// CHECK-LABEL: 'conv_merge_dim_rank3_merge_2' returned tensor<1x2x6xf32>
// CHECK-SAME: 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
// CHECK-SAME: 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01
mo.graph @conv_merge_dim_rank3_merge_2(%input : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 2, 6], f32>) {
    %bias = "mo.constant"() {value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.> : tensor<12xf32>} : () -> !mo.tensor<[1, 2, 6], f32>
    %st = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %di = "mo.constant"() {value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %pa = "mo.constant"() {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
    %ng = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
    %filter = "mo.constant"() {value = #M.dense_array<1.0, 1.0, 1.0> : tensor<1x1x1x3xf32>} : () -> !mo.tensor<[1, 1, 1, 3], f32>
    %0 = mo.add(%filter, %filter) : !mo.tensor<[1, 1, 1, 3], f32>
    %conv = mo.conv(%input, %0) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[1, 2, 2, 3], f32>, !mo.tensor<[1, 1, 1, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
    ) -> !mo.tensor<[1, 2, 2, 3], f32>
    %merged = mo.merge_dim[2](%conv): (!mo.tensor<[1, 2, 2, 3], f32>) -> !mo.tensor<[1, 2, 6], f32>
    %final = mo.add(%merged, %bias): !mo.tensor<[1, 2, 6], f32>
    mo.output %final : !mo.tensor<[1, 2, 6], f32>
}
