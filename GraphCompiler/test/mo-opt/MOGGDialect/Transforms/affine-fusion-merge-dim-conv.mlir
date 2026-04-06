// RUN: mo-opt %s --gc-pipeline="constant-fold=false" --split-input-file -o /dev/null --mlir-print-ir-after=mogg-fuse-epilogues 2>&1 | FileCheck %s

// CHECK-LABEL: mo.graph @conv_merge_dim_rank3_merge_1
// CHECK-DAG: fusionInfo = Epilogue(Affine("mo.conv":{{[0-9]+}}, "mo.merge_dim":{{[0-9]+}}), "mo.add":{{[0-9]+}}, operandIndices=[0])
mo.graph @conv_merge_dim_rank3_merge_1(%input : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 4, 3], f32>) {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.> : tensor<12xf32>} : () -> !mo.tensor<[1, 4, 3], f32>
    %st = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %di = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %pa = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
    %ng = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
    %filter = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.0, 1.0, 1.0> : tensor<1x1x1x3xf32>} : () -> !mo.tensor<[1, 1, 1, 3], f32, {layout = #mo.layout<RSCF>}>
    %0 = mo.add(%filter, %filter) : !mo.tensor<[1, 1, 1, 3], f32, {layout = #mo.layout<RSCF>}>
    %conv = mo.conv(%input, %0) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[1, 2, 2, 3], f32>, !mo.tensor<[1, 1, 1, 3], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
    ) -> !mo.tensor<[1, 2, 2, 3], f32>
    %merged = mo.merge_dim[1](%conv): (!mo.tensor<[1, 2, 2, 3], f32>) -> !mo.tensor<[1, 4, 3], f32>
    %final = mo.add(%merged, %bias): !mo.tensor<[1, 4, 3], f32>
    mo.output %final : !mo.tensor<[1, 4, 3], f32>
}

// -----

// CHECK-LABEL: mo.graph @conv_merge_dim_rank3_merge_2
// CHECK-DAG: fusionInfo = Epilogue(Affine("mo.conv":{{[0-9]+}}, "mo.merge_dim":{{[0-9]+}}), "mo.add":{{[0-9]+}}, operandIndices=[0])
mo.graph @conv_merge_dim_rank3_merge_2(%input : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 2, 6], f32>) {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.> : tensor<12xf32>} : () -> !mo.tensor<[1, 2, 6], f32>
    %st = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %di = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %pa = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
    %ng = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
    %filter = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.0, 1.0, 1.0> : tensor<1x1x1x3xf32>} : () -> !mo.tensor<[1, 1, 1, 3], f32, {layout = #mo.layout<RSCF>}>
    %0 = mo.add(%filter, %filter) : !mo.tensor<[1, 1, 1, 3], f32, {layout = #mo.layout<RSCF>}>
    %conv = mo.conv(%input, %0) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[1, 2, 2, 3], f32>, !mo.tensor<[1, 1, 1, 3], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
    ) -> !mo.tensor<[1, 2, 2, 3], f32>
    %merged = mo.merge_dim[2](%conv): (!mo.tensor<[1, 2, 2, 3], f32>) -> !mo.tensor<[1, 2, 6], f32>
    %final = mo.add(%merged, %bias): !mo.tensor<[1, 2, 6], f32>
    mo.output %final : !mo.tensor<[1, 2, 6], f32>
}
