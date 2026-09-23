// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels --split-input-file | FileCheck %s

// `mo.merge_dim` never becomes an index transform on this path -- it lowers to
// its own `merge_dim_indices` primitive -- so the collapse is spelled with
// `mo.static.reshape`, which FuseReshapeEpilogue moves onto the conv's store
// index. Result: the reshape and the add share the conv's epilogue.

// CHECK-LABEL: mo.graph @conv_reshape_rank3_merge_1
// CHECK: map.iter.opaque
// CHECK: map.index.reshape<#map.index.reshape.spec<[1, 4, 3] -> [1, 2, 2, 3]>>
// CHECK-DAG: fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.conv"{{.*}}, #mogg.fusion_view<[#mogg.fusion_leaf<"mo.static.reshape"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>], operandIndices = [0]>>
mo.graph @conv_reshape_rank3_merge_1(%input : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 4, 3], f32>) {
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
    %reshaped = mo.static.reshape(%conv): !mo.tensor<[1, 2, 2, 3], f32> -> !mo.tensor<[1, 4, 3], f32>
    %final = mo.add(%reshaped, %bias): !mo.tensor<[1, 4, 3], f32>
    mo.output %final : !mo.tensor<[1, 4, 3], f32>
}

// -----

// CHECK-LABEL: mo.graph @conv_reshape_rank3_merge_2
// CHECK: map.iter.opaque
// CHECK: map.index.reshape<#map.index.reshape.spec<[1, 2, 6] -> [1, 2, 2, 3]>>
// CHECK-DAG: fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.conv"{{.*}}, #mogg.fusion_view<[#mogg.fusion_leaf<"mo.static.reshape"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>], operandIndices = [0]>>
mo.graph @conv_reshape_rank3_merge_2(%input : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 2, 6], f32>) {
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
    %reshaped = mo.static.reshape(%conv): !mo.tensor<[1, 2, 2, 3], f32> -> !mo.tensor<[1, 2, 6], f32>
    %final = mo.add(%reshaped, %bias): !mo.tensor<[1, 2, 6], f32>
    mo.output %final : !mo.tensor<[1, 2, 6], f32>
}
