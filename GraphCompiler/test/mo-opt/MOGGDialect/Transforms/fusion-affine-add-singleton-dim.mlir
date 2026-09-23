// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s

// The kernel summary string should reflect an accurate fusion record.
// `mo.add_singleton_dim` never becomes an index transform on this path -- it
// lowers to its own indices primitive -- so the collapse is spelled with
// `mo.static.reshape`, which FuseReshapeEpilogue moves onto the matmul's store
// index. Result: the reshape and the add share the matmul's epilogue.
// DAG because the order things are printed in the middle of the pipeline is
// not predictable.

// CHECK-LABEL: mo.graph @simple_reshape_rank3_insert1_middle
// CHECK:       map.iter.opaque
// CHECK:       map.index.reshape<#map.index.reshape.spec<[2, 1, 3] -> [2, 3]>>
// CHECK-DAG:   fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul"{{.*}}, #mogg.fusion_view<[#mogg.fusion_leaf<"mo.static.reshape"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>], operandIndices = [0]>>
// CHECK-NOT:   map.iter.foreach
mo.graph @simple_reshape_rank3_insert1_middle(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>} : () -> !mo.tensor<[2, 1, 3], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[2, 10], f32>, !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3], f32>
    %out = mo.static.reshape(%res): !mo.tensor<[2, 3], f32> -> !mo.tensor<[2, 1, 3], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[2, 1, 3], f32>
    mo.output %final: !mo.tensor<[2, 1, 3], f32>
}
// CHECK-LABEL: mo.graph @simple_reshape_rank3_insert1_start
// CHECK:       map.iter.opaque
// CHECK:       map.index.reshape<#map.index.reshape.spec<[1, 2, 3] -> [2, 3]>>
// CHECK-DAG:   fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul"{{.*}}, #mogg.fusion_view<[#mogg.fusion_leaf<"mo.static.reshape"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>], operandIndices = [0]>>
// CHECK-NOT:   map.iter.foreach
mo.graph @simple_reshape_rank3_insert1_start(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[10, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>} : () -> !mo.tensor<[1, 2, 3], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[2, 10], f32>, !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3], f32>
    %out = mo.static.reshape(%res): !mo.tensor<[2, 3], f32> -> !mo.tensor<[1, 2, 3], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[1, 2, 3], f32>
    mo.output %final: !mo.tensor<[1, 2, 3], f32>
}
// CHECK-LABEL: mo.graph @simple_reshape_rank3_insert1_end
// CHECK:       map.iter.opaque
// CHECK:       map.index.reshape<#map.index.reshape.spec<[2, 3, 1] -> [2, 3]>>
// CHECK-DAG:   fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul"{{.*}}, #mogg.fusion_view<[#mogg.fusion_leaf<"mo.static.reshape"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>], operandIndices = [0]>>
// CHECK-NOT:   map.iter.foreach
mo.graph @simple_reshape_rank3_insert1_end(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3, 1], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>} : () -> !mo.tensor<[2, 3, 1], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[2, 10], f32>, !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3], f32>
    %out = mo.static.reshape(%res): !mo.tensor<[2, 3], f32> -> !mo.tensor<[2, 3, 1], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[2, 3, 1], f32>
    mo.output %final: !mo.tensor<[2, 3, 1], f32>
}
