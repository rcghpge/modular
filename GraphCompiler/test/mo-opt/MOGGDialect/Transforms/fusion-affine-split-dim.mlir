// RUN: mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s

// The kernel summary string should reflect an accurate fusion record.
// `mo.split_dim` never becomes an index transform on this path -- it lowers to
// its own indices primitive -- so the collapse is spelled with
// `mo.static.reshape`, which FuseReshapeEpilogue moves onto the matmul's store
// index. Result: the reshape and the add share the matmul's epilogue.
// DAG because the order things are printed in the middle of the pipeline is
// not predictable.

// CHECK-LABEL: mo.graph @simple_reshape_rank3_split_0
// CHECK:       map.iter.opaque
// CHECK:       map.index.reshape<#map.index.reshape.spec<[2, 2, 8] -> [4, 8]>>
// CHECK-DAG:   fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul"{{.*}}, #mogg.fusion_view<[#mogg.fusion_leaf<"mo.static.reshape"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>], operandIndices = [0]>>
// CHECK-NOT:   map.iter.foreach
mo.graph @simple_reshape_rank3_split_0(%arg0: !mo.tensor<[4, 10], f32>, %arg1: !mo.tensor<[10, 8], f32>) -> !mo.tensor<[2, 2, 8], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
       14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
       27., 28., 29., 30., 31., 32.> : tensor<32xf32>} : () -> !mo.tensor<[2, 2, 8], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[4, 10], f32>, !mo.tensor<[10, 8], f32>) -> !mo.tensor<[4, 8], f32>
    %out = mo.static.reshape(%res): !mo.tensor<[4, 8], f32> -> !mo.tensor<[2, 2, 8], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[2, 2, 8], f32>
    mo.output %final: !mo.tensor<[2, 2, 8], f32>
}

// CHECK-LABEL: mo.graph @simple_reshape_rank3_split_1
// CHECK:       map.iter.opaque
// CHECK:       map.index.reshape<#map.index.reshape.spec<[4, 2, 4] -> [4, 8]>>
// CHECK-DAG:   fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"mo.matmul"{{.*}}, #mogg.fusion_view<[#mogg.fusion_leaf<"mo.static.reshape"{{.*}}, #mogg.fusion_leaf<"mo.add"{{.*}}]>], operandIndices = [0]>>
// CHECK-NOT:   map.iter.foreach
mo.graph @simple_reshape_rank3_split_1(%arg0: !mo.tensor<[4, 10], f32>, %arg1: !mo.tensor<[10, 8], f32>) -> !mo.tensor<[4, 2, 4], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
       14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
       27., 28., 29., 30., 31., 32.> : tensor<32xf32>} : () -> !mo.tensor<[4, 2, 4], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[4, 10], f32>, !mo.tensor<[10, 8], f32>) -> !mo.tensor<[4, 8], f32>
    %out = mo.static.reshape(%res): !mo.tensor<[4, 8], f32> -> !mo.tensor<[4, 2, 4], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[4, 2, 4], f32>
    mo.output %final: !mo.tensor<[4, 2, 4], f32>
}
