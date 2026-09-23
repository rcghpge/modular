// RUN: mo-opt %s --gc-pipeline="constant-fold=false" --mlir-print-ir-after=mo-to-mogg 2>&1 | FileCheck %s

// Test layout transform lowering to MOGG dialect with experimental Mojo path.

// CHECK-LABEL: mo.graph @layout_transform_RSCF_to_FRSCf
// CHECK:  mogg._kernel ({{%.+}} : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>) -> !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}>
// CHECK:   fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.layout.transform_RSCF_to_FRSCf"
// CHECK:   mogg._call.opaque @{{.*}}.LayoutTransformRSCF2FRSCf inputs({{%.+}} : !mogg.tensor_) outputs({{%.+}} : !mogg.tensor_) {kernel_param = {num_groups = 1 : index}}
mo.graph @layout_transform_RSCF_to_FRSCf(%filter: !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>) -> !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}> {
  %packed = mo.layout.transform(%filter) {kgenParams = {num_groups = 1 : index}} : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}> -> !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}>
  mo.output %packed : !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}>
}

// CHECK-LABEL: mo.graph @matmul_packb_static
// CHECK:  mogg._kernel () -> !mo.tensor<{{[^>]*}}, f32, {layout = #mo.layout<KNkni>}>
// CHECK:   fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mo.layout.transform_KN_to_KNkni"
// CHECK:   mogg._call.opaque @{{.*}}.LayoutTransformMatmulKN2KNkni inputs({{%.+}} : !mogg.tensor_) outputs({{%.+}} : !mogg.tensor_) {kernel_param = {a_shape = [4000, 2], a_type = #kgen.dtype.constant<f32> : !kgen.dtype, b_shape = [2, 2], b_type = #kgen.dtype.constant<f32> : !kgen.dtype, c_shape = [4000, 2], c_type = #kgen.dtype.constant<f32> : !kgen.dtype}}
mo.graph @matmul_packb_static(%x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[4000, 2], f32>) {
  %w = mo.constant {value = #M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<2x2xf32>} : !mo.tensor<[2, 2], f32>
  %act = mo.static.broadcast_to(%x) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[2000, 2, 2], f32>
  %shape = mo.constant {value = #M.dense_array<4000, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %act2 = mo.reshape(%act, %shape) : (!mo.tensor<[2000, 2, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4000, 2], f32>
  %1 = mo.matmul(%act2, %w) : (!mo.tensor<[4000, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[4000, 2], f32>
  mo.output %1 : !mo.tensor<[4000, 2], f32>
}
