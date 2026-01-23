// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" --mlir-print-ir-after=mo-to-mogg 2>&1 | FileCheck %s

// Test layout transform lowering to MOGG dialect with experimental Mojo path.

// CHECK-LABEL: mo.graph @layout_transform_RSCF_to_FRSCf
// CHECK:  mogg.kernel([[ARG_RSCF:%.+]]: !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>) -> !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}> {
// CHECK:   [[OUT_RSCF:%.+]] = mogg.output_placeholders : !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}>
// CHECK:   mogg.call.execute["layout_transform_RSCF_to_FRSCf"] inputs([[ARG_RSCF]] : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>) outputs([[OUT_RSCF]] : !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}>) {kernel_param = {num_groups = 1 : index}}
// CHECK: summary = "mo.layout.transform_RSCF_to_FRSCf:{{[0-9]+}}"
mo.graph @layout_transform_RSCF_to_FRSCf(%filter: !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>) -> !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}> {
  %packed = mo.layout.transform(%filter) {kgenParams = {num_groups = 1 : index}} : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}> -> !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}>
  mo.output %packed : !mo.tensor<[1, 2, 2, 1, 16], f32, {layout = #mo.layout<FRSCf>}>
}

// CHECK-LABEL: mo.graph @matmul_packb_static
// CHECK:  mogg.kernel([[ARG_KN:%.+]]: !mo.tensor<{{[^>]*}}, f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<{{[^>]*}}, f32, {layout = #mo.layout<KNkni>}> {
// CHECK:   [[OUT_KNkni:%.+]] = mogg.output_placeholders : !mo.tensor<{{[^>]*}}, f32, {layout = #mo.layout<KNkni>}>
// CHECK:   mogg.call.execute["layout_transform_KN_to_KNkni"] inputs([[ARG_KN]] : !mo.tensor<{{[^>]*}}, f32, {layout = #mo.layout<KN>}>) outputs([[OUT_KNkni]] : !mo.tensor<{{[^>]*}}, f32, {layout = #mo.layout<KNkni>}>) {isThrows, kernel_param = {a_shape = [4000, 2], a_type = #kgen.dtype.constant<f32> : !kgen.dtype, b_shape = [2, 2], b_type = #kgen.dtype.constant<f32> : !kgen.dtype, c_shape = [4000, 2], c_type = #kgen.dtype.constant<f32> : !kgen.dtype}}
// CHECK: summary = "mo.layout.transform_KN_to_KNkni:{{[0-9]+}}"
mo.graph @matmul_packb_static(%x: !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[4000, 2], f32>) {
  %w = mo.constant {value = #M.dense_array<1.0, 1.0, 1.0, 1.0> : tensor<2x2xf32>} : !mo.tensor<[2, 2], f32>
  %act = mo.static.broadcast_to(%x) : !mo.tensor<[2, 2], f32> -> !mo.tensor<[2000, 2, 2], f32>
  %shape = mo.constant {value = #M.dense_array<4000, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %act2 = mo.reshape(%act, %shape) : (!mo.tensor<[2000, 2, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[4000, 2], f32>
  %1 = mo.matmul(%act2, %w) : (!mo.tensor<[4000, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[4000, 2], f32>
  mo.output %1 : !mo.tensor<[4000, 2], f32>
}
