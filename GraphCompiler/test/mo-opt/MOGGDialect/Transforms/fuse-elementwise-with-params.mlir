// RUN: mo-opt --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-mogg --mogg-fuse-elementwise --canonicalize --cse %s | FileCheck %s

// If we need to materialize a kernel then that should prevent it from being
// folded into subsequent fusions to prevent duplicate compute.

mo.graph @simpleTwoKernelsWithParams(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.custom {symbol = "elemwise_add_with_param", device = #M.device_ref<"cpu", 0>, parameters = {"bias" = 1}}(%arg0) : (!mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32>
  %1 = mo.custom {symbol = "elemwise_add_with_param", device = #M.device_ref<"cpu", 0>, parameters = {"bias" = 2}}(%0) : (!mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32>
  mo.output %1: !mo.tensor<[4, 3], f32>
}


// CHECK: mo.graph @simpleTwoKernelsWithParams(%[[ARG0:.*]]: !mo.tensor<[4, 3], f32>)
// CHECK-NEXT: mogg.kernel(%[[ARG0]]: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32> {
// CHECK: mogg.output_placeholders : !mo.tensor<[4, 3], f32>
// CHECK: mogg.bind (%{{.*}}) elemwise_lambda (%{{.*}}: f32, %{{.*}}: !pop.array<2, index>) {
// CHECK: mogg.call.elementwise["{{.*}}.ElemwiseAddWithParam"] (%{{.*}}) {kernel_param = {bias = 1 : i64}
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.ElemwiseAddWithParam"] (%{{.*}}) {kernel_param = {bias = 2 : i64}
// CHECK: mogg.output
