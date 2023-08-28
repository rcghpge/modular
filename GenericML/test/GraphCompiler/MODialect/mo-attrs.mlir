// RUN: mo-opt %s --allow-unregistered-dialect | FileCheck %s

mo.graph @conv_layouts(
  %nhwc_input: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
  %nchw_input: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}>,
  %rscf_filter: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<RSCF>}>,
  %fcrs_filter: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<FCRS>}>,
  %packed_filter: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<FRSCf>}>
) {
  mo.output
}
// CHECK:      mo.graph @conv_layouts(
// CHECK-SAME:   %{{.*}}: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
// CHECK-SAME:   %{{.*}}: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NCHW>}>,
// CHECK-SAME:   %{{.*}}: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<RSCF>}>,
// CHECK-SAME:   %{{.*}}: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<FCRS>}>,
// CHECK-SAME:   %{{.*}}: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<FRSCf>}>
// CHECK-SAME: ) no_inline {
// CHECK-NEXT:   mo.output
// CHECK-NEXT: }


mo.graph @matmul_layouts(
  %weights: !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>,
  %transposed_weights: !mo.tensor<[?, ?], f32, {layout = #mo.layout<NK>}>,
  %packed_weights: !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<KNkni>}>
) {
  mo.output
}
// CHECK:      mo.graph @matmul_layouts(
// CHECK-SAME:   %{{.*}}: !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>,
// CHECK-SAME:   %{{.*}}: !mo.tensor<[?, ?], f32, {layout = #mo.layout<NK>}>,
// CHECK-SAME:   %{{.*}}: !mo.tensor<[?, ?, ?, ?, ?], f32, {layout = #mo.layout<KNkni>}>
// CHECK-SAME: ) no_inline {
// CHECK-NEXT:   mo.output
// CHECK-NEXT: }


mo.graph @dtype_attr() {
  // No ops defined which eat DType attributes
  %foo1 = "fakedialect.bar_op" () {dtype = #mo.dtype<si8>}: () -> (!mo.tensor<[], f32>)
  %foo2 = "fakedialect.bar_op" () {dtype = #mo.dtype<f16>}: () -> (!mo.tensor<[], f32>)
  %foo3 = "fakedialect.bar_op" () {dtype = #mo.dtype<bf16>}: () -> (!mo.tensor<[], f32>)
  %foo4 = "fakedialect.bar_op" () {dtype = #mo.dtype<f32>}: () -> (!mo.tensor<[], f32>)
  mo.output
}
// CHECK: mo.graph @dtype_attr(
// CHECK-DAG: dtype = #mo.dtype<si8>
// CHECK-DAG: dtype = #mo.dtype<f16>
// CHECK-DAG: dtype = #mo.dtype<bf16>
// CHECK-DAG: dtype = #mo.dtype<f32>
// CHECK-NEXT: mo.output
