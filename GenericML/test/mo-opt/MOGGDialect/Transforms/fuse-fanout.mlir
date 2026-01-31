// RUN: mo-opt --mo-to-mogg --mogg-fuse-elementwise --canonicalize --cse %s | FileCheck %s

// If we need to materialize a kernel then that should prevent it from being
// folded into subsequent fusions to prevent duplicate compute.

mo.graph @simpleTwoKernels(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.tanh(%0) : !mo.tensor<[4, 3], f32>
  %2 = mo.add(%1, %1) : !mo.tensor<[4, 3], f32>
  %3 = mo.sub(%2, %arg0) : !mo.tensor<[4, 3], f32>
  mo.output %2, %3 : !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>
}


// CHECK: mo.graph @simpleTwoKernels(%[[ARG0:.*]]: !mo.tensor<[4, 3], f32>)
// CHECK-NEXT: %[[KERNEL_1:.*]] = mogg.kernel(%[[ARG0]] as %{{.*}}: !mo.tensor<[4, 3], f32>, %[[ARG0]] as %{{.*}}: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32> {
// CHECK: mogg.output_placeholders : !mo.tensor<[4, 3], f32>
// CHECK: mogg.bind (%{{.*}}, %{{.*}}) elemwise_lambda (%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: !pop.array<2, index>) {
// CHECK: mogg.call.elementwise["mo.sqrt"] (%{{.*}}) : f32 -> f32
// CHECK: mogg.call.elementwise["mo.sqrt"] (%{{.*}}) : f32 -> f32
// CHECK: mogg.call.elementwise["mo.tanh"] (%{{.*}}) : f32 -> f32
// CHECK: mogg.call.elementwise["mo.add"] (%{{.*}}, %{{.*}}) : f32, f32 -> f32
// CHECK: mogg.output


// CHECK: %[[KERNEL_2:.*]] = mogg.kernel(%[[KERNEL_1]]: !mo.tensor<[4, 3], f32>, %[[ARG0]]: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[4, 3], f32> {
// CHECK: mogg.output_placeholders : !mo.tensor<[4, 3], f32>
// CHECK: mogg.bind (%[[KERNEL_1]], %[[ARG0]]) elemwise_lambda (%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: !pop.array<2, index>) {
// CHECK: mogg.call.elementwise["mo.sub"] (%{{.*}}, %{{.*}}) : f32, f32 -> f32
// CHECK: mogg.output
// CHECK: mo.output %[[KERNEL_1]], %[[KERNEL_2]] : !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>
