// RUN: mo-opt %s --infer-layouts | FileCheck %s


// CHECK:      mo.graph @chained_conv2d_with_reused_filter(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
// CHECK-SAME:   %[[FILTER:.*]]: !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<RSCF>}>,
// CHECK-SAME:   %[[STRIDES:.*]]: !mo.tensor<[2], si64>,
// CHECK-SAME:   %[[DILATIONS:.*]]: !mo.tensor<[2], si64>
// CHECK-SAME: ) -> (!mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>, !mo.tensor<[?, ?, ?, ?], f32>) {
mo.graph @chained_conv2d_with_reused_filter(
  %input: !mo.tensor<[?, ?, ?, ?], f32>,
  %filter: !mo.tensor<[?, ?, ?, ?], f32>,
  %strides: !mo.tensor<[2], si64>,
  %dilations: !mo.tensor<[2], si64>
) -> (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>) {

  // CHECK-NEXT: %[[CONV1:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]]] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<RSCF>}>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>
  %conv1 = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations] : (
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: %[[CONV2:.*]] = mo.conv(%[[CONV1]], %[[FILTER]]) [strides = %[[STRIDES]], dilations = %[[DILATIONS]]] : (
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>,
  // CHECK-SAME:   !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<RSCF>}>,
  // CHECK-SAME:   !mo.tensor<[2], si64>,
  // CHECK-SAME:   !mo.tensor<[2], si64>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?, ?, ?], f32>
  %conv2 = mo.conv(%conv1, %filter) [strides = %strides, dilations = %dilations] : (
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[?, ?, ?, ?], f32>,
    !mo.tensor<[2], si64>,
    !mo.tensor<[2], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  // CHECK-NEXT: mo.output %[[CONV1]], %[[CONV2]] : !mo.tensor<[?, ?, ?, ?], f32, {layout = #mo.layout<NHWC>}>, !mo.tensor<[?, ?, ?, ?], f32>
  mo.output %conv1, %conv2 : !mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[?, ?, ?, ?], f32>
}


// CHECK:      mo.graph @matmul_dyn_weights_in_output(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[?, ?], f32>,
// CHECK-SAME:   %[[W1:.*]]: !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>,
// CHECK-SAME:   %[[W2:.*]]: !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>
// CHECK-SAME: ) -> (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>) {
mo.graph @matmul_dyn_weights_in_output(
  %input: !mo.tensor<[?, ?], f32>,
  %w1: !mo.tensor<[?, ?], f32>,
  %w2: !mo.tensor<[?, ?], f32>
) -> (!mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>) {

  // CHECK-NEXT: %[[WEIGHTS:.*]] = mo.add(%[[W1]], %[[W2]]) : !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>
  %weights = mo.add(%w1, %w2) : !mo.tensor<[?, ?], f32>

  // CHECK-NEXT: %[[MATMUL:.*]] = mo.matmul(%[[INPUT]], %[[WEIGHTS]]) : (
  // CHECK-SAME:   !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>
  // CHECK-SAME: ) -> !mo.tensor<[?, ?], f32>
  %matmul = mo.matmul(%input, %weights) : (
    !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>
  ) -> !mo.tensor<[?, ?], f32>

  // CHECK-NEXT: mo.output %[[MATMUL]], %[[WEIGHTS]] : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32, {layout = #mo.layout<KN>}>
  mo.output %matmul, %weights : !mo.tensor<[?, ?], f32>, !mo.tensor<[?, ?], f32>
}
