// This file tests the proper handling of explicit devices on tensors for RMO-MO Analogue ops
// RUN: mo-opt --mo-resolve-unknown-parameters --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope


// Ensures device annotations are properly forwarded in lowering
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @forward_device_annotations_batch_matmul<() -> [[SH1:[A-Za-z0-9_]*]]: !mosh.ape>([[ARG0:%.*]]: !mo.tensor<[2, 7, 4], f32, cpu:1>, [[ARG1:%.*]]: !mo.tensor<[2, 4, 9], f32, cpu:1>) -> !mo.tensor<[[SH1]], f32, cpu:1> {
//  CHECK-NEXT:   [[V0:%.*]] = mo.batch_matmul<() -> [[RMO_D3:[A-Za-z0-9_]*]], [[RMO_D4:[A-Za-z0-9_]*]], [[RMO_D5:[A-Za-z0-9_]*]]>([[ARG0]], [[ARG1]]) : (!mo.tensor<[2, 7, 4], f32, cpu:1>, !mo.tensor<[2, 4, 9], f32, cpu:1>) -> !mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32, cpu:1>
//  CHECK-NEXT:   [[V1:%.*]] = rmo.rebind_tensor_shape<() -> [[SH2:[A-Za-z0-9_]*]]: !mosh.ape>([[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D3]], [[RMO_D4]], [[RMO_D5]]], f32, cpu:1>) -> !mo.tensor<[[SH2]], f32, cpu:1>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH2]]> [[V1]] : !mo.tensor<[[SH2]], f32, cpu:1>
mo.graph @forward_device_annotations_batch_matmul(%arg0: !mo.tensor<[2, 7, 4], f32, cpu:1>, %arg1: !mo.tensor<[2, 4, 9], f32, cpu:1>) -> !mo.tensor<?, f32, cpu:1> {
  %0 = rmo.mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[2, 7, 4], f32, cpu:1>, !mo.tensor<[2, 4, 9], f32, cpu:1>) -> !mo.tensor<?, f32, cpu:1>
  mo.output %0 : !mo.tensor<?, f32, cpu:1>
}
