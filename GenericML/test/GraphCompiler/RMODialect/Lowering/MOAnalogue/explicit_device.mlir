// This file tests the proper handling of explicit devices on tensors for RMO-MO Analogue ops
// RUN: mo-opt --mo-legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope


// Ensures device annotations are properly forwarded in lowering
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @forward_device_annotations_batch_matmul<() -> Sh0: !mosh.ape>(%[[ARG0:.*]]: !mo.tensor<[2, 7, 4], f32, cpu:1>, %[[ARG1:.*]]: !mo.tensor<[2, 4, 9], f32, cpu:1>) -> !mo.tensor<Sh0, f32, cpu:1> {
//  CHECK-NEXT:   %[[V0:.*]] = mo.batch_matmul<() -> [[RMO_D1:.*]], [[RMO_D2:.*]], [[RMO_D3:.*]]>(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 7, 4], f32, cpu:1>, !mo.tensor<[2, 4, 9], f32, cpu:1>) -> !mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32, cpu:1>
//  CHECK-NEXT:   %[[V1:.*]] = rmo.rebind_tensor_shape<() -> [[SH1:.*]]: !mosh.ape>(%[[V0]]) {{.*}} : (!mo.tensor<[[[RMO_D1]], [[RMO_D2]], [[RMO_D3]]], f32, cpu:1>) -> !mo.tensor<[[SH1]], f32, cpu:1>
//  CHECK-NEXT:   mo.output<:!mosh.ape [[SH1]]> %[[V1]] : !mo.tensor<[[SH1]], f32, cpu:1>
mo.graph @forward_device_annotations_batch_matmul<() -> Sh0: !mosh.ape>(%arg0: !mo.tensor<[2, 7, 4], f32, cpu:1>, %arg1: !mo.tensor<[2, 4, 9], f32, cpu:1>) -> !mo.tensor<Sh0, f32, cpu:1> {
  %0 = rmo.mo.batch_matmul<() -> Sh1: !mosh.ape>(%arg0, %arg1) : (!mo.tensor<[2, 7, 4], f32, cpu:1>, !mo.tensor<[2, 4, 9], f32, cpu:1>) -> !mo.tensor<Sh1, f32, cpu:1>
  mo.output<:!mosh.ape Sh1> %0 : !mo.tensor<Sh1, f32, cpu:1>
}
