// This file tests the proper handling of explicit devices on tensors for RMO-MO Analogue ops
// RUN: mo-opt --split-input-file --resolve-unknown-parameters --legalize-rmo-operators="skip-rmo-rebinds" %s | FileCheck %s --enable-var-scope

// 
mo.graph @transfer_d2d(%arg0: !mo.tensor<[4, 4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32, gpu:1> {
  %0 = rmo.mo.transfer %arg0 : !mo.tensor<[4, 4], f32, gpu:0> to <"gpu", 1>
  mo.output %0 : !mo.tensor<[4, 4], f32, gpu:1>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transfer_d2d([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL:[A-Za-z0-9]*]]:[[IN_DEV_ID:[0-9]*]]>) -> !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL:[A-Za-z0-9]*]]:[[OUT_DEV_ID:[0-9]*]]> {
//  CHECK-NEXT:   [[V1:%.*]]  = mo.transfer [[ARG0]] : !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL]]:[[IN_DEV_ID]]> to <"[[OUT_DEV_LABEL]]", [[OUT_DEV_ID]]>
//  CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL]]:[[OUT_DEV_ID]]>

// -----

mo.graph @transfer_d2h(%arg0: !mo.tensor<[4, 4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32, cpu:0> {
  %0 = rmo.mo.transfer %arg0 : !mo.tensor<[4, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %0 : !mo.tensor<[4, 4], f32, cpu:0>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transfer_d2h([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL:[A-Za-z0-9]*]]:[[IN_DEV_ID:[0-9]*]]>) -> !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL:[A-Za-z0-9]*]]:[[OUT_DEV_ID:[0-9]*]]> {
//  CHECK-NEXT:   [[V1:%.*]]  = mo.transfer [[ARG0]] : !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL]]:[[IN_DEV_ID]]> to <"[[OUT_DEV_LABEL]]", [[OUT_DEV_ID]]>
//  CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL]]:[[OUT_DEV_ID]]>

// -----

mo.graph @transfer_h2d(%arg0: !mo.tensor<[4, 4], f32, cpu:0>) -> !mo.tensor<[4, 4], f32, gpu:1> {
  %0 = rmo.mo.transfer %arg0 : !mo.tensor<[4, 4], f32, cpu:0> to <"gpu", 1>
  mo.output %0 : !mo.tensor<[4, 4], f32, gpu:1>
}
// CHECK-LABEL: mo.graph
//  CHECK-SAME: @transfer_h2d([[ARG0:%[A-Za-z0-9_]*]]: !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL:[A-Za-z0-9]*]]:[[IN_DEV_ID:[0-9]*]]>) -> !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL:[A-Za-z0-9]*]]:[[OUT_DEV_ID:[0-9]*]]> {
//  CHECK-NEXT:   [[V1:%.*]] = mo.transfer [[ARG0]] : !mo.tensor<[4, 4], f32, [[IN_DEV_LABEL]]:[[IN_DEV_ID]]> to <"[[OUT_DEV_LABEL]]", [[OUT_DEV_ID]]>
//  CHECK-NEXT:   mo.output [[V1]] : !mo.tensor<[4, 4], f32, [[OUT_DEV_LABEL]]:[[OUT_DEV_ID]]>

