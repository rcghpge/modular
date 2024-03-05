// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @cast_ok(%arg0: !mo.tensor<?, f32>) -> !mo.tensor<[?, ?, ?], si32> {
  %1 = rmo.cast(%arg0) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, ?, ?], si32>
  mo.output %1 : !mo.tensor<[?, ?, ?], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<[?, ?, ?], si32> no_inline {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.cast([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<[?, ?, ?], si32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[?, ?, ?], si32>
