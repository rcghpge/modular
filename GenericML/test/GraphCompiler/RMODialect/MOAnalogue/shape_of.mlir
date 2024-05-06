// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @shape_of_ok(%input: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.shape_of(%input): (!mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @shape_of_ok([[VARG0:[^ ]*]]: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.shape_of([[VARG0]]) : (!mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[?], si64>

// -----

mo.graph @shape_of_not_ok(%input: !mo.tensor<?, f32>) -> !mo.tensor<[?], f32> {
    // expected-error@below {{'rmo.mo.shape_of' op result #0 must be tensor with integer or index elements, but got '!mo.tensor<[?], f32>'}}
    %1 = rmo.mo.shape_of(%input): (!mo.tensor<?, f32>) -> !mo.tensor<[?], f32>
    mo.output %1: !mo.tensor<[?], f32>
}
