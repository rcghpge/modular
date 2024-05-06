// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @reshape_op_ok(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<?, si64>) -> !mo.tensor<[?], si64> {
    %1 = rmo.mo.reshape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @reshape_op_ok([[VARG0:[^ ]*]]: !mo.tensor<?, si64>, [[VARG1:[^ ]*]]: !mo.tensor<?, si64>) -> !mo.tensor<[?], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.reshape([[VARG0]], [[VARG1]]) : (!mo.tensor<?, si64>, !mo.tensor<?, si64>) -> !mo.tensor<[?], si64>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<[?], si64>

// -----

mo.graph @reshape_op_not_ok(%shape0: !mo.tensor<?, si64>, %shape1: !mo.tensor<?, f32>) -> !mo.tensor<[?], si64> {
    // expected-error@below {{'rmo.mo.reshape' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<?, f32>'}}
    %1 = rmo.mo.reshape(%shape0, %shape1) : (!mo.tensor<?, si64>, !mo.tensor<?, f32>) -> !mo.tensor<[?], si64>
    mo.output %1: !mo.tensor<[?], si64>
}
