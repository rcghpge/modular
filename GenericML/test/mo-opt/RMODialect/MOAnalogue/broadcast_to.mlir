// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @broadcast_to_ok<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%data: !mo.tensor<Sh0, si64>, %shape: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], si64> {
    %1 = rmo.mo.broadcast_to<() -> D1>(%data, %shape) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D1], si64>
    mo.output<D1> %1: !mo.tensor<[D1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_ok<[[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape -> [[D0:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[D0]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.broadcast_to<() -> [[D1:[^ ]*]]>([[VARG0]], [[VARG1]]) : (!mo.tensor<[[SH0]], si64>, !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[D1]]], si64>
// CHECK-NEXT: mo.output<[[D1]]> [[V0]] : !mo.tensor<[[[D1]]], si64>

// -----

mo.graph @broadcast_to_ok_f32<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%data: !mo.tensor<Sh0, f32>, %shape: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], f32> {
    %1 = rmo.mo.broadcast_to<() -> D1>(%data, %shape) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D1], f32>
    mo.output<D1> %1: !mo.tensor<[D1], f32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_to_ok_f32<[[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape -> [[D0:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[D0]]], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.broadcast_to<() -> [[D1:[^ ]*]]>([[VARG0]], [[VARG1]]) : (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[D1]]], f32>
// CHECK-NEXT: mo.output<[[D1]]> [[V0]] : !mo.tensor<[[[D1]]], f32>

// -----

mo.graph @broadcast_to_not_ok<Sh0: !mosh.ape, Sh1: !mosh.ape>(%data: !mo.tensor<Sh0, si64>, %shape: !mo.tensor<Sh1, f32>) -> !mo.tensor<[D0], si64> {
    // expected-error@below {{'rmo.mo.broadcast_to' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<Sh1, f32>'}}
    %1 = rmo.mo.broadcast_to(%data, %shape) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, f32>) -> !mo.tensor<[D0], si64>
    mo.output %1: !mo.tensor<[D0], si64>
}
