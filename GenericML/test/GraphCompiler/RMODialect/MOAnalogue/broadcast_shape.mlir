// RUN: mo-opt %s --split-input-file --verify-diagnostics | FileCheck %s

mo.graph @broadcast_shape_ok<Sh0: !mosh.ape, Sh1: !mosh.ape -> D0>(%shape0: !mo.tensor<Sh0, si64>, %shape1: !mo.tensor<Sh1, si64>) -> !mo.tensor<[D0], si64> {
    %1 = rmo.mo.broadcast_shape<() -> D1>(%shape0, %shape1) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, si64>) -> !mo.tensor<[D1], si64>
    mo.output<D1> %1: !mo.tensor<[D1], si64>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @broadcast_shape_ok<[[SH0:[^ ]*]]: !mosh.ape, [[SH1:[^ ]*]]: !mosh.ape -> [[D0:[^ ]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH0]], si64>, [[VARG1:[^ ]*]]: !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[D0]]], si64> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.broadcast_shape<() -> [[D1:[^ ]*]]>([[VARG0]], [[VARG1]]) : (!mo.tensor<[[SH0]], si64>, !mo.tensor<[[SH1]], si64>) -> !mo.tensor<[[[D1]]], si64>
// CHECK-NEXT: mo.output<[[D1]]> [[V0]] : !mo.tensor<[[[D1]]], si64>

// -----

mo.graph @broadcast_shape_not_ok<Sh0: !mosh.ape, Sh1: !mosh.ape>(%shape0: !mo.tensor<Sh0, si64>, %shape1: !mo.tensor<Sh1, f32>) -> !mo.tensor<[D0], si64> {
    // expected-error@below {{'rmo.mo.broadcast_shape' op operand #1 must be rank 1 or unranked with signed 64 or signed 32 integer elements, but got '!mo.tensor<Sh1, f32>'}}
    %1 = rmo.mo.broadcast_shape(%shape0, %shape1) : (!mo.tensor<Sh0, si64>, !mo.tensor<Sh1, f32>) -> !mo.tensor<[D0], si64>
    mo.output %1: !mo.tensor<[D0], si64>
}
