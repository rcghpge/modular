// RUN: mo-opt %s | FileCheck %s

mo.graph @cast_ok<Sh: !mosh.ape -> S0, S1, S2>(%arg0: !mo.tensor<Sh, f32>) -> !mo.tensor<[S0, S1, S2], si32> {
  %1 = rmo.mo.cast<() -> D0, D1, D2>(%arg0) : (!mo.tensor<Sh, f32>) -> !mo.tensor<[D0, D1, D2], si32>
  mo.output<D0, D1, D2> %1 : !mo.tensor<[D0, D1, D2], si32>
}
// CHECK-LABEL: mo.graph
// CHECK-SAME: @cast_ok<[[SH:[^:]*]]: !mosh.ape -> [[S0:[^,]*]], [[S1:[^,]*]], [[S2:[^>]*]]>([[VARG0:[^ ]*]]: !mo.tensor<[[SH]], f32>) -> !mo.tensor<{{\[}}[[S0]], [[S1]], [[S2]]{{\]}}, si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.cast<() -> [[D0:[^,]*]], [[D1:[^,]*]], [[D2:[^>]*]]>([[VARG0]]) : (!mo.tensor<[[SH]], f32>) -> !mo.tensor<{{\[}}[[D0]], [[D1]], [[D2]]{{\]}}, si32>
// CHECK-NEXT: mo.output<[[D0]], [[D1]], [[D2]]> [[V0]] : !mo.tensor<{{\[}}[[D0]], [[D1]], [[D2]]{{\]}}, si32>
