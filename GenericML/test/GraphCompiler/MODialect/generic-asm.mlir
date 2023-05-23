// RUN: mo-opt --mlir-print-op-generic %s | mo-opt | FileCheck %s

// CHECK-LABEL: mo.graph @materialize_linalg_solve
// CHECK-SAME: <[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape -> [[SH2:.*]]: !mosh.ape>(
mo.graph @materialize_linalg_solve<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
  %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>
// CHECK-SAME: ) -> (!mo.tensor<[[SH2]], f32>, !mo.tensor<[[SH1]], f32>) {
) -> (!mo.tensor<Sh2, f32>, !mo.tensor<Sh1, f32>) {
  // CHECK-NEXT: mo.output<:!mosh.ape [[SH0]]> %[[ARG0]], %[[ARG1]] : !mo.tensor<[[SH0]], f32>, !mo.tensor<[[SH1]], f32>
  mo.output<:!mosh.ape Sh0> %arg0, %arg1 : !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>
}
