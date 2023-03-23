// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @rebinds_with_same_input<[[N:.*]], [[M:.*]], [[SH0:.*]]: !mo.shape, [[SH1:.*]]: !mo.shape>(
mo.graph @rebinds_with_same_input<N, M, Sh0: !mo.shape, Sh1: !mo.shape>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[?, ?], f32>
  %arg0: !mo.tensor<[?, ?], f32>
// CHECK-SAME: ) -> (!mo.tensor<[[SH0]], f32>, !mo.tensor<[[[N]], [[M]]], f32>, !mo.tensor<[[SH1]], f32>) {
) -> (!mo.tensor<Sh0, f32>, !mo.tensor<[N, M], f32>, !mo.tensor<Sh1, f32>) {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.rebind(%[[ARG0]]) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[[SH0]], f32>
  %0 = mo.rebind(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<Sh0, f32>
  // CHECK-NEXT: %[[VAL1:.*]] = mo.rebind(%[[VAL0]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[[N]], [[M]]], f32>
  %1 = mo.rebind(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N, M], f32>
  // CHECK-NEXT: %[[VAL2:.*]] = mo.rebind(%[[VAL0]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[SH1]], f32>
  %2 = mo.rebind(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<Sh1, f32>
  // CHECK-NEXT: mo.output
  mo.output %0, %1, %2 : !mo.tensor<Sh0, f32>, !mo.tensor<[N, M], f32>, !mo.tensor<Sh1, f32>
}

// CHECK: mo.graph @rebind_param_materialization<[[A:.*]], [[B:.*]], [[SH0:.*]]: !mo.shape ->  [[N:.*]],  [[M:.*]],  [[SH1:.*]]: !mo.shape>(
mo.graph @rebind_param_materialization<A, B, Sh0: !mo.shape -> N, M, Sh1: !mo.shape>(
  // CHECK-SAME: %arg0: !mo.tensor<[[[A]], [[B]]], f32>, %arg1: !mo.tensor<[[SH0]], f32>
  %arg0 : !mo.tensor<[A, B], f32>, %arg1 : !mo.tensor<Sh0, f32>
// CHECK-SAME: ) -> (!mo.tensor<[[[N]], [[M]]], f32>, !mo.tensor<[[SH1]], f32>) {
) -> (!mo.tensor<[N, M], f32>,!mo.tensor<Sh1, f32>) {
  // CHECK-DAG: kgen.param.declare [[K:.*]] = <[[A]]>
  // CHECK-DAG: kgen.param.declare [[L:.*]] = <[[B]]>
  // CHECK-DAG: %[[VAL0:.*]] = mo.rebind(%arg0) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[[[K]], [[L]]], f32>
  %0 = mo.rebind<() -> K, L>(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[K, L], f32>
  // CHECK-DAG: kgen.param.declare [[SH2:.*]]: !mo.shape = <[[SH0]]>
  // CHECK-DAG: %[[VAL1:.*]] = mo.rebind(%arg1) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[SH2]], f32>
  %1 = mo.rebind<() -> Sh2: !mo.shape>(%arg1) : !mo.tensor<Sh0, f32> -> !mo.tensor<Sh2, f32>
  // CHECK: mo.output<[[K]], [[L]], :!mo.shape [[SH2]]> %[[VAL0]], %[[VAL1]] : !mo.tensor<[[[K]], [[L]]], f32>, !mo.tensor<[[SH2]], f32>
  mo.output<K, L, :!mo.shape Sh2> %0, %1 : !mo.tensor<[K, L], f32>, !mo.tensor<Sh2, f32>
}

// CHECK: mo.graph @rebind_no_fold<[[N:.*]], [[M:.*]], [[SH:.*]]: !mo.shape>(
mo.graph @rebind_no_fold<N, M, Sh: !mo.shape>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[?], f32>, %[[ARG1:.*]]: !mo.tensor<[?, ?], f32>
  %arg0: !mo.tensor<[?], f32>, %arg1: !mo.tensor<[?, ?], f32>
// CHECK-SAME: ) -> (!mo.tensor<[[SH]], f32>, !mo.tensor<[[[N]], [[M]]], f32>) {
) -> (!mo.tensor<Sh, f32>, !mo.tensor<[N, M], f32>) {
  // CHECK-DAG: mo.rebind(%[[ARG0]]) : !mo.tensor<[?], f32> -> !mo.tensor<[[SH]], f32>
  %0 = mo.rebind(%arg0) : !mo.tensor<[?], f32> -> !mo.tensor<Sh, f32>
  // CHECK-DAG: mo.rebind(%[[ARG1]]) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[[[N]], [[M]]], f32>
  %1 = mo.rebind(%arg1) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N, M], f32>
  // CHECK-NEXT: mo.output
  mo.output %0, %1 : !mo.tensor<Sh, f32>, !mo.tensor<[N, M], f32>
}
