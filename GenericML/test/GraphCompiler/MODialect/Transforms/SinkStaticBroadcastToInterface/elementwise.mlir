// RUN: mo-opt %s --canonicalize -cse | FileCheck %s

// CHECK: mo.graph @binary_static_rank<[[A:.*]], [[B:.*]] -> [[C:.*]], [[D:.*]]>(
mo.graph @binary_static_rank<A, B -> C, D>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>, %[[ARG1:.*]]: !mo.tensor<[[[A]], [[B]]], f32>
  %arg0: !mo.tensor<[A, B], f32>, %arg1: !mo.tensor<[A, B], f32>
// CHECK-SAME: ) -> !mo.tensor<[2, [[C]], [[D]]], f32> {
) -> !mo.tensor<[2, C, D], f32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, A, B], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.sub(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[[[A]], [[B]]], f32>
  %2 = mo.sub(%0, %1) : !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[2, [[A]], [[B]]], f32>
  // CHECK-NEXT: mo.output<[[A]], [[B]]> %[[RESBCAST]] : !mo.tensor<[2, [[A]], [[B]]], f32>
  mo.output<A, B> %2 : !mo.tensor<[2, A, B], f32>
}

// CHECK: mo.graph @binary_no_common_shape<[[A:.*]] -> [[B:.*]]>(
mo.graph @binary_no_common_shape<A -> B>(
  // CHECK-SAME: %[[ARG0]]: !mo.tensor<[[[A]]], f32>, %[[ARG1]]: !mo.tensor<[2, [[A]]], f32>
  %arg0: !mo.tensor<[A], f32>, %arg1: !mo.tensor<[2, A], f32>
// CHECK-SAME: ) -> !mo.tensor<[3, 2, [[B]]], f32> {
) -> !mo.tensor<[3, 2, B], f32>{
  // CHECK-DAG: %[[ARG0BCAST:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[[A]]], f32> -> !mo.tensor<[3, 2, [[A]]], f32>
  // CHECK-DAG: %[[ARG1BCAST:.*]] = mo.static.broadcast_to(%[[ARG1]]) : !mo.tensor<[2, [[A]]], f32> -> !mo.tensor<[3, 2, [[A]]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A], f32> -> !mo.tensor<[3, 2, A], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[2, A], f32> -> !mo.tensor<[3, 2, A], f32>
  // CHECK: %[[RES:.*]] = mo.add(%[[ARG0BCAST]], %[[ARG1BCAST]]) : !mo.tensor<[3, 2, [[A]]], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[3, 2, A], f32>
  // CHECK: mo.output<[[A]]> %[[RES]] : !mo.tensor<[3, 2, [[A]]], f32>
  mo.output<A> %2 : !mo.tensor<[3, 2, A], f32>
}

// CHECK: mo.graph @binary_dyn_rank_common<[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape -> [[SH2:.*]]: !mosh.ape>(
mo.graph @binary_dyn_rank_common<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH0]], f32>
  %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh0, f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH2]], f32> {
) -> !mo.tensor<Sh2, f32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<Sh1, f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<Sh0, f32> -> !mo.tensor<Sh1, f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.mul(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[[SH0]], f32>
  %2 = mo.mul(%0, %1) : !mo.tensor<Sh1, f32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[SH1]], f32>
  // CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> %[[RESBCAST]] : !mo.tensor<[[SH1]], f32>
  mo.output<:!mosh.ape Sh1> %2 : !mo.tensor<Sh1, f32>
}

// CHECK: mo.graph @binary_dyn_rank_not_common<[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape, [[SH2:.*]]: !mosh.ape -> [[SH3:.*]]: !mosh.ape>(
mo.graph @binary_dyn_rank_not_common<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape -> Sh3: !mosh.ape>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[[SH1]], f32>
  %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<Sh1, f32>
// CHECK-SAME: ) -> !mo.tensor<[[SH3]], f32> {
) -> !mo.tensor<Sh3, f32>{
  // CHECK-DAG: %[[ARG0BCAST:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[SH2]], f32>
  // CHECK-DAG: %[[ARG1BCAST:.*]] = mo.static.broadcast_to(%[[ARG1]]) : !mo.tensor<[[SH1]], f32> -> !mo.tensor<[[SH2]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<Sh2, f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<Sh1, f32> -> !mo.tensor<Sh2, f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.max(%[[ARG0BCAST]], %[[ARG1BCAST]]) : !mo.tensor<[[SH2]], f32>
  %2 = mo.max(%0, %1) : !mo.tensor<Sh2, f32>
  // CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> %[[RES]] : !mo.tensor<[[SH2]], f32>
  mo.output<:!mosh.ape Sh2> %2 : !mo.tensor<Sh2, f32>
}

// CHECK: mo.graph @binary_static_dyn_rank_mixed<[[SH0:.*]]: !mosh.ape>(
mo.graph @binary_static_dyn_rank_mixed<Sh0: !mosh.ape>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[ARG1:.*]]: !mo.tensor<[1, 3], f32>
  %arg0: !mo.tensor<Sh0, f32>, %arg1: !mo.tensor<[1, 3], f32>
// CHECK-SAME: ) -> !mo.tensor<[2, 3], f32> {
) -> !mo.tensor<[2, 3], f32> {
  // CHECK-DAG: %[[ARG0BCAST:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[2, 3], f32>
  // CHECK-DAG: %[[ARG1BCAST:.*]] = mo.static.broadcast_to(%[[ARG1]]) : !mo.tensor<[1, 3], f32> -> !mo.tensor<[2, 3], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<[2, 3], f32>
  %1 = mo.static.broadcast_to(%arg1) : !mo.tensor<[1, 3], f32> -> !mo.tensor<[2, 3], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.min(%[[ARG0BCAST]], %[[ARG1BCAST]]) : !mo.tensor<[2, 3], f32>
  %2 = mo.min(%0, %1) : !mo.tensor<[2, 3], f32>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], f32>
  mo.output %2 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @unary_static_rank<[[A:.*]], [[B:.*]] -> [[C:.*]], [[D:.*]]>(
mo.graph @unary_static_rank<A, B -> C, D>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]]], f32>) -> !mo.tensor<[2, [[C]], [[D]]], f32> {
    %arg0: !mo.tensor<[A, B], f32>) -> !mo.tensor<[2, C, D], f32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B], f32> -> !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.exp(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]]], f32>
  %1 = mo.exp(%0) : !mo.tensor<[2, A, B], f32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[[[A]], [[B]]], f32> -> !mo.tensor<[2, [[A]], [[B]]], f32>
  // CHECK-NEXT: mo.output<[[A]], [[B]]> %[[RESBCAST]] : !mo.tensor<[2, [[A]], [[B]]], f32>
  mo.output<A, B> %1 : !mo.tensor<[2, A, B], f32>
}

// CHECK: mo.graph @unary_dyn_rank<[[SH0:.*]]: !mosh.ape, [[SH1:.*]]: !mosh.ape -> [[SH2:.*]]: !mosh.ape>(
mo.graph @unary_dyn_rank<Sh0: !mosh.ape, Sh1: !mosh.ape -> Sh2: !mosh.ape>(
    // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>) -> !mo.tensor<[[SH2]], f32> {
    %arg0: !mo.tensor<Sh0, f32>) -> !mo.tensor<Sh2, f32>{
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<Sh1, f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.relu(%[[ARG0]]) : !mo.tensor<[[SH0]], f32>
  %2 = mo.relu(%0) : !mo.tensor<Sh1, f32>
  // CHECK-NEXT: %[[RESBCAST:.*]] = mo.static.broadcast_to(%[[RES]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[SH1]], f32>
  // CHECK-NEXT: mo.output<:!mosh.ape [[SH1]]> %[[RESBCAST]] : !mo.tensor<[[SH1]], f32>
  mo.output<:!mosh.ape Sh1> %2 : !mo.tensor<Sh1, f32>
}
