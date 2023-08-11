// RUN: mo-opt %s --canonicalize -cse | FileCheck %s

// CHECK: mo.graph @bcast_low_rank<[[A:.*]], [[B:.*]], [[C:.*]] -> [[D:.*]], [[E:.*]], [[F:.*]]>(
mo.graph @bcast_low_rank<A, B, C -> D, E, F>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[], f32>, %[[NUMLOWER:.*]]: !mo.tensor<[], si32>, %[[NUMUPPER:.*]]: !mo.tensor<[], si32>, %[[EXCLUDE:.*]]: !mo.tensor<[], bool>
  %arg0: !mo.tensor<[], f32>, %numLower: !mo.tensor<[], si32>, %numUpper: !mo.tensor<[], si32>, %exclude: !mo.tensor<[], bool>
// CHECK-SAME: ) -> !mo.tensor<[[[D]], [[E]], [[F]]], f32> no_inline {
) -> !mo.tensor<[D, E, F], f32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[B]], [[C]]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<[A, B, C], f32>
  // CHECK-NEXT: %[[VAL1:.*]] = mo.linalg.band_part(%[[VAL0]], %[[NUMLOWER]], %[[NUMUPPER]], %[[EXCLUDE]]) :
  // CHECK-SAME: (!mo.tensor<[[[B]], [[C]]], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[[[B]], [[C]]], f32>
  %1 = mo.linalg.band_part(%0, %numLower, %numUpper, %exclude) : (!mo.tensor<[A, B, C], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[A, B, C], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[VAL1]]) : !mo.tensor<[[[B]], [[C]]], f32> -> !mo.tensor<[[[A]], [[B]], [[C]]], f32>
  // CHECK-NEXT: mo.output<[[A]], [[B]], [[C]]> %[[RES]] : !mo.tensor<[[[A]], [[B]], [[C]]], f32>
  mo.output<A, B, C> %1 : !mo.tensor<[A, B, C], f32>
}

// CHECK: mo.graph @bcast_batch_dims<[[A:.*]], [[B:.*]], [[C:.*]] -> [[D:.*]]>(
mo.graph @bcast_batch_dims<A, B, C -> D>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], 1, 1], f32>, %[[NUMLOWER:.*]]: !mo.tensor<[], si32>, %[[NUMUPPER:.*]]: !mo.tensor<[], si32>, %[[EXCLUDE:.*]]: !mo.tensor<[], bool>
  %arg0: !mo.tensor<[A, 1, 1], f32>, %numLower: !mo.tensor<[], si32>, %numUpper: !mo.tensor<[], si32>, %exclude: !mo.tensor<[], bool>
// CHECK-SAME: ) -> !mo.tensor<[[[D]], [[D]], [[D]], 5], f32> no_inline {
) -> !mo.tensor<[D, D, D, 5], f32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[[A]], 1, 1], f32> -> !mo.tensor<[[[A]], [[B]], 5], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, 1, 1], f32> -> !mo.tensor<[B, B, B, 5], f32>
  // CHECK-NEXT: %[[VAL1:.*]] = mo.linalg.band_part(%[[VAL0]], %[[NUMLOWER]], %[[NUMUPPER]], %[[EXCLUDE]]) :
  // CHECK-SAME: (!mo.tensor<[[[A]], [[B]], 5], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[[[A]], [[B]], 5], f32>
  %1 = mo.linalg.band_part(%0, %numLower, %numUpper, %exclude) : (!mo.tensor<[B, B, B, 5], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[B, B, B, 5], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[VAL1]]) : !mo.tensor<[[[A]], [[B]], 5], f32> -> !mo.tensor<[[[B]], [[B]], [[B]], 5], f32>
  // CHECK-NEXT: mo.output<[[B]]> %[[RES]] : !mo.tensor<[[[B]], [[B]], [[B]], 5], f32>
  mo.output<B> %1 : !mo.tensor<[B, B, B, 5], f32>
}

// CHECK: mo.graph @multiple_users<[[A:.*]], [[B:.*]], [[C:.*]] -> [[D:.*]]>(
mo.graph @multiple_users<A, B, C -> D>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], 1, 1], f32>, %[[NUMLOWER:.*]]: !mo.tensor<[], si32>, %[[NUMUPPER:.*]]: !mo.tensor<[], si32>, %[[EXCLUDE:.*]]: !mo.tensor<[], bool>
  %arg0: !mo.tensor<[A, 1, 1], f32>, %numLower: !mo.tensor<[], si32>, %numUpper: !mo.tensor<[], si32>, %exclude: !mo.tensor<[], bool>
// CHECK-SAME: ) -> (!mo.tensor<[[[D]], [[D]], [[D]], 5], f32>, !mo.tensor<[[[D]], [[D]], [[D]], 5], f32>) no_inline {
) -> (!mo.tensor<[D, D, D, 5], f32>, !mo.tensor<[D, D, D, 5], f32>) {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[[A]], 1, 1], f32> -> !mo.tensor<[[[A]], [[B]], 5], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, 1, 1], f32> -> !mo.tensor<[B, B, B, 5], f32>

  // CHECK-DAG: %[[VAL1:.*]] = mo.linalg.band_part(%[[VAL0]], %[[NUMLOWER]], %[[NUMUPPER]], %[[EXCLUDE]]) :
  %1 = mo.linalg.band_part(%0, %numLower, %numUpper, %exclude) : (!mo.tensor<[B, B, B, 5], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[B, B, B, 5], f32>
  // CHECK-DAG: %[[RES1:.*]] = mo.static.broadcast_to(%[[VAL1]])

  // CHECK-DAG: %[[VAL2:.*]] = mo.linalg.band_part(%[[VAL0]], %[[NUMLOWER]], %[[NUMUPPER]], %[[EXCLUDE]]) {someAttr = "foo"}
  %2 = mo.linalg.band_part(%0, %numLower, %numUpper, %exclude) {someAttr = "foo"} : (!mo.tensor<[B, B, B, 5], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[B, B, B, 5], f32>
  // CHECK-DAG: %[[RES2:.*]] = mo.static.broadcast_to(%[[VAL2]])

  // CHECK-NEXT: mo.output<[[B]]> %[[RES1]], %[[RES2]] : !mo.tensor<[B, B, B, 5], f32>, !mo.tensor<[B, B, B, 5], f32>
  mo.output<B> %1, %2 : !mo.tensor<[B, B, B, 5], f32>, !mo.tensor<[B, B, B, 5], f32>
}

// CHECK: mo.graph @no_match_dyn_rank_bcast_in<[[A:.*]], [[B:.*]], [[C:.*]], [[SH0:.*]]: !mosh.ape -> [[D:.*]], [[E:.*]], [[F:.*]]>(
mo.graph @no_match_dyn_rank_bcast_in<A, B, C, Sh0: !mosh.ape -> D, E, F>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[SH0]], f32>, %[[NUMLOWER:.*]]: !mo.tensor<[], si32>, %[[NUMUPPER:.*]]: !mo.tensor<[], si32>, %[[EXCLUDE:.*]]: !mo.tensor<[], bool>
  %arg0: !mo.tensor<Sh0, f32>, %numLower: !mo.tensor<[], si32>, %numUpper: !mo.tensor<[], si32>, %exclude: !mo.tensor<[], bool>
// CHECK-SAME: ) -> !mo.tensor<[[[D]], [[E]], [[F]]], f32> no_inline {
) -> !mo.tensor<[D, E, F], f32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[SH0]], f32> -> !mo.tensor<[[[A]], [[B]], [[C]]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<Sh0, f32> -> !mo.tensor<[A, B, C], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.linalg.band_part(%[[VAL0]], %[[NUMLOWER]], %[[NUMUPPER]], %[[EXCLUDE]]) :
  // CHECK-SAME: (!mo.tensor<[[[A]], [[B]], [[C]]], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[[[A]], [[B]], [[C]]], f32>
  %1 = mo.linalg.band_part(%0, %numLower, %numUpper, %exclude) : (!mo.tensor<[A, B, C], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[A, B, C], f32>
  // CHECK-NEXT: mo.output<[[A]], [[B]], [[C]]> %[[RES]] : !mo.tensor<[[[A]], [[B]], [[C]]], f32>
  mo.output<A, B, C> %1 : !mo.tensor<[A, B, C], f32>
}

// CHECK: mo.graph @no_match_dyn_rank<[[A:.*]], [[B:.*]], [[C:.*]], [[SH0:.*]]: !mosh.ape -> [[SH1:.*]]: !mosh.ape>(
mo.graph @no_match_dyn_rank<A, B, C, Sh0: !mosh.ape -> Sh1: !mosh.ape>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[[[A]], [[B]], [[C]]], f32>, %[[NUMLOWER:.*]]: !mo.tensor<[], si32>, %[[NUMUPPER:.*]]: !mo.tensor<[], si32>, %[[EXCLUDE:.*]]: !mo.tensor<[], bool>
  %arg0: !mo.tensor<[A, B, C], f32>, %numLower: !mo.tensor<[], si32>, %numUpper: !mo.tensor<[], si32>, %exclude: !mo.tensor<[], bool>
// CHECK-SAME: ) -> !mo.tensor<[[SH1]], f32> no_inline {
) -> !mo.tensor<Sh1, f32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[[[A]], [[B]], [[C]]], f32> -> !mo.tensor<[[SH0]], f32>
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[A, B, C], f32> -> !mo.tensor<Sh0, f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.linalg.band_part(%[[VAL0]], %[[NUMLOWER]], %[[NUMUPPER]], %[[EXCLUDE]]) :
  // CHECK-SAME: (!mo.tensor<[[SH0]], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[[SH0]], f32>
  %1 = mo.linalg.band_part(%0, %numLower, %numUpper, %exclude) : (!mo.tensor<Sh0, f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<Sh0, f32>
  // CHECK-NEXT: mo.output<:!mosh.ape [[SH0]]> %1 : !mo.tensor<[[SH0]], f32>
  mo.output<:!mosh.ape Sh0> %1 : !mo.tensor<Sh0, f32>
}

// CHECK: mo.graph @intermediate_variable_dominance<[[A:.*]], [[B:.*]], [[C:.*]] -> [[D:.*]], [[E:.*]], [[F:.*]]>(
mo.graph @intermediate_variable_dominance<A, B, C -> D, E, F>(
  // CHECK-SAME: %[[ARG0:.*]]: !mo.tensor<[], f32>, %[[NUMLOWER:.*]]: !mo.tensor<[], si32>, %[[EXCLUDE:.*]]: !mo.tensor<[], bool>
  %arg0: !mo.tensor<[], f32>, %numLower: !mo.tensor<[], si32>, %exclude: !mo.tensor<[], bool>
// CHECK-SAME: ) -> !mo.tensor<[[[D]], [[E]], [[F]]], f32> no_inline {
) -> !mo.tensor<[D, E, F], f32> {
  // Note that new instructions are inserted after `%numUpper`, before `%res`,
  // instead of before `%bcast`.
  // CHECK-NEXT: %[[NUMUPPER:.*]] = mo.add(%[[NUMLOWER]], %[[NUMLOWER]]) : !mo.tensor<[], si32>
  // CHECK-NEXT: %[[VAL0:.*]] = mo.static.broadcast_to(%[[ARG0]]) : !mo.tensor<[], f32> -> !mo.tensor<[[[B]], [[C]]], f32>
  %bcast = mo.static.broadcast_to(%arg0) : !mo.tensor<[], f32> -> !mo.tensor<[A, B, C], f32>
  %numUpper = mo.add(%numLower, %numLower) : !mo.tensor<[], si32>

  // CHECK-NEXT: %[[VAL1:.*]] = mo.linalg.band_part(%[[VAL0]], %[[NUMLOWER]], %[[NUMUPPER]], %[[EXCLUDE]]) :
  // CHECK-SAME: (!mo.tensor<[[[B]], [[C]]], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[[[B]], [[C]]], f32>
  %res = mo.linalg.band_part(%bcast, %numLower, %numUpper, %exclude) : (!mo.tensor<[A, B, C], f32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], bool>) -> !mo.tensor<[A, B, C], f32>
  // CHECK-NEXT: %[[RES:.*]] = mo.static.broadcast_to(%[[VAL1]]) : !mo.tensor<[[[B]], [[C]]], f32> -> !mo.tensor<[[[A]], [[B]], [[C]]], f32>
  // CHECK-NEXT: mo.output<[[A]], [[B]], [[C]]> %[[RES]] : !mo.tensor<[[[A]], [[B]], [[C]]], f32>
  mo.output<A, B, C> %res : !mo.tensor<[A, B, C], f32>
}
