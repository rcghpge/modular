// RUN: mo-opt %s | mo-opt | FileCheck %s

// CHECK: kgen.func @assert(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) {
kgen.func @assert(%arg0: index, %arg1: index) {
  // CHECK: %[[VAL0:.*]] = index.cmp eq(%[[ARG0]], %[[ARG1]])
  %0 = index.cmp eq(%arg0, %arg1)
  // CHECK: mo.shape.assert %[[VAL0]]
  mo.shape.assert %0
  // CHECK: mo.shape.assert %[[VAL0]], "args are not equal"
  mo.shape.assert %0, "args are not equal"
  kgen.return
}

// CHECK: kgen.func @concat(%[[ARG0:.*]]: !mo.shape, %[[ARG1:.*]]: !mo.shape) -> !mo.shape {
kgen.func @concat(%sh0: !mo.shape, %sh1: !mo.shape) -> !mo.shape {
  // CHECK: %[[VAL0:.*]] = mo.shape.concat(%[[ARG0]], %[[ARG1]])
  %0 = mo.shape.concat(%sh0, %sh1)
  kgen.return %0 : !mo.shape
}

// CHECK: kgen.func @equal(%[[ARG0:.*]]: !mo.shape, %[[ARG1:.*]]: !mo.shape) -> i1 {
kgen.func @equal(%sh0: !mo.shape, %sh1: !mo.shape) -> i1 {
  // CHECK: %[[VAL0:.*]] = mo.shape.eq(%[[ARG0]], %[[ARG1]])
  %0 = mo.shape.eq(%sh0, %sh1)
  kgen.return %0 : i1
}

// CHECK: kgen.func @get_dim(%[[SH0:.*]]: !mo.shape) -> (index, index) {
kgen.func @get_dim(%sh0: !mo.shape) -> (index, index) {
  // CHECK: %[[VAL0:.*]] = mo.shape.get_dim(%[[SH0]])[0]
  %0 = mo.shape.get_dim(%sh0)[0]
  // CHECK: %[[VAL0:.*]] = mo.shape.get_dim(%[[SH0]])[-1]
  %1 = mo.shape.get_dim(%sh0)[-1]
  kgen.return %0, %1 : index, index
}

// CHECK: kgen.func @get_rank(%[[SH0:.*]]: !mo.shape) -> index {
kgen.func @get_rank(%sh0: !mo.shape) -> index {
  // CHECK: %[[VAL0:.*]] = mo.shape.get_rank(%[[SH0]])
  %0 = mo.shape.get_rank(%sh0)
  kgen.return %0 : index
}

// CHECK: kgen.func @new_dim(%[[D0:.*]]: index, %[[D1:.*]]: index)
kgen.func @new_dim(%d0: index, %d1: index)
    // CHECK-SAME: -> (!mo.shape, !mo.shape, !mo.shape) {
    -> (!mo.shape, !mo.shape, !mo.shape) {
  // CHECK: %[[SH0:.*]] = mo.shape.new()
  %sh0 = mo.shape.new()
  // CHECK: %[[SH1:.*]] = mo.shape.new(%[[D0]])
  %sh1 = mo.shape.new(%d0)
  // CHECK: %[[SH2:.*]] = mo.shape.new(%[[D0]], %[[D1]], %[[D0]])
  %sh2 = mo.shape.new(%d0, %d1, %d0)
  kgen.return %sh0, %sh1, %sh2 : !mo.shape, !mo.shape, !mo.shape
}

// CHECK: kgen.func @slice(%[[SH0:.*]]: !mo.shape) -> (!mo.shape, !mo.shape) {
kgen.func @slice(%sh0: !mo.shape) -> (!mo.shape, !mo.shape) {
  // CHECK-DAG: %[[SH1:.*]] = mo.shape.slice(%[[SH0]])[0, 2]
  %sh1 = mo.shape.slice(%sh0)[0, 2]
  // CHECK-DAG: %[[SH2:.*]] = mo.shape.slice(%[[SH0]])[0, -2]
  %sh2 = mo.shape.slice(%sh0)[0, -2]
  kgen.return %sh1, %sh2 : !mo.shape, !mo.shape
}
