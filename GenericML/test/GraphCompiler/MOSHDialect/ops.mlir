// RUN: mosh-opt %s | mosh-opt | FileCheck %s

// CHECK: kgen.func @assert(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index) -> (!mosh.chain, !mosh.chain) {
kgen.func @assert(%arg0: index, %arg1: index) -> (!mosh.chain, !mosh.chain) {
  // CHECK: %[[VAL0:.*]] = index.cmp eq(%[[ARG0]], %[[ARG1]])
  %0 = index.cmp eq(%arg0, %arg1)
  // CHECK: mosh.assert %[[VAL0]]
  %ch0 = mosh.assert %0
  // CHECK: mosh.assert %[[VAL0]], "args are not equal"
  %ch1 = mosh.assert %0, "args are not equal"
  kgen.return %ch0, %ch1 : !mosh.chain, !mosh.chain
}

// CHECK: kgen.func @guard(%[[ARG0:.*]]: !mosh.ape, %[[ARG1:.*]]: !mosh.ape, %[[COND:.*]]: i1) -> !mosh.chain {
kgen.func @guard(%sh0: !mosh.ape, %sh1: !mosh.ape, %cond: i1) -> !mosh.chain {
  // CHECK: %[[CH:.*]] = mosh.assert %[[COND]]
  %ch = mosh.assert %cond
  // CHECK: mosh.guard[%[[CH]]] %[[ARG0]], %[[ARG1]] : !mosh.ape, !mosh.ape
  mosh.guard[%ch] %sh0, %sh1 : !mosh.ape, !mosh.ape
  // CHECK: kgen.return %[[CH]] : !mosh.chain
  kgen.return %ch : !mosh.chain
}

// CHECK: kgen.func @broadcast(%[[SH0:.*]]: !mosh.ape, %[[SH1:.*]]: !mosh.ape) -> !mosh.ape {
kgen.func @broadcast(%sh0: !mosh.ape, %sh1: !mosh.ape) -> !mosh.ape {
  // CHECK: %[[RES:.*]] = mosh.broadcast(%[[SH0]], %[[SH1]])
  %0 = mosh.broadcast(%sh0, %sh1)
  kgen.return %0 : !mosh.ape
}

// CHECK: kgen.func @concat(%[[ARG0:.*]]: !mosh.ape, %[[ARG1:.*]]: !mosh.ape) -> !mosh.ape {
kgen.func @concat(%sh0: !mosh.ape, %sh1: !mosh.ape) -> !mosh.ape {
  // CHECK: %[[VAL0:.*]] = mosh.concat(%[[ARG0]], %[[ARG1]])
  %0 = mosh.concat(%sh0, %sh1)
  kgen.return %0 : !mosh.ape
}

// CHECK: kgen.func @equal(%[[ARG0:.*]]: !mosh.ape, %[[ARG1:.*]]: !mosh.ape) -> i1 {
kgen.func @equal(%sh0: !mosh.ape, %sh1: !mosh.ape) -> i1 {
  // CHECK: %[[VAL0:.*]] = mosh.eq(%[[ARG0]], %[[ARG1]])
  %0 = mosh.eq(%sh0, %sh1)
  kgen.return %0 : i1
}

// CHECK: kgen.func @get_dim(%[[SH0:.*]]: !mosh.ape) -> (index, index) {
kgen.func @get_dim(%sh0: !mosh.ape) -> (index, index) {
  // CHECK: %[[VAL0:.*]] = mosh.get_dim(%[[SH0]])[0]
  %0 = mosh.get_dim(%sh0)[0]
  // CHECK: %[[VAL0:.*]] = mosh.get_dim(%[[SH0]])[-1]
  %1 = mosh.get_dim(%sh0)[-1]
  kgen.return %0, %1 : index, index
}

// CHECK: kgen.func @get_rank(%[[SH0:.*]]: !mosh.ape) -> index {
kgen.func @get_rank(%sh0: !mosh.ape) -> index {
  // CHECK: %[[VAL0:.*]] = mosh.get_rank(%[[SH0]])
  %0 = mosh.get_rank(%sh0)
  kgen.return %0 : index
}

// CHECK: kgen.func @new_dim(%[[D0:.*]]: index, %[[D1:.*]]: index)
kgen.func @new_dim(%d0: index, %d1: index)
    // CHECK-SAME: -> (!mosh.ape, !mosh.ape, !mosh.ape) {
    -> (!mosh.ape, !mosh.ape, !mosh.ape) {
  // CHECK: %[[SH0:.*]] = mosh.new()
  %sh0 = mosh.new()
  // CHECK: %[[SH1:.*]] = mosh.new(%[[D0]])
  %sh1 = mosh.new(%d0)
  // CHECK: %[[SH2:.*]] = mosh.new(%[[D0]], %[[D1]], %[[D0]])
  %sh2 = mosh.new(%d0, %d1, %d0)
  kgen.return %sh0, %sh1, %sh2 : !mosh.ape, !mosh.ape, !mosh.ape
}

// CHECK: kgen.func @num_elements(%[[SH0:.*]]: !mosh.ape) -> index {
kgen.func @num_elements(%sh0: !mosh.ape) -> index {
  // CHECK: %[[VAL0:.*]] = mosh.num_elements(%[[SH0]])
  %0 = mosh.num_elements(%sh0)
  kgen.return %0 : index
}

// CHECK: kgen.func @slice(%[[SH0:.*]]: !mosh.ape) -> (!mosh.ape, !mosh.ape) {
kgen.func @slice(%sh0: !mosh.ape) -> (!mosh.ape, !mosh.ape) {
  // CHECK-DAG: %[[SH1:.*]] = mosh.slice(%[[SH0]])[0, 2]
  %sh1 = mosh.slice(%sh0)[0, 2]
  // CHECK-DAG: %[[SH2:.*]] = mosh.slice(%[[SH0]])[0, -2]
  %sh2 = mosh.slice(%sh0)[0, -2]
  // CHECK-DAG: %[[SH3:.*]] = mosh.slice(%[[SH0]])[1]
  %sh3 = mosh.slice(%sh0)[1]
  // CHECK-DAG: %[[SH4:.*]] = mosh.slice(%[[SH0]])[-2]
  %sh4 = mosh.slice(%sh0)[-2]
  kgen.return %sh1, %sh2 : !mosh.ape, !mosh.ape
}
