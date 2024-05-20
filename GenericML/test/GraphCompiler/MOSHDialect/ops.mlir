// RUN: mosh-opt %s | mosh-opt | FileCheck %s

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

// CHECK: kgen.func @get_dim(%[[SH0:.*]]: !mosh.ape, %[[DIM:.*]]: index) -> (index, index) {
kgen.func @get_dim(%sh0: !mosh.ape, %dim: index) -> (index, index) {
  // CHECK: %[[VAL0:.*]] = mosh.get_dim(%[[SH0]])[%[[DIM]]]
  %0 = mosh.get_dim(%sh0)[%dim]
  kgen.return %0, %0 : index, index
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

// CHECK: kgen.func @slice(%[[SH0:.*]]: !mosh.ape, %[[START:.*]]: index, %[[END:.*]]: index) -> !mosh.ape {
kgen.func @slice(%sh0: !mosh.ape, %start: index, %end: index) -> !mosh.ape {
  // CHECK: %[[VAL0:.*]] = mosh.slice(%[[SH0]])[%[[START]], %[[END]]]
  %sh1 = mosh.slice(%sh0)[%start, %end]
  kgen.return %sh1 : !mosh.ape
}

// CHECK: kgen.generator @param_to_value<[[D:.*]]>() -> (index, !mosh.ape) {
kgen.generator @param_to_value<D>() -> (index, !mosh.ape) {

  // CHECK-NEXT: %[[IDX:.*]]   = mosh.param.to_value = <[[D]]>
  // CHECK-NEXT: %[[SHAPE:.*]] = mosh.param.to_value: !mosh.ape = <[[[D]], 42]>
  %idx   = mosh.param.to_value = <D>
  %shape = mosh.param.to_value: !mosh.ape = <[D, 42]>

  // CHECK: kgen.return %[[IDX]], %[[SHAPE]] : index, !mosh.ape
  kgen.return %idx, %shape : index, !mosh.ape
}
