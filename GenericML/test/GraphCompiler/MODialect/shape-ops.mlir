// RUN: mo-opt %s | mo-opt | FileCheck %s

// CHECK: kgen.func @get_dim(%[[SH0:.*]]: !mo.shape) -> index {
kgen.func @get_dim(%sh0: !mo.shape) -> index {
  // CHECK: %[[VAL0:.*]] = mo.shape.get_dim(%[[SH0]])[0]
  %0 = mo.shape.get_dim(%sh0)[0]
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

// CHECK: kgen.func @get_rank(%[[SH0:.*]]: !mo.shape) -> index {
kgen.func @get_rank(%sh0: !mo.shape) -> index {
  // CHECK: %[[VAL0:.*]] = mo.shape.get_rank(%[[SH0]])
  %0 = mo.shape.get_rank(%sh0)
  kgen.return %0 : index
}
