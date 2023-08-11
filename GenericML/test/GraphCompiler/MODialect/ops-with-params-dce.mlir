// RUN: mo-opt %s --canonicalize | mo-opt | FileCheck %s


// CHECK: mo.graph @ops_with_params_no_dce(%[[ARG0:.*]]: !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32> no_inline {
mo.graph @ops_with_params_no_dce(%arg0: !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32> {
  // CHECK-NEXT: %[[VAL0:.*]] = mo.rebind<() -> [[D0:.*]]>(%[[ARG0]])
  // CHECK-SAME: : !mo.tensor<[?], si32> -> !mo.tensor<[[[D0]]], si32>
  %0 = mo.rebind<() -> D0>(%arg0) : !mo.tensor<[?], si32> -> !mo.tensor<[D0], si32>
  // CHECK-NEXT: %[[VAL1:.*]] = mo.broadcast_to<() -> [[SH0:.*]]: !mosh.ape>(%[[ARG0]], %[[ARG0]])
  %1 = mo.broadcast_to<() -> Sh0: !mosh.ape>(%arg0, %arg0) : (
    // CHECK-SAME: : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[[SH0]], si32>
    !mo.tensor<[?], si32>, !mo.tensor<[?], si32>)-> !mo.tensor<Sh0, si32>
  // CHECK-NEXT: %[[VAL2:.*]] = mo.reshape<() -> [[D1:.*]], [[D2:.*]]>(%[[ARG0]], %[[ARG0]])
  %2 = mo.reshape<() -> D1, D2>(%arg0, %arg0) : (
    // CHECK-SAME: : (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) -> !mo.tensor<[[[D1]], [[D2]]], si32>
    !mo.tensor<[?], si32>, !mo.tensor<[?], si32>)-> !mo.tensor<[D1, D2], si32>
  // CHECK-NEXT: mo.output %[[ARG0]] : !mo.tensor<[?], si32>
  mo.output %arg0 : !mo.tensor<[?], si32>
}

// CHECK: mo.graph @ops_without_params_dce(%[[ARG0:.*]]: !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32> no_inline {
mo.graph @ops_without_params_dce(%arg0: !mo.tensor<[?], si32>) -> !mo.tensor<[?], si32> {
  %0 = mo.rebind(%arg0) : !mo.tensor<[?], si32> -> !mo.tensor<[?], si32>
  %1 = mo.broadcast_to(%arg0, %arg0) : (
    !mo.tensor<[?], si32>, !mo.tensor<[?], si32>
  )-> !mo.tensor<?, si32>
  %2 = mo.reshape(%arg0, %arg0) : (
    !mo.tensor<[?], si32>, !mo.tensor<[?], si32>
  )-> !mo.tensor<[?], si32>
  // CHECK-NEXT: mo.output %[[ARG0]] : !mo.tensor<[?], si32>
  mo.output %arg0 : !mo.tensor<[?], si32>
}
