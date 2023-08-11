// RUN: mo-opt %s | mo-opt | FileCheck %s

// CHECK: mo.graph @rebind_w_params(%[[ARG0:.*]]: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[], si8> no_inline {
mo.graph @rebind_w_params(%arg0: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[], si8> {
  // CHECK: %[[VAL0:.*]] = mo.rebind<() -> [[N:.*]], [[M:.*]]>(%[[ARG0]]) :
  // CHECK-SAME: !mo.tensor<[?, ?], f32> -> !mo.tensor<[[[N]], [[M]]], f32>
  %0 = mo.rebind<() -> N, M>(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N, M], f32>
  // CHECK: %[[VAL1:.*]] = mo.rebind<() -> [[SH:.*]]: !mosh.ape>(%[[VAL0]]) : !mo.tensor<[[[N]], [[M]]], f32> -> !mo.tensor<[[SH]], f16>
  %1 = mo.rebind<() -> Sh: !mosh.ape>(%0) : !mo.tensor<[N, M], f32> -> !mo.tensor<Sh, f16>
  // CHECK: %[[VAL2:.*]] = mo.rebind(%[[VAL1]]) : !mo.tensor<[[SH]], f16> -> !mo.tensor<[], si8>
  %2 = mo.rebind(%1) : !mo.tensor<Sh, f16> -> !mo.tensor<[], si8>
  // CHECK-NEXT: mo.output %[[VAL2]] : !mo.tensor<[], si8>
  mo.output %2 : !mo.tensor<[], si8>
}

// CHECK: mo.graph @rebind_in_if(%[[ARG0:.*]]: !mo.tensor<[?, ?], f32>, %[[COND:.*]]: !mo.tensor<[], bool>) no_inline {
mo.graph @rebind_in_if(%arg0: !mo.tensor<[?, ?], f32>, %cond : !mo.tensor<[], bool>) {
  // CHECK-NEXT: %[[VAL5:.*]] = mo.if(%[[COND]] : !mo.tensor<[], bool>) -> !mo.tensor<[?, ?], f32> {
  %5 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[?, ?], f32> {
    // CHECK-NEXT: %[[VAL3:.*]] = mo.rebind<() -> [[SH2:.*]]: !mosh.ape>(%[[ARG0]]) :
    // CHECK-SAME: !mo.tensor<[?, ?], f32> -> !mo.tensor<[[SH2]], f32>
    %3 = mo.rebind<() -> Sh2: !mosh.ape>(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<Sh2, f32>
    // CHECK: mo.yield %[[ARG0]] : !mo.tensor<[?, ?], f32>
    mo.yield %arg0 : !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: %[[VAL4:.*]] = mo.rebind<() -> [[SH3:.*]]: !mosh.ape>(%[[ARG0]]) :
    // CHECK-SAME: !mo.tensor<[?, ?], f32> -> !mo.tensor<[[SH3]], f32>
    %4 = mo.rebind<() -> Sh3: !mosh.ape>(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<Sh3, f32>
    // CHECK: mo.yield %[[ARG0]] : !mo.tensor<[?, ?], f32>
    mo.yield %arg0 : !mo.tensor<[?, ?], f32>
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: mo.output
  mo.output
}
