// RUN: mo-opt --canonicalize %s | mo-opt | FileCheck %s


// CHECK: mo.graph @gelu(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> attributes {isSubgraph} {
mo.graph @gelu(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) attributes {isSubgraph} {
  // CHECK-NEXT: mo.gelu(%[[ARG0]]) : !mo.tensor<[4, 5], f32>
  %0 = mo.gelu(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// CHECK: mo.graph @main(%[[ARG0:.*]]: !mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32> {
mo.graph @main(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // CHECK-NEXT: %0 = mo.call {symbol = @gelu}(%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
  %0 = mo.call {symbol = @gelu}(%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}