// RUN: mo-opt %s --mo-to-mogg --canonicalize --translate-mogg-to-kgen --mo-to-primitives --lower-primitives | FileCheck %s

// CHECK-LABEL: @while_op
// CHECK: } execute
// CHECK-NEXT: grt.while
// CHECK: summary = "mo.greater
// CHECK: } do {
// CHECK: summary = "mo.add

mo.graph @while_op(%x: !mo.tensor<[], f32>, %arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
   %res = mo.while (%x as %xi: !mo.tensor<[], f32>) {
     %cond = mo.greater (%x, %arg0) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.yield %cond : !mo.tensor<[], bool>
   } do {
     %new_x = mo.add (%xi, %xi) : !mo.tensor<[], f32>
     mo.yield %new_x : !mo.tensor<[], f32>
   }
   mo.output %res : !mo.tensor<[], f32>
}

// CHECK-LABEL: @multi_while_op
// CHECK: } execute (%[[CH0:.*]]: !mef.chain,
// CHECK: %[[WHILE:.*]]:5 = grt.while (%[[CH0]] as %[[CH0_IN:.*]]: !mef.chain,
// CHECK: } do {
// CHECK: mogg.generic.compiled.region (%[[CH0_IN]]{{.*}} -> !mef.chain{

// CHECK: grt.while (%[[WHILE]]#0 as %[[CH0_IN:.*]]: !mef.chain,{{.*}}, %[[WHILE]]#0 as %[[CH1_IN:.*]]: !mef.chain
// CHECK: } do {
// CHECK: %[[CH0:.*]] = grt.chain.create (%[[CH1_IN]],
// CHECK: mogg.generic.compiled.region (%[[CH0]]{{.*}} -> !mef.chain{

mo.graph @multi_while_op(%x: !mo.tensor<[], f32>, %arg0: !mo.tensor<[], f32>) -> (!mo.tensor<[], f32>) {
   %res = mo.while (%x as %xi: !mo.tensor<[], f32>) {
     %cond = mo.greater (%x, %arg0) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.yield %cond : !mo.tensor<[], bool>
   } do {
     %new_x = mo.add (%xi, %xi) : !mo.tensor<[], f32>
     mo.yield %new_x : !mo.tensor<[], f32>
   }

   %res2 = mo.while (%x as %xi: !mo.tensor<[], f32>) {
     %cond = mo.greater (%x, %arg0) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.yield %cond : !mo.tensor<[], bool>
   } do {
     %new_x = mo.add (%res, %xi) : !mo.tensor<[], f32>
     mo.yield %new_x : !mo.tensor<[], f32>
   }

   mo.output %res2 : !mo.tensor<[], f32>
}
