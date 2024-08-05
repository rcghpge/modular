// RUN: mo-opt %s --add-devices --mo-to-mogg --canonicalize --translate-mogg-to-kgen --inline-lambdas --mo-to-primitives --lower-primitives --lower-abi --canonicalize -cse --jit-compile-kernels --lower-mogg-region-compile | FileCheck %s

// CHECK-LABEL: @while_op
// CHECK: } execute
// CHECK-NEXT: grt.while
// CHECK: mgp.generic.execute{{.*}}dSummary = "mo.greater
// CHECK: } do {
// CHECK: mgp.generic.execute{{.*}}dSummary = "mo.add

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
