// RUN: mo-opt %s --mo-to-mgp=constant-fold=false | FileCheck %s

// CHECK-LABEL: mgp.model @print
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG:.*]]: !mgp.tensor<4x5xf32>)
// CHECK: %[[BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][1], %[[ARG]]
// CHECK: %[[SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][1], %[[ARG]]
// CHECK: %[[CH1:.*]] = mgp.debug.tensor.print %{{.*}}, %[[CTX]][1], %[[BUF]], %[[SPEC]], "label"
// CHECK: mef.output %[[CH1]]

mo.graph @print(%arg0: !mo.tensor<[4, 5], f32>) {
  mo.debug.tensor.print(%arg0) {label = "label"} : !mo.tensor<[4, 5], f32>
  mo.output
}
