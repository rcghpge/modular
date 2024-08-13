// RUN: mo-opt %s --mo-to-primitives | FileCheck %s

// CHECK-LABEL: mgp.model @print
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG:[^:]*]]: !mgp.tensor<4x5xf32>)
// CHECK: %[[BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][0], %[[ARG]]
// CHECK: %[[SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][0], %[[ARG]]
// CHECK: %[[CH1:.*]] = mgp.debug.tensor.print %{{.*}}, %[[CTX]][0], %[[BUF]], %[[SPEC]], "label"

mo.graph @print(%arg0: !mo.tensor<[4, 5], f32>) {
  mo.debug.tensor.print(%arg0) {label = "label"} : !mo.tensor<[4, 5], f32>
  mo.output
}
