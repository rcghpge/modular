// RUN: mo-opt %s --mo-to-mgp --allow-unregistered-dialect | FileCheck %s

// CHECK: mgp.model @print
// CHECK:  } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG:.*]]: !mgp.cpu.tensor<4x5xf32>
// CHECK: %[[OUT_CHAIN:.*]] = mgp.debug.cpu.tensor.print %[[CHAIN]], %[[ARG]], "label" : !mgp.cpu.tensor<4x5xf32>

mo.graph @print(%arg0: !mo.tensor<[4, 5], f32>) {
  mo.debug.tensor.print(%arg0) {label = "label"} : !mo.tensor<[4, 5], f32>
  mo.output
}
