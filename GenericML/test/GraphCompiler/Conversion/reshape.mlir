// RUN: mo-opt %s --mo-to-primitives | FileCheck %s

// Entry point
// CHECK:   mgp.model @static_reshape : mof
//
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<?x?xf32>, %[[ARG3:.*]]: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {

// CHECK:   %[[SPEC3:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][0], %[[ARG2]]
// CHECK:   %[[DIMS:.:*]]:2 = mgp.tensor_spec.dims.dynamic %[[SPEC3]]

// CHECK:   %[[SPEC1:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][0], %[[ARG3]]
// CHECK:   %[[RESIZED_DIMS:.:*]]:2 = mgp.tensor_spec.dims.dynamic %[[SPEC1]]


// CHECK:   %[[BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][0], %[[ARG2]]
// CHECK:   %[[NEW_SPEC:.*]] = mgp.tensor_spec.create %[[RESIZED_DIMS]]#0, %[[RESIZED_DIMS]]#1 : !mgp.tensor_spec<?x?xf32>

// CHECK:   %[[OUT:.*]] = mgp.tensor.create %[[CH0]], %[[CTX]][0], %[[BUF]], %[[NEW_SPEC]] : !mgp.tensor<?x?xf32>
// CHECK:   mef.output %[[CH0]], %[[OUT]]

mo.graph @static_reshape<D4, D5, D6, D7>(%arg0: !mo.tensor<[D4, D5], f32>, %arg1: !mo.tensor<[D6, D7], f32>) -> !mo.tensor<[D6, D7], f32> {
  %1 = mo.static.reshape(%arg0) : !mo.tensor<[D4, D5], f32> -> !mo.tensor<[D6, D7], f32>
  mo.output %1 : !mo.tensor<[D6, D7], f32>
}
