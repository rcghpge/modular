// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --allow-unregistered-dialect --canonicalize -cse | FileCheck %s

// CHECK: mgp.model @static_ranks : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>, %[[ARG3:.*]]: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// CHECK:   %[[BUF0:.*]], %[[SPEC0:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:   %[[BUF1:.*]], %[[SPEC1:.*]] = mgp.cpu.tensor.extract %[[ARG3]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:   %[[VAL0:.*]] = "mop.shape.matmul"(%[[SPEC0]], %[[SPEC1]]) : (!mgp.cpu.tensor_spec<?x?xf32>, !mgp.cpu.tensor_spec<?x?xf32>) -> !mgp.cpu.tensor_spec<?x?xf32>
// CHECK:   %[[VAL1:.*]] = mgp.cpu.buffer.alloc.spec %[[CHAIN]], %[[VAL0]], align {{.*}} : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK:   %[[VAL2:.*]] = "mop.kernels.matmul"(%[[CHAIN]], %[[BUF0]], %[[SPEC0]], %[[BUF1]], %[[SPEC1]], %[[VAL1]], %[[VAL0]])
// CHECK:   %[[VAL3:.*]] = mgp.cpu.tensor.create %[[VAL2]], %[[VAL1]], %[[VAL0]]  : !mgp.cpu.tensor<?x?xf32>
// CHECK:   mef.output %[[CHAIN]], %[[VAL3]] : !mef.chain, !mgp.cpu.tensor<?x?xf32>
// CHECK: }
mo.graph @static_ranks<D0, D1, D2, D3>(%arg0: !mo.tensor<[D0, D1], f32>, %arg1: !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D3], f32> {
  %0 = mo.rebind(%arg1) : !mo.tensor<[D2, D3], f32> -> !mo.tensor<[D1, D3], f32>
  %1 = mo.matmul(%arg0, %0) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D1, D3], f32>) -> !mo.tensor<[D0, D3], f32>
  mo.output %1 : !mo.tensor<[D0, D3], f32>
}
