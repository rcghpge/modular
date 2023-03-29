// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --allow-unregistered-dialect --canonicalize -cse | FileCheck %s

// CHECK: mgp.model @static_ranks : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>, %[[ARG3:.*]]: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// CHECK:   %[[BUF0:.*]], %[[SPEC0:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:   %[[DIM0:.*]]:2 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC0]] : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK:   %[[BUF1:.*]], %[[SPEC1:.*]] = mgp.cpu.tensor.extract %[[ARG3]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:   %[[DIM1:.*]]:2 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC1]] : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK:   %[[VAL0:.*]] = mgp.cpu.tensor_spec.create %[[DIM0]]#0, %[[DIM1]]#1 : !mgp.cpu.tensor_spec<?x?xf32>
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

// CHECK: mgp.model @static_ranks_int_to_unknown : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:.*]]: !mgp.cpu.tensor<42x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// CHECK:   %[[CDIM:.*]] = mip.constant.index 42
// CHECK:   %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<42x?xf32>
// CHECK:   %[[DYNDIM:.*]] = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<42x?xf32>
// CHECK:   %[[NEWSPEC:.*]] = mgp.cpu.tensor_spec.create %[[CDIM]], %[[DYNDIM]] : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK:   %[[RES:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %[[BUF]], %[[NEWSPEC]]  : !mgp.cpu.tensor<?x?xf32>
// CHECK:   mef.output %[[CHAIN]], %[[RES]] : !mef.chain, !mgp.cpu.tensor<?x?xf32>
// CHECK: }
mo.graph @static_ranks_int_to_unknown<D0, D1>(
  %arg0: !mo.tensor<[42, D0], f32>
) -> !mo.tensor<[?, D0], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[42, D0], f32> -> !mo.tensor<[?, D0], f32>
  mo.output %res : !mo.tensor<[?, D0], f32>
}

// CHECK: mgp.model @static_ranks_unknown_to_int : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<42x?xf32>) {
// CHECK:   %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:   %[[DYNDIMS:.*]]:2 = mgp.cpu.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK:   %[[NEWSPEC:.*]] = mgp.cpu.tensor_spec.create %[[DYNDIMS]]#1 : !mgp.cpu.tensor_spec<42x?xf32>
// CHECK:   %[[RES:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %[[BUF]], %[[NEWSPEC]]  : !mgp.cpu.tensor<42x?xf32>
// CHECK:   mef.output %[[CHAIN]], %[[RES]] : !mef.chain, !mgp.cpu.tensor<42x?xf32>
// CHECK: }
mo.graph @static_ranks_unknown_to_int<D0, D1>(
  %arg0: !mo.tensor<[D1, D0], f32>
) -> !mo.tensor<[42, D0], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, D0], f32>
  mo.output %res : !mo.tensor<[42, D0], f32>
}

// CHECK: mgp.model @static_ranks_unknown_to_static : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<42x10xf32>) {
// CHECK:   %[[NEWSPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<42x10xf32>
// CHECK:   %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:   %[[RES:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %[[BUF]], %[[NEWSPEC]] !mgp.cpu.buffer : !mgp.cpu.tensor<42x10xf32>
// CHECK:   mef.output %[[CHAIN]], %[[RES]] : !mef.chain, !mgp.cpu.tensor<42x10xf32>
// CHECK: }
mo.graph @static_ranks_unknown_to_static<D0, D1>(
  %arg0: !mo.tensor<[D1, D0], f32>
) -> !mo.tensor<[42, 10], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, 10], f32>
  mo.output %res : !mo.tensor<[42, 10], f32>
}
