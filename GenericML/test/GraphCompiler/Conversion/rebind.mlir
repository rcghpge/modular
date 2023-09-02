// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --allow-unregistered-dialect --cse --canonicalize | FileCheck %s

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
mo.graph @static_ranks_unknown_to_int<D0, D1 -> ()>(
  %arg0: !mo.tensor<[D1, D0], f32>
) -> !mo.tensor<[42, D0], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, D0], f32>
  mo.output %res : !mo.tensor<[42, D0], f32>
}

// CHECK: mgp.model @static_ranks_unknown_to_static : tf init {
// CHECK: } execute (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:.*]]: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<42x10xf32>) {
// CHECK:   %[[BUF:.*]], %[[SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:   %[[NEWSPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<42x10xf32>
// CHECK:   %[[RES:.*]] = mgp.cpu.tensor.create %[[CHAIN]], %[[BUF]], %[[NEWSPEC]] !mgp.cpu.buffer : !mgp.cpu.tensor<42x10xf32>
// CHECK:   mef.output %[[CHAIN]], %[[RES]] : !mef.chain, !mgp.cpu.tensor<42x10xf32>
// CHECK: }
mo.graph @static_ranks_unknown_to_static<D0, D1>(
  %arg0: !mo.tensor<[D1, D0], f32>
) -> !mo.tensor<[42, 10], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, 10], f32>
  mo.output %res : !mo.tensor<[42, 10], f32>
}
