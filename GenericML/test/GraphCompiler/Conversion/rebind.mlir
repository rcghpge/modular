// RUN: mo-opt %s --add-devices --mo-to-mogg --mo-to-primitives --allow-unregistered-dialect --cse --canonicalize | FileCheck %s

// CHECK-LABEL: mgp.model @static_ranks_int_to_unknown : mof
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG2:[a-z0-9]*]]: !mgp.tensor<42x?xf32>)
// CHECK:   %[[CDIM:.*]] = mip.constant.index 42
// CHECK:   %[[SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][0], %[[ARG2]] : !mgp.tensor<42x?xf32>
// CHECK:   %[[DYNDIM:.*]] = mgp.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.tensor_spec<42x?xf32>
// CHECK:   %[[BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][0], %[[ARG2]] : !mgp.tensor<42x?xf32>
// CHECK:   %[[NEWSPEC:.*]] = mgp.tensor_spec.create %[[CDIM]], %[[DYNDIM]] : !mgp.tensor_spec<?x?xf32>
// CHECK:   %[[RES:.*]] = mgp.tensor.create %[[CH0]], %[[CTX]][0], %[[BUF]], %[[NEWSPEC]] : !mgp.tensor<?x?xf32>
// CHECK:   mef.output %[[CH0]], %[[RES]] : !mef.chain, !mgp.tensor<?x?xf32>
// CHECK: }
mo.graph @static_ranks_int_to_unknown<D0, D1>(
  %arg0: !mo.tensor<[42, D0], f32>
) -> !mo.tensor<[?, D0], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[42, D0], f32> -> !mo.tensor<[?, D0], f32>
  mo.output %res : !mo.tensor<[?, D0], f32>
}

// CHECK-LABEL: mgp.model @static_ranks_unknown_to_int : mof
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG2:[a-z0-9]*]]: !mgp.tensor<?x?xf32>)
// CHECK:   %[[SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][0], %[[ARG2]] : !mgp.tensor<?x?xf32>
// CHECK:   %[[DYNDIMS:.*]]:2 = mgp.tensor_spec.dims.dynamic %[[SPEC]] : !mgp.tensor_spec<?x?xf32>
// CHECK:   %[[BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][0], %[[ARG2]] : !mgp.tensor<?x?xf32>
// CHECK:   %[[NEWSPEC:.*]] = mgp.tensor_spec.create %[[DYNDIMS]]#1 : !mgp.tensor_spec<42x?xf32>
// CHECK:   %[[RES:.*]] = mgp.tensor.create %[[CH0]], %[[CTX]][0], %[[BUF]], %[[NEWSPEC]]  : !mgp.tensor<42x?xf32>
// CHECK:   mef.output %[[CH0]], %[[RES]] : !mef.chain, !mgp.tensor<42x?xf32>
// CHECK: }
mo.graph @static_ranks_unknown_to_int<D0, D1 -> ()>(
  %arg0: !mo.tensor<[D1, D0], f32>
) -> !mo.tensor<[42, D0], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, D0], f32>
  mo.output %res : !mo.tensor<[42, D0], f32>
}

// CHECK-LABEL: mgp.model @static_ranks_unknown_to_static : mof
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK: %[[ARG2:[a-z0-9]*]]: !mgp.tensor<?x?xf32>)
// CHECK:   %[[BUF:.*]] = mgp.tensor.extract.buffer %[[CTX]][0], %[[ARG2]] : !mgp.tensor<?x?xf32>
// CHECK:   %[[NEWSPEC:.*]] = mgp.tensor_spec.create : !mgp.tensor_spec<42x10xf32>
// CHECK:   %[[SIZED_BUF:.*]] = mgp.buffer.resize %[[BUF]] : !mgp.buffer<1680>
// CHECK:   %[[RES:.*]] = mgp.tensor.create %[[CH0]], %[[CTX]][0], %[[SIZED_BUF]], %[[NEWSPEC]] : !mgp.tensor<42x10xf32>
// CHECK:   mef.output %[[CH0]], %[[RES]] : !mef.chain, !mgp.tensor<42x10xf32>
// CHECK: }
mo.graph @static_ranks_unknown_to_static<D0, D1>(
  %arg0: !mo.tensor<[D1, D0], f32>
) -> !mo.tensor<[42, 10], f32> {
  %res = mo.rebind(%arg0) : !mo.tensor<[D1, D0], f32> -> !mo.tensor<[42, 10], f32>
  mo.output %res : !mo.tensor<[42, 10], f32>
}
