// RUN: mo-opt %s -allow-unregistered-dialect --remove-redundant-operands --split-input-file | FileCheck %s

mgp.model @remove_redundant_operands : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %out_chain = mgp.runtime.create %0, %1[1]
  mef.output %out_chain, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -20
  %3 = mogg.generic.compiled.region (%lhs, %rhs, %lhs, %rhs) : (%arg2: index, %arg3: index, %arg4: index, %arg5: index) -> index{
    %5 = mogg.call "index.add"(%arg2, %arg3) {} : index, index -> index
    %6 = mogg.call "index.add"(%5, %arg4) {} : index, index -> index
    mogg.output %5 : index
  } {device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"....">}
  %4 = "test.sink"(%3, %0) : (index, !mef.chain) -> !mef.chain
  mef.output %4 : !mef.chain
}

// CHECK-LABEL: mgp.model @remove_redundant_operands
// CHECK: %[[RES:[^ ]*]] = mogg.generic.compiled.region (%[[OP0:[^ ]*]], %[[OP1:[^ ]*]]) : (%[[ARG0:[^ ]*]]: index, %[[ARG1:[^ ]*]]: index) -> index{
// CHECK-DAG: %[[INT0:[^ ]*]] = mogg.call{{.*}}(%[[ARG0]], %[[ARG1]])
// CHECK-DAG: %[[INT1:[^ ]*]] = mogg.call{{.*}}(%[[INT0]], %[[ARG0]])
// CHECK-DAG: inOutSignature = #M.inout_sig<"..">

// -----

mgp.model @remove_redundant_chains : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %true = index.bool.constant true
  %true_0 = index.bool.constant true
  %0:2 = mogg.generic.compiled.region (%true, %true_0) : (%arg4: i1, %arg5: i1)  -> (!mef.chain, !mef.chain) {
    %1 = mogg.call "mgp.assert"(%arg4) {message = "ASSERT_0" : !kgen.string} : i1 -> !mef.chain
    %2 = mogg.call "mgp.assert"(%arg5) {message = "ASSERT_1" : !kgen.string} : i1 -> !mef.chain
    mogg.output %1, %2 : !mef.chain, !mef.chain
  } {canBeFusedInto = true, canFuseIntoOther = true, device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false, summary = "mgp.assert+mgp.assert", emitsMojo = true}
  %res = "test.sink"(%0#0, %0#1) : (!mef.chain, !mef.chain) -> !mef.chain
  mef.output %res : !mef.chain
}

// CHECK-LABEL: mgp.model @remove_redundant_chains
// CHECK:         %[[OUT:[^ ]*]] = mogg.generic.compiled.region ({{.*}}) -> !mef.chain{
// CHECK:           mogg.output %{{[^ ]*}} : !mef.chain
// CHECK:           emitsMojo = true
// CHECK:         "test.sink"(%[[OUT]], %[[OUT]])
