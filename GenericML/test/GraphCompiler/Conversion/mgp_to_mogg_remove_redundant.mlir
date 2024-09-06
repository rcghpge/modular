// RUN: mo-opt %s --remove-redundant-operands | FileCheck %s

mgp.model @remove_redundant_operands : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %out_chain = mgp.runtime.create %0, %1[1] device <"cpu", 0>
  mef.output %out_chain, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %0 = grt.chain.create ()
  %lhs = index.constant 42
  %rhs = index.constant -20
  %3 = mogg.generic.compiled.region (%lhs, %rhs, %lhs, %rhs) : (%arg2: index, %arg3: index, %arg4: index, %arg5: index) -> index{
    %5 = mogg.call "index.add"(%arg2, %arg3) {} : index, index -> index
    %6 = mogg.call "index.add"(%5, %arg4) {} : index, index -> index
    mogg.output %5 : index
  } {inOutSignature = #M.inout_sig<"">}
  %4 = mip.print.index %3, %0
  mef.output %4 : !mef.chain
}

// CHECK-LABEL: mgp.model @remove_redundant_operands
// CHECK: %[[RES:[^ ]*]] = mogg.generic.compiled.region (%[[OP0:[^ ]*]], %[[OP1:[^ ]*]]) : (%[[ARG0:[^ ]*]]: index, %[[ARG1:[^ ]*]]: index) -> index{
// CHECK-DAG: %[[INT0:[^ ]*]] = mogg.call{{.*}}(%[[ARG0]], %[[ARG1]])
// CHECK-DAG: %[[INT1:[^ ]*]] = mogg.call{{.*}}(%[[INT0]], %[[ARG0]])
