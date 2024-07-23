// RUN: mo-opt %s --remove-redundant-operands | FileCheck %s

mgp.model @remove_redundant_operands : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %out_chain = mgp.runtime.create %0, %1[1] device <"cpu", 0>
  mef.output %out_chain, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque) -> !mef.chain {
  %0 = grt.chain.create ()
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -20
  %3 = mogg.generic.compiled.region (%lhs, %rhs, %lhs, %rhs) : (%arg2: index, %arg3: index, %arg4: index, %arg5: index) -> index{
    %5 = mogg.call "mip.add"(%arg2, %arg3) {} : index, index -> index
    %6 = mogg.call "mip.add"(%5, %arg4) {} : index, index -> index
    mogg.output %5 : index
  } {inOutSignature = #M.inout_sig<"">}
  %4 = mip.print.index %3, %0
  mef.output %4 : !mef.chain
}

// CHECK-LABEL: mgp.model @remove_redundant_operands
// CHECK: %[[RES:[^ ]*]] = mogg.generic.compiled.region (%[[OP0:[^ ]*]], %[[OP1:[^ ]*]]) : (%[[ARG0:[^ ]*]]: index, %[[ARG1:[^ ]*]]: index) -> index{
// CHECK-DAG: %[[INT0:[^ ]*]] = mogg.call{{.*}}(%[[ARG0]], %[[ARG1]])
// CHECK-DAG: %[[INT1:[^ ]*]] = mogg.call{{.*}}(%[[INT0]], %[[ARG0]])
