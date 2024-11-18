// RUN: mo-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: mgp.model @chain_empty
mgp.model @chain_empty : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.opaque<"cpu0_context">
// CHECK: execute (%[[ARG0:.*]]: !mef.chain,
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.opaque<"cpu0_context">) -> !mef.chain {
  %ch1 = grt.chain.create ()
  %ch2 = grt.chain.create (%arg0)
  %ch3 = grt.chain.create (%ch1, %ch2)
  %ch4 = grt.chain.create (%ch3)
  // CHECK-NEXT: mef.output %[[ARG0]] : !mef.chain
  mef.output %ch3 : !mef.chain
}
