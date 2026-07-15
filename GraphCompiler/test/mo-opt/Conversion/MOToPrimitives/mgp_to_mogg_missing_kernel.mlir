// RUN: not mo-opt -allow-unregistered-dialect --mogg-lower-mogg-region-compile %s 2>&1 | FileCheck %s

mgp.model @missing_kernel : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 1
  %2 = mgp.runtime.create %0, %1[0]
  mef.output %2, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %cst = index.constant 42
  %0 = mogg.execute_kernel @__missing(%cst) fusionInfo = mogg.generic.compiled.region:14 {constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = false, outputParamDecls = #kgen<param.decls[]>, usedParams = []} : (index) -> index
  %1 = "test.sink"(%0, %ch) : (index, !mef.chain) -> !mef.chain
  mef.output %1 : !mef.chain
}

// CHECK: error: compiled kernel '__missing' is not present in any loaded module
