// RUN: mo-opt %s -mogg-merge-identical-kernel-stubs -symbol-dce | FileCheck %s

// Regression test for GEX-3774: two structurally-identical region stubs each
// call a differently-named-but-equivalent inner stub via mogg.call.stub.async.
// Because the async call is a call-graph edge, the inner-stub name is
// canonicalized before the region is hashed, so the inner stubs and the
// regions each collapse to one in a single merge + symbol-dce run.

mogg.kernel.stub @inner_1() fusionInfo = inner_1
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"gpu", 0>, isAsync = true} {
  mogg.output
}

mogg.kernel.stub @inner_2() fusionInfo = inner_2
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"gpu", 0>, isAsync = true} {
  mogg.output
}

mogg.kernel.stub @region_a() fusionInfo = region_a
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"gpu", 0>, isAsync = true} {
  %t = mogg.call.stub.async @inner_1 -> !mogg.task_handle
  mogg.async.join(%t) : (!mogg.task_handle)
  mogg.output
}

mogg.kernel.stub @region_b() fusionInfo = region_b
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"gpu", 0>, isAsync = true} {
  %t = mogg.call.stub.async @inner_2 -> !mogg.task_handle
  mogg.async.join(%t) : (!mogg.task_handle)
  mogg.output
}

// Exactly one inner stub and one region stub survive, and the surviving
// region's async callsite points at the canonical inner stub.
// CHECK:       mogg.kernel.stub @inner_1
// CHECK-NOT:   mogg.kernel.stub @inner_2
// CHECK:       mogg.kernel.stub @region_a
// CHECK-NOT:   mogg.kernel.stub @region_b
// CHECK:         mogg.call.stub.async @inner_1

// Both model executions target the single merged region.
// CHECK-LABEL: mgp.model @top_level
// CHECK:       mogg.execute_kernel @region_a
// CHECK:       mogg.execute_kernel @region_a
// CHECK-NOT:   @region_b
// CHECK-NOT:   @inner_2
mgp.model @top_level subgraph : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.device_context<<"gpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 2
    %2 = mgp.runtime.create %0, %1[1]
    mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.device_context<<"gpu", 0>>, %input : !mgp.tensor<4x16xf32>, %idx: index) -> (!mef.chain) {
    mogg.execute_kernel @region_a() fusionInfo = mogg.generic.compiled.region:14 {constantResources = [], device = #M.device_ref<"gpu", 0>, isAsync = false, outputParamDecls = #kgen<param.decls[]>, usedParams = []} : () -> ()
    mogg.execute_kernel @region_b() fusionInfo = mogg.generic.compiled.region:14 {constantResources = [], device = #M.device_ref<"gpu", 0>, isAsync = false, outputParamDecls = #kgen<param.decls[]>, usedParams = []} : () -> ()
    mef.output %arg0 : !mef.chain
}
