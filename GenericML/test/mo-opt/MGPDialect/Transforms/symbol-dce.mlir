// RUN: mo-opt -symbol-dce

mogg.kernel.stub @callee_1() fusionInfo = callee_1
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  mogg.output
}

mogg.kernel.stub @callee_2() fusionInfo = callee_2
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  mogg.output
}

mogg.kernel.stub @caller() fusionInfo = caller
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  mogg.call.stub @callee_1
  mogg.call.stub @callee_2
  mogg.output
}

// CHECK-NOT: mogg.kernel.stub @callee_2
// CHECK-NOT: mogg.kernel.stub @caller
// CHECK: mogg.call.stub @callee_1
// CHECK-NOT: mogg.call.stub
mgp.model @post_abi_fusion : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %3 = mgp.runtime.create %0, %1[1]
  mef.output %3, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  mogg.call.stub @callee_1
  mef.output %arg0 : !mef.chain
}
