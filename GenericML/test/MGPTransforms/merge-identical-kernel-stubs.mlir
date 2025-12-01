// RUN: EMIT_MOJO=1 mo-opt -merge-identical-kernel-stubs

mogg.kernel.stub @func_1()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_1"} {
  mogg.output
}

mogg.kernel.stub @func_2()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_2"} {
  mogg.output
}

// CHECK-LABEL: @func_3
// CHECK:      mogg.call.stub @func_1
// CHECK-NEXT: mogg.call.stub @func_1
// CHECK-NOT:  mogg.call.stub @func
mogg.kernel.stub @func_3()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_3"} {
  mogg.call.stub @func_1
  mogg.call.stub @func_2
  mogg.output
}

// CHECK-LABEL: @func_4
// CHECK:      mogg.call.stub @func_1
// CHECK:      mogg.call.stub @func_1
// CHECK-NOT:  mogg.call.stub @func
mogg.kernel.stub @func_4()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_4"} {
  mogg.call.stub @func_1
  mogg.call.stub @func_2
  mogg.output
}

// CHECK-LABEL: @func_5
// CHECK:      mogg.call.stub @func_3
// CHECK:      mogg.call.stub @func_3
// CHECK-NOT:  mogg.call.stub @func
mogg.kernel.stub @func_5()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_5"} {
  mogg.call.stub @func_3
  mogg.call.stub @func_4
  mogg.output
}

mogg.kernel.stub @func_6()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_6", foo = "foo"} {
  mogg.output
}

mogg.kernel.stub @func_7()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_7", foo = "foo"} {
  mogg.output
}

// CHECK-LABEL: @top_level
// CHECK: mogg.execute_kernel{{.*}}calleeSymbol = @func_[[VAL:(6|7)]],
// CHECK: mogg.execute_kernel{{.*}}calleeSymbol = @func_[[VAL]],
mgp.model @top_level subgraph : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.device_context<<"gpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 2
    %2 = mgp.runtime.create %0, %1[1]
    mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.device_context<<"gpu", 0>>, %input : !mgp.tensor<4x16xf32>, %idx: index) -> (!mef.chain) {
    "mogg.execute_kernel"() {constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = false, calleeSymbol = @func_6, outputParamDecls = #kgen<param.decls[]>, summary = "mogg.generic.compiled.region:14", usedParams = []} : () -> ()
    "mogg.execute_kernel"() {constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = false, calleeSymbol = @func_7, outputParamDecls = #kgen<param.decls[]>, summary = "mogg.generic.compiled.region:14", usedParams = []} : () -> ()
    mef.output %arg0 : !mef.chain
}
