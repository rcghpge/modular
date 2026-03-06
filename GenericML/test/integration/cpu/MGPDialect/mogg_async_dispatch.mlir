// Tests for MOGG async dispatch operations:
//   - mogg.call.stub.async: Calls a kernel stub asynchronously via create_task
//   - mogg.async.join: Waits for async task handles and checks for errors
//
// These operations form the foundation for parallel kernel execution.

// RUN: mo-opt %s --mgp-lower-primitives --mogg-outline-kernels --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-jit-compile-kernels --mogg-lower-mogg-region-compile --mgp-outline-models-to-callables -o %t.mlir
// RUN: mt --execute %t.mlir | FileCheck %s

//===----------------------------------------------------------------------===//
// Kernel stubs for async dispatch testing
//===----------------------------------------------------------------------===//

mogg.kernel.stub @noop_stub ()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = false, summary = "noop_stub"} {
  mogg.output
}

mogg.kernel.stub @noop_stub_2 ()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = false, summary = "noop_stub_2"} {
  mogg.output
}

//===----------------------------------------------------------------------===//
// Test: Single async stub call with join
//===----------------------------------------------------------------------===//

// CHECK-LABEL: --- Running 'single_async_call':
// CHECK-NOT: <error
mgp.model @single_async_call : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch1, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {

  %out_chain = mogg.generic.compiled.region (%arg0) :
      (%ch: !mef.chain) -> !mef.chain {

    %task = mogg.call.stub.async @noop_stub -> !mogg.task_handle
    mogg.async.join(%task) : (!mogg.task_handle)
    %ready = mogg.async.ready : !mef.chain

    mogg.output %ready : !mef.chain
  } {device = #M.device_ref<"cpu", 0>, canBeFusedInto = false, canFuseIntoOther = false,
     inOutSignature = #M.inout_sig<"i">, summary = "single_async_call", isAsync = false}

  mef.output %out_chain : !mef.chain
}

//===----------------------------------------------------------------------===//
// Test: Two parallel async stub calls with join
//===----------------------------------------------------------------------===//

// CHECK-LABEL: --- Running 'two_parallel_async_calls':
// CHECK-NOT: <error
mgp.model @two_parallel_async_calls : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch1, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {

  %out_chain = mogg.generic.compiled.region (%arg0) :
      (%ch: !mef.chain) -> !mef.chain {

    %t0 = mogg.call.stub.async @noop_stub -> !mogg.task_handle
    %t1 = mogg.call.stub.async @noop_stub_2 -> !mogg.task_handle
    mogg.async.join(%t0, %t1) : (!mogg.task_handle, !mogg.task_handle)
    %ready = mogg.async.ready : !mef.chain

    mogg.output %ready : !mef.chain
  } {device = #M.device_ref<"cpu", 0>, canBeFusedInto = false, canFuseIntoOther = false,
     inOutSignature = #M.inout_sig<"i">, summary = "two_parallel_async_calls", isAsync = false}

  mef.output %out_chain : !mef.chain
}

//===----------------------------------------------------------------------===//
// Test: Four parallel async stub calls with join
//===----------------------------------------------------------------------===//

// CHECK-LABEL: --- Running 'four_parallel_async_calls':
// CHECK-NOT: <error
mgp.model @four_parallel_async_calls : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch1, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {

  %out_chain = mogg.generic.compiled.region (%arg0) :
      (%ch: !mef.chain) -> !mef.chain {

    %t0 = mogg.call.stub.async @noop_stub -> !mogg.task_handle
    %t1 = mogg.call.stub.async @noop_stub -> !mogg.task_handle
    %t2 = mogg.call.stub.async @noop_stub_2 -> !mogg.task_handle
    %t3 = mogg.call.stub.async @noop_stub_2 -> !mogg.task_handle
    mogg.async.join(%t0, %t1, %t2, %t3) : (!mogg.task_handle, !mogg.task_handle, !mogg.task_handle, !mogg.task_handle)
    %ready = mogg.async.ready : !mef.chain

    mogg.output %ready : !mef.chain
  } {device = #M.device_ref<"cpu", 0>, canBeFusedInto = false, canFuseIntoOther = false,
     inOutSignature = #M.inout_sig<"i">, summary = "four_parallel_async_calls", isAsync = false}

  mef.output %out_chain : !mef.chain
}
