// RUN: mo-opt %s --mgp-raise-back-index-constant | FileCheck %s

// `mgp-raise-back-index-constant` recovers an `index.constant` from the
// `mgp.resource.constant` + `mgp.buffer.to_index` lowering when a kernel stub
// has exactly one caller: the inlined constant is then unambiguous and costs
// fewer lines of emitted Mojo. A stub with multiple callers is left alone,
// since the constant resource may differ between call sites.

// Single caller: the slot-0 constant resource (little-endian int64 42) is
// raised back to `index.constant 42`, which feeds the output directly.

// CHECK-LABEL: mogg.kernel.stub @raise_single
// CHECK:         %[[C:.+]] = index.constant 42
// CHECK:         mogg.output %[[C]] : index
mogg.kernel.stub @raise_single(%chain: !mef.chain, %ctx: !mgp.context) -> (index) fusionInfo = raise_single
  attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = false} {
  %buf = mgp.resource.constant() resourceSlot = 0 : () -> !mgp.buffer<<"cpu", 0>>
  %idx = mgp.buffer.to_index %chain, %buf : !mgp.buffer<<"cpu", 0>>
  mogg.output %idx : index
}

// Two callers: the lowering is preserved because the constant could differ
// between the two call sites, so no `index.constant` is recovered.

// CHECK-LABEL: mogg.kernel.stub @raise_multi
// CHECK-NOT:     index.constant
// CHECK:         mgp.buffer.to_index
mogg.kernel.stub @raise_multi(%chain: !mef.chain, %ctx: !mgp.context) -> (index) fusionInfo = raise_multi
  attributes {abi = 1 : i32, constantResources = [], device = "cpu", isAsync = false} {
  %buf = mgp.resource.constant() resourceSlot = 0 : () -> !mgp.buffer<<"cpu", 0>>
  %idx = mgp.buffer.to_index %chain, %buf : !mgp.buffer<<"cpu", 0>>
  mogg.output %idx : index
}

mgp.model @top subgraph : mof devices [#M.device_ref<"cpu", 0>]
init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 1
    mef.output %0, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %idx: index) -> (!mef.chain) {
    %s = mogg.execute_kernel @raise_single(%arg0, %arg1) fusionInfo = mogg.generic.compiled.region:1 constantResources = [#M.aligned_bytes<"0x2A00000000000000", align 8>], isAsync = false, usedParams = [] {device = #M.device_ref<"cpu", 0>}
        : (!mef.chain, !mgp.context) -> (index)
    %m0 = mogg.execute_kernel @raise_multi(%arg0, %arg1) fusionInfo = mogg.generic.compiled.region:2 constantResources = [#M.aligned_bytes<"0x2A00000000000000", align 8>], isAsync = false, usedParams = [] {device = #M.device_ref<"cpu", 0>}
        : (!mef.chain, !mgp.context) -> (index)
    %m1 = mogg.execute_kernel @raise_multi(%arg0, %arg1) fusionInfo = mogg.generic.compiled.region:3 constantResources = [#M.aligned_bytes<"0x2A00000000000000", align 8>], isAsync = false, usedParams = [] {device = #M.device_ref<"cpu", 0>}
        : (!mef.chain, !mgp.context) -> (index)
    mef.output %arg0 : !mef.chain
}
