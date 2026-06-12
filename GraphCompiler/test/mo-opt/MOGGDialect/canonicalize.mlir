// RUN: mo-opt -allow-unregistered-dialect --canonicalize --split-input-file %s | FileCheck %s

mo.graph @genericRegionChainCreate(%arg0: index, %arg1: index) {
  %ch = grt.chain.create(%arg0 : index, %arg1 : index)

  mogg.generic.compiled.region (%ch) : (%in : !mef.chain) {
    mogg.output
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"."> }

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionChainCreate
// CHECK:       (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK:       mogg.generic.compiled.region (%[[ARG0]], %[[ARG1]])
// CHECK:       inOutSignature = #M.inout_sig<"..">

// -----

mo.graph @genericRegionChainCreateNoInline(%arg0: index, %arg1: index) {
  %ch = grt.chain.create(%arg0 : index, %arg1 : index)

  mogg.generic.compiled.region (%ch) : (%in : !mef.chain) {
    mogg.output
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"."> }

  mogg.generic.compiled.region (%ch) : (%in : !mef.chain) {
    mogg.output
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"."> }

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionChainCreateNoInline
// CHECK:       (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK:       %[[CH:.*]] = grt.chain.create (%[[ARG0]]: index, %[[ARG1]]: index)
// CHECK:       mogg.generic.compiled.region (%[[CH]])
// CHECK:       inOutSignature = #M.inout_sig<".">
// CHECK:       mogg.generic.compiled.region (%[[CH]])
// CHECK:       inOutSignature = #M.inout_sig<".">

// -----

mo.graph @genericRegionEmptyChain() {
  %ch = grt.chain.create()

   mogg.generic.compiled.region (%ch) : (%in : !mef.chain) {
    mogg.output
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"."> }

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionEmptyChain
// CHECK:       mogg.generic.compiled.region ()
// CHECK:       inOutSignature = #M.inout_sig<"">

// -----

// Duplicate operands are collapsed to the first occurrence, the duplicate body
// argument is erased, and uses of it are redirected to the first occurrence's
// argument. The signature drops the corresponding entry.
mo.graph @genericRegionDedupOperands(%arg0: index, %arg1: index) {
  %0 = mogg.generic.compiled.region (%arg0, %arg1, %arg0) : (%a: index, %b: index, %c: index) -> index {
    %u = mogg.async.unpack(%c) : (!mogg.async<index>) -> index
    %p = mogg.async.pack(%u) : (index) -> !mogg.async<index>
    mogg.output %p : !mogg.async<index>
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"..."> }

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionDedupOperands
// CHECK:       (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK:       mogg.generic.compiled.region (%[[ARG0]], %[[ARG1]]) :
// CHECK-SAME:  (%[[A:[^,]*]]: index, %[[B:[^,]*]]: index)
// CHECK:         mogg.async.unpack(%[[A]])
// CHECK:       inOutSignature = #M.inout_sig<"..">

// -----

// When duplicate operands carry different in/out semantics, the surviving
// operand absorbs the union: `kIn` ('i') + `kMut` ('m') -> `kMut` ('m').
mo.graph @genericRegionDedupUnionSemantics(%arg0: index) {
  mogg.generic.compiled.region (%arg0, %arg0) : (%a: index, %b: index) {
    mogg.output
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"im"> }

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionDedupUnionSemantics
// CHECK:       mogg.generic.compiled.region (%[[ARG0:[^,)]*]]) :
// CHECK-SAME:  (%{{[^,]*}}: index)
// CHECK:       inOutSignature = #M.inout_sig<"m">

// -----

// Unused mogg.try results are dropped: the try/except mogg.yield ops drop the
// corresponding operands, and the now-dead mogg.async.error in the except
// block is erased.
func.func @tryUnusedResult(%arg0: index) -> index {
  %0:2 = mogg.try {
    %a = mogg.async.pack(%arg0) : (index) -> !mogg.async<index>
    mogg.yield %a, %a : !mogg.async<index>, !mogg.async<index>
  } except {
  ^bb0(%err: !mogg.error):
    %e0 = mogg.async.error %err : !mogg.async<index>
    %e1 = mogg.async.error %err : !mogg.async<index>
    mogg.yield %e0, %e1 : !mogg.async<index>, !mogg.async<index>
  } : !mogg.async<index>, !mogg.async<index>
  %u = mogg.async.unpack(%0#0) : (!mogg.async<index>) -> index
  return %u : index
}

// CHECK-LABEL: func.func @tryUnusedResult
// CHECK:       %[[TRY:.*]] = mogg.try {
// CHECK:         mogg.yield %{{[^,]*}} : !mogg.async<index>
// CHECK:       } except {
// CHECK:         %[[ERR:.*]] = mogg.async.error
// CHECK-NOT:     mogg.async.error
// CHECK:         mogg.yield %[[ERR]] : !mogg.async<index>
// CHECK:       } : !mogg.async<index>
// CHECK:       mogg.async.unpack(%[[TRY]])

// -----

// Duplicate `!mef.chain` outputs are collapsed to a single output, and uses of
// the removed chain results are rewired to the surviving one.
mgp.model @remove_redundant_chains : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %true = index.bool.constant true
  %true_0 = index.bool.constant true
  %0:2 = mogg.generic.compiled.region (%true, %true_0) : (%arg4: i1, %arg5: i1)  -> (!mef.chain, !mef.chain) {
    %1 = mogg.async.ready : !mef.chain
    %2 = mogg.async.ready : !mef.chain
    mogg.output %1, %2 : !mef.chain, !mef.chain
  } fusionInfo = Sequence(mgp.assert, mgp.assert) {canBeFusedInto = true, canFuseIntoOther = true, device = #M.device_ref<"cpu", 0>, emitsMojo = false, inOutSignature = #M.inout_sig<"">, isAsync = false}
  %res = "test.sink"(%0#0, %0#1) : (!mef.chain, !mef.chain) -> !mef.chain
  mef.output %res : !mef.chain
}

// CHECK-LABEL: mgp.model @remove_redundant_chains
// CHECK:         %[[OUT:[^ ]*]] = mogg.generic.compiled.region ({{.*}}) -> !mef.chain {
// CHECK:           mogg.output %{{[^ ]*}} : !mef.chain
// CHECK:         "test.sink"(%[[OUT]], %[[OUT]])

// -----

// Op attributes (e.g. emitsMojo) are preserved when a region is left unchanged.
mgp.model @preserve_kernel_attributes : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain) {
  %true = index.bool.constant true
  %true_0 = index.bool.constant true
  mogg.generic.compiled.region (%true, %true_0) : (%arg4: i1, %arg5: i1)  -> () {
    mogg.output
  } fusionInfo = Sequence(mgp.assert, mgp.assert) {canBeFusedInto = true, canFuseIntoOther = true, device = #M.device_ref<"cpu", 0>, emitsMojo = true, inOutSignature = #M.inout_sig<"">, isAsync = false}
  mef.output %arg0 : !mef.chain
}

// CHECK-LABEL: mgp.model @preserve_kernel_attributes
// CHECK:           emitsMojo = true
