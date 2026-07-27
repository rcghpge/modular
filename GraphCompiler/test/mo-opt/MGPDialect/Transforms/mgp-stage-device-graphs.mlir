// RUN: mo-opt %s --mgp-stage-device-graphs --allow-unregistered-dialect --split-input-file | FileCheck %s

// An `isDeviceGraph` model: every `mogg.generic.compiled.region` in the execute
// block is moved into a single `IsolatedFromAbove` `mgp.device_graph.staging`
// op. Free values the regions reference (here the model input) become operands
// passed in as bare-`T` block args; values consumed outside become the op's
// results (type `T`), yielded as bare `T`.

// CHECK-LABEL: mgp.model @dg
// CHECK: execute (%{{.*}}: !mef.chain{{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>) ->
// The staging op takes the free value as an operand and exposes it to the body
// as a bare-`T` block arg alongside the device-graph builder.
// CHECK: %[[STAGING:.*]] = mgp.device_graph.staging inputs(%[[IN]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>):
// The bare input feeds region A directly; region B consumes A's result. No
// async pack/unpack is introduced at the staging boundary.
// CHECK-NOT: mogg.async.unpack
// CHECK: %[[A:.*]] = mogg.generic.compiled.region (%[[INT]])
// CHECK: %[[B:.*]] = mogg.generic.compiled.region (%[[A]])
// The escaping value is yielded directly as bare `T`.
// CHECK: mgp.device_graph.staging.yield %[[B]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: mef.output %{{.*}}, %[[STAGING]] : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>

mgp.model @dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %a = mogg.generic.compiled.region (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%b) : !mgp.buffer<<"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
  } fusionInfo = "test.a" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
  %b = mogg.generic.compiled.region (%a) : (%arg: !mgp.buffer<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.buffer<<"gpu", 0>>
    %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %ch = mogg.chain.dummy() : () -> !mef.chain
    %t = mgp.tensor.create %ch, %u, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%t) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  } fusionInfo = "test.b" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
  mef.output %ch0, %b : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}

// -----

// A non-`mogg.generic.compiled.region` op interleaved between two staged regions
// (here an `mgp.guard`) must be staged together with them. `mgp.guard` is
// lowered to an equivalent `mogg.generic.compiled.region` and then staged as
// other compiled regions.

// CHECK-LABEL: mgp.model @dg_guard
// No `mgp.guard` op is stranded ahead of the staging op (that stranding is the
// cycle); the guard has been rewritten to a compiled region.
// CHECK: execute (
// CHECK-NOT: mgp.guard
// The staging op takes only the model input -- its `inputs` must not reference a
// value the staging op itself produces.
// CHECK: %[[STAGING:.*]] = mgp.device_graph.staging inputs(%[[IN:[a-zA-Z0-9_]+]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>):
// Region A (the assert) produces the chain.
// CHECK: %[[A:[a-zA-Z0-9_]+]] = mogg.generic.compiled.region (%[[INT]])
// CHECK: fusionInfo = "test.assert"
// The guard becomes a forwarding compiled region: it takes region A's chain and
// the input, forwards the input through, and carries the "mgp.guard" label.
// CHECK: %[[G:[a-zA-Z0-9_]+]] = mogg.generic.compiled.region (%[[A]], %[[INT]]) : (%{{.*}}: !mef.chain, %{{.*}}: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
// CHECK: fusionInfo = "mgp.guard"
// Region B consumes the guard region's forwarded result.
// CHECK: mogg.generic.compiled.region (%[[G]])
// CHECK: fusionInfo = "test.b"
// CHECK: mgp.device_graph.staging.yield

mgp.model @dg_guard : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  // Region A: produces a chain (mirrors the `mgp.assert` region that feeds a
  // guard in the real graph).
  %ca = mogg.generic.compiled.region (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mef.chain {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %r = mogg.async.ready : !mef.chain
    mogg.output %r : !mef.chain
  } fusionInfo = "test.assert" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
  // Guard: gates the model input on region A's chain. Not a compiled region, so
  // the current pass leaves it behind and forms the cycle.
  %g = mgp.guard %ca, %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
  // Region B: consumes the guarded value.
  %b = mogg.generic.compiled.region (%g) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %ch = mogg.chain.dummy() : () -> !mef.chain
    %bf = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
    %t = mgp.tensor.create %ch, %bf, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%t) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  } fusionInfo = "test.b" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
  mef.output %ch0, %b : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}

// -----

// A model WITHOUT `isDeviceGraph` is left untouched: no staging op is created
// and the compiled region stays directly in the execute block.

// CHECK-LABEL: mgp.model @not_dg
// CHECK: mogg.generic.compiled.region
// CHECK-NOT: mgp.device_graph.staging

mgp.model @not_dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.buffer<<"gpu", 0>>) {
  %a = mogg.generic.compiled.region (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%b) : !mgp.buffer<<"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
  } fusionInfo = "test.a" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
  mef.output %ch0, %a : !mef.chain, !mgp.buffer<<"gpu", 0>>
}
