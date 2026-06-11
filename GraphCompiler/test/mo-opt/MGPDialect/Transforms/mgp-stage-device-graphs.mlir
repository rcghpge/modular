// RUN: mo-opt %s --mgp-stage-device-graphs --allow-unregistered-dialect --split-input-file | FileCheck %s

// An `isDeviceGraph` model: every `mogg.generic.compiled.region` in the execute
// block is moved into a single `IsolatedFromAbove` `mgp.device_graph.staging`
// op. Free values the regions reference (here the model input) become operands
// passed in as `!mogg.async<T>` block args; values consumed outside become the
// op's results (type `T`), yielded as `!mogg.async<T>`.

// CHECK-LABEL: mgp.model @dg
// CHECK: execute (%{{.*}}: !mef.chain{{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>) ->
// The staging op takes the free value as an operand and exposes it to the body
// as a `!mogg.async<T>` block arg alongside the device-graph builder.
// CHECK: %[[STAGING:.*]] = mgp.device_graph.staging inputs(%[[IN]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INA:[a-zA-Z0-9_]+]]: !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>):
// The async input is unpacked, then fed to region A; region B consumes A's result.
// CHECK: %[[INU:.*]] = mogg.async.unpack(%[[INA]])
// CHECK: %[[A:.*]] = mogg.generic.compiled.region (%[[INU]])
// CHECK: %[[B:.*]] = mogg.generic.compiled.region (%[[A]])
// The escaping value is packed and yielded as async; the result is the plain `T`.
// CHECK: %[[P:.*]] = mogg.async.pack(%[[B]])
// CHECK: mgp.device_graph.staging.yield %[[P]] : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
// CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: mef.output %{{.*}}, %[[STAGING]] : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>

mgp.model @dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %a = mogg.generic.compiled.region (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%b) : (!mgp.buffer<<"gpu", 0>>) -> !mogg.async<!mgp.buffer<<"gpu", 0>>>
    mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
  } fusionInfo = "test.a" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
  %b = mogg.generic.compiled.region (%a) : (%arg: !mgp.buffer<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.buffer<<"gpu", 0>>>) -> !mgp.buffer<<"gpu", 0>>
    %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %ch = mogg.chain.dummy() : () -> !mef.chain
    %t = mgp.tensor.create %ch, %u, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%t) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
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
    %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%b) : (!mgp.buffer<<"gpu", 0>>) -> !mogg.async<!mgp.buffer<<"gpu", 0>>>
    mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
  } fusionInfo = "test.a" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
  mef.output %ch0, %a : !mef.chain, !mgp.buffer<<"gpu", 0>>
}
