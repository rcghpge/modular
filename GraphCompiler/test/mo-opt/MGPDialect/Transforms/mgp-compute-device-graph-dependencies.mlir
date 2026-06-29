// RUN: mo-opt %s --mgp-compute-device-graph-dependencies --allow-unregistered-dialect --split-input-file | FileCheck %s

// Each `mogg.generic.compiled.region` staged by `mgp-stage-device-graphs` is
// replaced 1:1 by an `mgp.device_graph.dependencies` op: the region body is
// dissolved into the new op's body (referencing enclosing SSA directly), the
// inter-region ordering becomes explicit `!mgp.device_graph.node` edges, and
// `mogg.async` pack/unpack ops are inserted to bridge the dissolved boundary.

// CHECK-LABEL: mgp.model @dg
// CHECK: execute ({{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>) ->
// The isolated staging op takes the input as an operand and a bare-`T` block
// arg; no async unpack is introduced at the staging boundary.
// CHECK: mgp.device_graph.staging inputs(%[[IN]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%[[SB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>):

// Region A: no producer node -> empty `dependencies()`. Each dependencies op
// takes the staging builder as its `builder(...)` operand and carries a
// `!mgp.device_graph.builder` block arg. Its body packs the bare staging input
// up so the verbatim `async.unpack` lines up; no device context is used, so no
// `unrealized_conversion_cast` is emitted.
// CHECK: %[[NODEA:.*]], %[[A:.*]] = mgp.device_graph.dependencies builder(%[[SB]]) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK: mogg.async.pack(%[[INT]])
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.buffer<<"gpu", 0>>
// CHECK: } : !mgp.buffer<<"gpu", 0>>

// Region B: depends on A -> `dependencies(%[[NODEA]])`. Its body references A's
// new body result directly (boundary dissolved).
// CHECK: %[[NODEB:.*]], %[[B:.*]] = mgp.device_graph.dependencies builder(%[[SB]]) dependencies(%[[NODEA]]) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK: mogg.async.pack(%[[A]])
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } : !mgp.tensor<2x3xf32, <"gpu", 0>>

// The staging yield carries B's body result directly as bare `T`.
// CHECK: mgp.device_graph.staging.yield %[[B]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NOT: mogg.generic.compiled.region

mgp.model @dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %staging = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    %a = mogg.generic.compiled.region (%in_t) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
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
    mgp.device_graph.staging.yield %b : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %staging : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}

// -----

// A staged region that uses a device context: the conversion re-sources the
// context from the dependencies op's `!mgp.device_graph.builder` block arg via
// a `builtin.unrealized_conversion_cast`, so the body no longer references the
// outer (staging) device-context block arg. The region's ABI wraps the context
// as `!mogg.async<!mgp.device_context>`, so the cast is re-packed at the
// boundary and the body's verbatim `mogg.async.unpack` consumes it (the
// round-trip is folded later, in Stage 3).

// CHECK-LABEL: mgp.model @dg_ctx
// CHECK: mgp.device_graph.staging inputs({{.*}}) {
// CHECK-NEXT: ^bb0(%[[SB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder,
// The dependencies op takes the staging builder as its `builder(...)` operand.
// CHECK: mgp.device_graph.dependencies builder(%[[SB]]) dependencies() {
// CHECK-NEXT: ^bb0(%[[BUILDER:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// The device context is re-sourced from the body's builder block arg, not the
// outer context.
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BUILDER]] : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
// CHECK: %[[CTXP:.*]] = mogg.async.pack(%[[CAST]])
// CHECK: %[[CTXU:.*]] = mogg.async.unpack(%[[CTXP]])
// CHECK: "test.use_ctx"(%[[CTXU]])
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NOT: mogg.generic.compiled.region

mgp.model @dg_ctx : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %staging = mgp.device_graph.staging inputs(%in, %gpu : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.device_context<<"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>,
       %ctx_t: !mgp.device_context<<"gpu", 0>>):
    %a = mogg.generic.compiled.region (%in_t, %ctx_t) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>, %ctxarg: !mgp.device_context<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
      %ctx_in = mogg.async.unpack(%ctxarg) : (!mogg.async<!mgp.device_context<<"gpu", 0>>>) -> !mgp.device_context<<"gpu", 0>>
      %b = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      "test.use_ctx"(%ctx_in) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
      %ch = mogg.chain.dummy() : () -> !mef.chain
      %t = mgp.tensor.create %ch, %b, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%t) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = "test.ctx" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
    mgp.device_graph.staging.yield %a : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %staging : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}

// -----

// A model WITHOUT `isDeviceGraph` is left untouched: no conversion happens.

// CHECK-LABEL: mgp.model @not_dg
// CHECK: mogg.generic.compiled.region
// CHECK-NOT: mgp.device_graph.dependencies

mgp.model @not_dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.buffer<<"gpu", 0>>) {
  %staging = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    %a = mogg.generic.compiled.region (%in_t) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
      %b = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%b) : (!mgp.buffer<<"gpu", 0>>) -> !mogg.async<!mgp.buffer<<"gpu", 0>>>
      mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
    } fusionInfo = "test.a" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
    mgp.device_graph.staging.yield %a : !mgp.buffer<<"gpu", 0>>
  } -> !mgp.buffer<<"gpu", 0>>
  mef.output %ch0, %staging : !mef.chain, !mgp.buffer<<"gpu", 0>>
}

// -----

// Out-of-order graph region: the consumer region (%b) appears BEFORE the
// producer region (%a) it depends on. The staging body is a graph region, so
// this is legal, and conversion must not rely on topological block order. The
// consumer is converted first and ends up with a forward node edge / forward
// body reference to the producer's later-defined results -- legal in the
// enclosing graph region (the pass's own verifier run guards this).

// CHECK-LABEL: mgp.model @dg_reversed
// CHECK: mgp.device_graph.staging inputs(%{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%[[SB:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>):
// The consumer is emitted first: it carries a (forward) dependency edge and
// yields the tensor. Both ops take the staging builder as their operand.
// CHECK: mgp.device_graph.dependencies builder(%[[SB]]) dependencies(%{{[a-zA-Z0-9_]+}}) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } : !mgp.tensor<2x3xf32, <"gpu", 0>>
// The producer is emitted second with no dependencies; its body packs the bare
// staging input and yields the buffer.
// CHECK: %[[NODEA:.*]], %[[A:.*]] = mgp.device_graph.dependencies builder(%[[SB]]) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK: mogg.async.pack(%[[INT]])
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.buffer<<"gpu", 0>>
// CHECK: } : !mgp.buffer<<"gpu", 0>>
// CHECK-NOT: mogg.generic.compiled.region

mgp.model @dg_reversed : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %staging = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    %b = mogg.generic.compiled.region (%a) : (%arg: !mgp.buffer<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.buffer<<"gpu", 0>>>) -> !mgp.buffer<<"gpu", 0>>
      %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
      %ch = mogg.chain.dummy() : () -> !mef.chain
      %t = mgp.tensor.create %ch, %u, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%t) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = "test.b" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
    %a = mogg.generic.compiled.region (%in_t) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
      %bb = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%bb) : (!mgp.buffer<<"gpu", 0>>) -> !mogg.async<!mgp.buffer<<"gpu", 0>>>
      mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
    } fusionInfo = "test.a" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
    mgp.device_graph.staging.yield %b : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %staging : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}
