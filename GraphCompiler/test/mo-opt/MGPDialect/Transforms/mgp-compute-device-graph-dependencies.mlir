// RUN: mo-opt %s --mgp-compute-device-graph-dependencies --allow-unregistered-dialect --split-input-file | FileCheck %s

// Each `mogg.generic.compiled.region` staged by `mgp-stage-device-graphs` is
// replaced 1:1 by an `mgp.device_graph.dependencies` op: the region body is
// dissolved into the new op's body (referencing enclosing SSA directly), the
// inter-region ordering becomes explicit `!mgp.device_graph.node` edges, and
// `mogg.async` pack/unpack ops are inserted to bridge the dissolved boundary.

// CHECK-LABEL: mgp.model @dg
// CHECK: execute ({{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>) ->
// The isolated staging op takes the input as an operand / async block arg, which
// it unpacks at the top of the body.
// CHECK: mgp.device_graph.staging inputs(%[[IN]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INA:[a-zA-Z0-9_]+]]: !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>):
// CHECK: %[[INU:.*]] = mogg.async.unpack(%[[INA]])

// Region A: no producer node -> empty `dependencies()`. Its body packs the
// staging-unpacked input back up so the verbatim `async.unpack` lines up.
// CHECK: %[[NODEA:.*]], %[[A:.*]] = mgp.device_graph.dependencies dependencies() {
// CHECK: mogg.async.pack(%[[INU]])
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.buffer<<"gpu", 0>>
// CHECK: } : () -> (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>)

// Region B: depends on A -> `dependencies(%[[NODEA]])`. Its body references A's
// new body result directly (boundary dissolved).
// CHECK: %[[NODEB:.*]], %[[B:.*]] = mgp.device_graph.dependencies dependencies(%[[NODEA]]) {
// CHECK: mogg.async.pack(%[[A]])
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } : (!mgp.device_graph.node) -> (!mgp.device_graph.node, !mgp.tensor<2x3xf32, <"gpu", 0>>)

// The staging yield packs B's body result; the op result is the plain tensor.
// CHECK: mogg.async.pack(%[[B]])
// CHECK: mgp.device_graph.staging.yield %{{.*}} : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
// CHECK-NOT: mogg.generic.compiled.region

mgp.model @dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %staging = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_a: !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>):
    %in_u = mogg.async.unpack(%in_a) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %a = mogg.generic.compiled.region (%in_u) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
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
    %out_p = mogg.async.pack(%b) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    mgp.device_graph.staging.yield %out_p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
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
       %in_a: !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>):
    %in_u = mogg.async.unpack(%in_a) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %a = mogg.generic.compiled.region (%in_u) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
      %b = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%b) : (!mgp.buffer<<"gpu", 0>>) -> !mogg.async<!mgp.buffer<<"gpu", 0>>>
      mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
    } fusionInfo = "test.a" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
    %out_p = mogg.async.pack(%a) : (!mgp.buffer<<"gpu", 0>>) -> !mogg.async<!mgp.buffer<<"gpu", 0>>>
    mgp.device_graph.staging.yield %out_p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
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
// CHECK: %[[INU:.*]] = mogg.async.unpack(%{{.*}})
// The consumer is emitted first: it carries a (forward) dependency edge and
// yields the tensor.
// CHECK: mgp.device_graph.dependencies dependencies(%{{[a-zA-Z0-9_]+}}) {
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } : (!mgp.device_graph.node) -> (!mgp.device_graph.node, !mgp.tensor<2x3xf32, <"gpu", 0>>)
// The producer is emitted second with no dependencies; its body packs the
// staging-unpacked input and yields the buffer.
// CHECK: %[[NODEA:.*]], %[[A:.*]] = mgp.device_graph.dependencies dependencies() {
// CHECK: mogg.async.pack(%[[INU]])
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.buffer<<"gpu", 0>>
// CHECK: } : () -> (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>)
// CHECK-NOT: mogg.generic.compiled.region

mgp.model @dg_reversed : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %staging = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_a: !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>):
    %in_u = mogg.async.unpack(%in_a) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = mogg.generic.compiled.region (%a) : (%arg: !mgp.buffer<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.buffer<<"gpu", 0>>>) -> !mgp.buffer<<"gpu", 0>>
      %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
      %ch = mogg.chain.dummy() : () -> !mef.chain
      %t = mgp.tensor.create %ch, %u, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%t) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = "test.b" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
    %a = mogg.generic.compiled.region (%in_u) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
      %bb = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%bb) : (!mgp.buffer<<"gpu", 0>>) -> !mogg.async<!mgp.buffer<<"gpu", 0>>>
      mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
    } fusionInfo = "test.a" {device = #M.device_ref<"gpu", 0>, inOutSignature = #M.inout_sig<"">, isAsync = false}
    %out_p = mogg.async.pack(%b) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    mgp.device_graph.staging.yield %out_p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %staging : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}
