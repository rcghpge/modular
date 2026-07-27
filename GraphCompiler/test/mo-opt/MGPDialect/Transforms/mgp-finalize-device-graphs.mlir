// RUN: mo-opt %s --mgp-finalize-device-graphs --allow-unregistered-dialect --split-input-file | FileCheck %s

// Stage 6 of device-graph synthesis. The `mgp.device_graph.staging` op of an
// `isDeviceGraph` model is lifted into a `mgp.device_graph.create` wrapped in a
// `mogg.generic.compiled.region` (so it compiles via the kernel path), plus a
// bare `mgp.device_graph.execute` that replays the graph; `mef.output` is
// rewired to the execute op's output.

// CHECK-LABEL: mgp.model @dg
// CHECK: execute (%[[CH0:[a-zA-Z0-9_]+]]: !mef.chain, %[[CTX:[a-zA-Z0-9_]+]]: !mgp.context, %{{.+}}: !mgp.opaque<"weights_registry">, %[[GPU:[a-zA-Z0-9_]+]]: !mgp.device_context<<"gpu", 0>>, %[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>)

// The build work is wrapped in a compiled region (context + device context +
// inputs threaded in via the async ABI, unpacked inside); it yields the built
// graph as its single result.
// CHECK: %[[GRAPH:[a-zA-Z0-9_]+]] = mogg.generic.compiled.region (%[[CTX]], %[[GPU]], %[[IN]]) : (%{{.+}}: !mgp.context, %{{.+}}: !mgp.device_context<<"gpu", 0>>, %{{.+}}: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>> {
// CHECK: mogg.async.unpack
// The stateless create op (no chain/slot); the `!mgp.context` is a build-time
// block arg after the builder, not part of the signature.
// CHECK: mgp.device_graph.create %{{.+}}, %{{.+}} inputs(%{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%[[B:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder, %{{.+}}: !mgp.context, %{{.+}}: !mgp.tensor<2x3xf32, <"gpu", 0>>):
// CHECK: %[[NODE:[a-zA-Z0-9_]+]], %[[R:[a-zA-Z0-9_]+]] = mgp.device_graph.dependencies builder(%[[B]]) dependencies() {
// CHECK: mgp.device_graph.yield %[[R]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>
// The built graph is packed and yielded from the region.
// CHECK: %[[PACKED:[a-zA-Z0-9_]+]] = mogg.async.pack
// CHECK: mogg.output %[[PACKED]] : !mogg.async<!mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>>

// Replay (outside the region), ordered after the build by the graph value.
// Device graphs carry no chain, so the graph signature is chain-free; the
// execute op produces its own fresh completion chain result before the value
// output. `mef.output` keeps the model's original chain.
// CHECK: %{{[a-zA-Z0-9_]+}}, %[[RES:[a-zA-Z0-9_]+]] = mgp.device_graph.execute %[[GRAPH]], %[[GPU]] inputs(%[[IN]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>, !mgp.device_context<<"gpu", 0>> -> !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: mef.output %[[CH0]], %[[RES]] : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NOT: mgp.device_graph.staging

mgp.model @dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    %node, %r = mgp.device_graph.dependencies builder(%builder) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      mgp.device_graph.dependencies.yield %in_t : !mgp.tensor<2x3xf32, <"gpu", 0>>
    } : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %r : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}

// -----

// A model WITHOUT `isDeviceGraph` is left untouched: no create/execute is
// introduced and the staging op stays.

// CHECK-LABEL: mgp.model @not_dg
// CHECK: mgp.device_graph.staging
// CHECK-NOT: mgp.device_graph.create

mgp.model @not_dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    mgp.device_graph.staging.yield %in_t : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// The staged body is a graph region, so an op may textually precede the def of a
// value it uses (here `test.consume_size` precedes `test.produce_size`). Finalize
// splices the staged ops into the SSACFG `mgp.device_graph.create` build body,
// where dominance is enforced, so the pass must topologically reorder the def
// ahead of its use. Regression guard for the `mgp.buffer.plan` ordering fix:
// without the sort this fails the verifier with "does not dominate this use"
// (`mgp-dynamic-memory-planning` emits `mgp.buffer.plan` before its size ops).

// CHECK-LABEL: mgp.model @dg_unordered
// CHECK: mgp.device_graph.create
// CHECK: %[[SZ:[a-zA-Z0-9_]+]] = "test.produce_size"
// CHECK: "test.consume_size"(%[[SZ]])
mgp.model @dg_unordered : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    // Use precedes def -- legal in the staging graph region, illegal once
    // spliced into the SSACFG create body unless topologically reordered.
    %consumed = "test.consume_size"(%sz) : (index) -> index
    %sz = "test.produce_size"() : () -> index
    %node, %r = mgp.device_graph.dependencies builder(%builder) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      mgp.device_graph.dependencies.yield %in_t : !mgp.tensor<2x3xf32, <"gpu", 0>>
    } : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %r : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}

// -----

// Device graphs have no chain concept. When the staged body has a chain input
// (dead pass-through) and yields a chain output (host-side sequencing), finalize
// must (a) keep `!mef.chain` out of the `mgp.device_graph` signature entirely and
// (b) give `mgp.device_graph.execute` its own fresh completion chain result, then
// rewire the staging op's chain result to it.

// CHECK-LABEL: mgp.model @dg_chain
// The graph signature is chain-free on BOTH sides (inputs and results).
// CHECK: mgp.device_graph.create %{{.+}}, %{{.+}} inputs(%{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK: } : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>
// The execute op produces a completion chain (result #0) + the value output; the
// staging op's chain result is rewired to that chain (fed to `mef.output`).
// CHECK: %[[CHAIN:[a-zA-Z0-9_]+]], %[[RES:[a-zA-Z0-9_]+]] = mgp.device_graph.execute %{{.+}}, %{{.+}} inputs(%{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>) : !mgp.device_graph<(!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>>, !mgp.device_context<<"gpu", 0>> -> !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: mef.output %[[CHAIN]], %[[RES]] : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
mgp.model @dg_chain : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  // Staged I/O carries a chain input (%ch0, dead in-body) and a chain output.
  %out:2 = mgp.device_graph.staging inputs(%ch0, %in : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %ch_in: !mef.chain,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    %node, %r = mgp.device_graph.dependencies builder(%builder) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      mgp.device_graph.dependencies.yield %in_t : !mgp.tensor<2x3xf32, <"gpu", 0>>
    } : !mgp.tensor<2x3xf32, <"gpu", 0>>
    // Host-side "done" chain, yielded as a graph output today; must be dropped.
    %done = "test.make_chain"() : () -> !mef.chain
    mgp.device_graph.staging.yield %done, %r : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %out#0, %out#1 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
} {isDeviceGraph}
