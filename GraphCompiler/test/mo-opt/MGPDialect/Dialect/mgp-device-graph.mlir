// RUN: mo-opt %s --allow-unregistered-dialect | FileCheck %s

// Smoke test: parse, verify, and round-trip-print the device-graph ops.

// CHECK-LABEL: func.func @device_graph_round_trip
func.func @device_graph_round_trip(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>,
    %wt: !mgp.buffer<<"gpu", 0>>,
    %out: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {

  // The body's input block args carry the input operand types verbatim (bare
  // `T`, no `!mogg.async` wrapping). The graph's output is value-returning: the
  // body yields it, and its type forms the signature's results.
  // CHECK: %[[GRAPH:[a-zA-Z0-9_]+]] = mgp.device_graph.create %{{.+}}, %{{.+}} inputs(%{{.+}}, %{{.+}} : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder, %{{.+}}: !mgp.context, %[[INB:[a-zA-Z0-9_]+]]: !mgp.buffer<<"gpu", 0>>, %{{.+}}: !mgp.buffer<<"gpu", 0>>):
  // CHECK: mgp.device_graph.yield %[[INB]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>>>
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in, %wt : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mgp.buffer<<"gpu", 0>>,
         %wt_b: !mgp.buffer<<"gpu", 0>>):
      mgp.device_graph.yield %in_b : !mgp.buffer<<"gpu", 0>>
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>

  // Value-returning: replay surfaces the graph output as its sole result; there
  // is no chain result (the output carries the data dependency) and no
  // destination `outputs` operand.
  // CHECK: %{{.+}} = mgp.device_graph.execute %[[GRAPH]], %{{.+}} inputs(%{{.+}}, %{{.+}} : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>>>, !mgp.device_context<<"gpu", 0>> -> !mgp.buffer<<"gpu", 0>>
  %res = mgp.device_graph.execute
      %graph, %dctx
      inputs(%in, %wt : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>)
      : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
        !mgp.device_context<<"gpu", 0>>
        -> !mgp.buffer<<"gpu", 0>>

  // Neither create nor execute produces a chain; the model output chain is the
  // incoming chain.
  return %ch : !mef.chain
}

// An empty `inputs()` group should parse and round-trip.

// CHECK-LABEL: func.func @device_graph_empty_groups
func.func @device_graph_empty_groups(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>) -> !mef.chain {

  // CHECK: mgp.device_graph.create %{{.+}}, %{{.+}} inputs() {
  // CHECK: } : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<() -> ()>
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs() {
    ^bb0(%builder: !mgp.device_graph.builder, %ctx_b: !mgp.context):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<() -> ()>

  // Empty `inputs()` and no graph outputs: the op has zero results and the
  // `-> ` result-type clause is omitted on print.
  // CHECK: mgp.device_graph.execute %{{.+}}, %{{.+}} inputs() : !mgp.device_graph<() -> ()>, !mgp.device_context<<"gpu", 0>>
  // CHECK-NOT: ->
  mgp.device_graph.execute
      %graph, %dctx
      inputs()
      : !mgp.device_graph<() -> ()>, !mgp.device_context<<"gpu", 0>>

  return %ch : !mef.chain
}

// A single-input build body: the block args carry the input operand types
// verbatim (bare `T`).

// CHECK-LABEL: func.func @device_graph_plain_args_ok
func.func @device_graph_plain_args_ok(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {

  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder, %{{.+}}: !mgp.context, %[[INB:[a-zA-Z0-9_]+]]: !mgp.buffer<<"gpu", 0>>):
  // CHECK: mgp.device_graph.yield %[[INB]] : !mgp.buffer<<"gpu", 0>>
  %graph = mgp.device_graph.create
      %ctx, %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %ctx_b: !mgp.context,
         %in_b: !mgp.buffer<<"gpu", 0>>):
      mgp.device_graph.yield %in_b : !mgp.buffer<<"gpu", 0>>
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>

  return %ch : !mef.chain
}
