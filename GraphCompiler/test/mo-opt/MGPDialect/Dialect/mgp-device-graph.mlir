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

  // The body's input/output block args are `!mogg.async<T>` wrappers; the op
  // operands and the result-graph signature stay the plain buffer types `T`.
  // CHECK: %[[CH1:[a-zA-Z0-9_]+]], %[[GRAPH:[a-zA-Z0-9_]+]] = mgp.device_graph.create %{{.+}}, %{{.+}}[3], %{{.+}} inputs(%{{.+}}, %{{.+}} : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) outputs(%{{.+}} : !mgp.buffer<<"gpu", 0>>) {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder, %{{.+}}: !mogg.async<!mgp.buffer<<"gpu", 0>>>, %{{.+}}: !mogg.async<!mgp.buffer<<"gpu", 0>>>, %{{.+}}: !mogg.async<!mgp.buffer<<"gpu", 0>>>):
  // The operandless mgp.device_graph.yield terminator is elided on print.
  // CHECK: } : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>>>
  %ch1, %graph = mgp.device_graph.create
      %ch, %ctx[3], %dctx
      inputs(%in, %wt : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>)
      outputs(%out : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in_b: !mogg.async<!mgp.buffer<<"gpu", 0>>>,
         %wt_b: !mogg.async<!mgp.buffer<<"gpu", 0>>>,
         %out_b: !mogg.async<!mgp.buffer<<"gpu", 0>>>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>

  // CHECK: %{{.+}} = mgp.device_graph.execute %[[CH1]], %[[GRAPH]], %{{.+}} inputs(%{{.+}}, %{{.+}} : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) outputs(%{{.+}} : !mgp.buffer<<"gpu", 0>>) : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>>>, !mgp.device_context<<"gpu", 0>>
  %ch2 = mgp.device_graph.execute
      %ch1, %graph, %dctx
      inputs(%in, %wt : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>)
      outputs(%out : !mgp.buffer<<"gpu", 0>>)
      : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
        !mgp.device_context<<"gpu", 0>>

  return %ch2 : !mef.chain
}

// Empty `inputs()` and `outputs()` groups should parse and round-trip.

// CHECK-LABEL: func.func @device_graph_empty_groups
func.func @device_graph_empty_groups(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>) -> !mef.chain {

  // CHECK: mgp.device_graph.create %{{.+}}, %{{.+}}[7], %{{.+}} inputs() outputs() {
  // CHECK: } : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<() -> ()>
  %ch1, %graph = mgp.device_graph.create
      %ch, %ctx[7], %dctx
      inputs() outputs() {
    ^bb0(%builder: !mgp.device_graph.builder):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>> -> !mgp.device_graph<() -> ()>

  // CHECK: mgp.device_graph.execute %{{.+}}, %{{.+}}, %{{.+}} inputs() outputs() : !mgp.device_graph<() -> ()>, !mgp.device_context<<"gpu", 0>>
  %ch2 = mgp.device_graph.execute
      %ch1, %graph, %dctx
      inputs() outputs()
      : !mgp.device_graph<() -> ()>, !mgp.device_context<<"gpu", 0>>

  return %ch2 : !mef.chain
}

// A body that consumes its input/output args directly as the plain operand type
// `T` (rather than `!mogg.async<T>`) is also accepted, mirroring
// `mogg.generic.compiled.region`'s ABI-lowering flexibility.

// CHECK-LABEL: func.func @device_graph_plain_args_ok
func.func @device_graph_plain_args_ok(
    %ch: !mef.chain,
    %ctx: !mgp.context,
    %dctx: !mgp.device_context<<"gpu", 0>>,
    %in: !mgp.buffer<<"gpu", 0>>,
    %out: !mgp.buffer<<"gpu", 0>>) -> !mef.chain {

  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder, %{{.+}}: !mgp.buffer<<"gpu", 0>>, %{{.+}}: !mgp.buffer<<"gpu", 0>>):
  %ch1, %graph = mgp.device_graph.create
      %ch, %ctx[1], %dctx
      inputs(%in : !mgp.buffer<<"gpu", 0>>)
      outputs(%out : !mgp.buffer<<"gpu", 0>>) {
    ^bb0(%builder: !mgp.device_graph.builder,
         %in_b: !mgp.buffer<<"gpu", 0>>,
         %out_b: !mgp.buffer<<"gpu", 0>>):
      mgp.device_graph.yield
  } : !mgp.device_context<<"gpu", 0>>
      -> !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>

  return %ch1 : !mef.chain
}
