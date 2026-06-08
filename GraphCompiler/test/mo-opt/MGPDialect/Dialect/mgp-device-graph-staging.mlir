// RUN: mo-opt %s --allow-unregistered-dialect | FileCheck %s
// RUN: mo-opt %s --allow-unregistered-dialect | mo-opt --allow-unregistered-dialect

// Round-trip the transient device-graph staging scaffold ops:
// mgp.device_graph.dependencies (+ .yield) and mgp.device_graph.staging
// (+ .yield).

// CHECK-LABEL: func.func @dependencies_round_trip
func.func @dependencies_round_trip(%in: !mgp.buffer<<"gpu", 0>>)
    -> !mgp.device_graph.node {
  // No predecessors, one body result; the body references the func arg
  // directly -- the op is not IsolatedFromAbove.
  // CHECK: %[[N0:.+]], %[[R0:.+]] = mgp.device_graph.dependencies dependencies() {
  // CHECK: %[[V:.+]] = "test.work"(%{{.+}})
  // CHECK: mgp.device_graph.dependencies.yield %[[V]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : () -> (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>)
  %n0, %r0 = mgp.device_graph.dependencies dependencies() {
    %v = "test.work"(%in) : (!mgp.buffer<<"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
  } : () -> (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>)

  // A real dependency edge on %n0, and no body results (the operandless yield
  // is elided on print).
  // CHECK: %[[N1:.+]] = mgp.device_graph.dependencies dependencies(%[[N0]]) {
  // CHECK: } : (!mgp.device_graph.node) -> !mgp.device_graph.node
  %n1 = mgp.device_graph.dependencies dependencies(%n0) {
    "test.work2"(%r0) : (!mgp.buffer<<"gpu", 0>>) -> ()
    mgp.device_graph.dependencies.yield
  } : (!mgp.device_graph.node) -> !mgp.device_graph.node

  return %n1 : !mgp.device_graph.node
}

// CHECK-LABEL: func.func @staging_round_trip
func.func @staging_round_trip(%in: !mgp.tensor<2x3xf32, <"gpu", 0>>)
    -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
  // IsolatedFromAbove: the free value enters as an operand and a `!mogg.async<T>`
  // block arg (unpacked inside); the body yields `!mogg.async<T>` while the op
  // result is the plain `T`.
  // CHECK: %{{.+}} = mgp.device_graph.staging inputs(%{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder, %[[INA:.+]]: !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>):
  // CHECK: %[[INU:.+]] = mogg.async.unpack(%[[INA]])
  // CHECK: mgp.device_graph.staging.yield %{{.+}} : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_a: !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>):
    %in_u = mogg.async.unpack(%in_a) : (!mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %n, %t = mgp.device_graph.dependencies dependencies() {
      mgp.device_graph.dependencies.yield %in_u : !mgp.tensor<2x3xf32, <"gpu", 0>>
    } : () -> (!mgp.device_graph.node, !mgp.tensor<2x3xf32, <"gpu", 0>>)
    %p = mogg.async.pack(%t) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    mgp.device_graph.staging.yield %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  return %out : !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// A `dependencies` op may reference a node defined LATER in the staging body:
// the body is a graph region (`HasOnlyGraphRegion`), so block order does not
// imply dominance. The `| mo-opt` round-trip RUN line is what enforces this --
// an SSACFG region would reject the use-before-def.

// CHECK-LABEL: func.func @forward_reference
func.func @forward_reference() {
  // CHECK: mgp.device_graph.staging inputs() {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // The consumer references %[[NA]], which is only defined by the next op.
  // CHECK: mgp.device_graph.dependencies dependencies(%[[NA:.+]]) {
  // CHECK: %[[NA]] = mgp.device_graph.dependencies dependencies() {
  mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %nb = mgp.device_graph.dependencies dependencies(%na) {
      mgp.device_graph.dependencies.yield
    } : (!mgp.device_graph.node) -> !mgp.device_graph.node
    %na = mgp.device_graph.dependencies dependencies() {
      mgp.device_graph.dependencies.yield
    } : () -> !mgp.device_graph.node
    mgp.device_graph.staging.yield
  }
  return
}
