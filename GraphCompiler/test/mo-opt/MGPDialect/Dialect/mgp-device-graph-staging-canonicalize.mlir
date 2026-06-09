// RUN: mo-opt %s --split-input-file --allow-unregistered-dialect --canonicalize | FileCheck %s

// Canonicalizations for `mgp.device_graph.staging`:
//   1. unused results are dropped (along with the matching `staging.yield`
//      operands), and
//   2. duplicated `staging.yield` operands are de-duplicated, collapsing the
//      duplicate results onto a single one.
//
// The bodies are self-contained (no `inputs`) so the cases focus on the
// result/yield rewrites; the op stays `IsolatedFromAbove`.

// An unused staged result is removed; only the used value is yielded.
// CHECK-LABEL: func.func @remove_unused_result
func.func @remove_unused_result() -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
  // CHECK: %[[OUT:.+]] = mgp.device_graph.staging inputs() {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[USED:.+]] = "test.work"
  // CHECK: %[[DEAD:.+]] = "test.dead"
  // CHECK: mgp.device_graph.staging.yield %[[USED]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: return %[[OUT]]
  %used, %unused = mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = "test.dead"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %a, %b : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  return %used : !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// The same value yielded twice collapses to a single result; both uses point
// at it.
// CHECK-LABEL: func.func @deduplicate_results
func.func @deduplicate_results() {
  // CHECK: %[[OUT:.+]] = mgp.device_graph.staging inputs() {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[V:.+]] = "test.work"
  // CHECK: mgp.device_graph.staging.yield %[[V]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: "test.use"(%[[OUT]], %[[OUT]])
  %first, %second = mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %a, %a : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  "test.use"(%first, %second) : (!mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
  return
}

// -----

// Both patterns compose: a duplicated-and-unused result set reduces to the one
// distinct used value.
// CHECK-LABEL: func.func @dedup_and_remove_unused
func.func @dedup_and_remove_unused() -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
  // CHECK: %[[OUT:.+]] = mgp.device_graph.staging inputs() {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[USED:.+]] = "test.work"
  // CHECK: mgp.device_graph.staging.yield %[[USED]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: return %[[OUT]]
  %a0, %a1, %dead = mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = "test.dead"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %a, %a, %b : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  return %a0 : !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// Negative case: all results used and distinct -- left unchanged.
// CHECK-LABEL: func.func @no_change
func.func @no_change() {
  // CHECK: %[[A:.+]]:2 = mgp.device_graph.staging inputs() {
  // CHECK: %[[X:.+]] = "test.work"
  // CHECK: %[[Y:.+]] = "test.work2"
  // CHECK: mgp.device_graph.staging.yield %[[X]], %[[Y]] : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: "test.use"(%[[A]]#0, %[[A]]#1)
  %x, %y = mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = "test.work2"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %a, %b : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  "test.use"(%x, %y) : (!mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
  return
}

// -----

// Folder for `mgp.device_graph.dependencies`: duplicated dependency edges are
// redundant (listing a predecessor twice does not change ordering), so the
// repeats are dropped, keeping the first occurrence of each.
// CHECK-LABEL: func.func @deduplicate_dependencies
func.func @deduplicate_dependencies() {
  // CHECK: %[[A:.+]] = "test.work"
  // CHECK: %[[B:.+]] = "test.work2"
  // CHECK: mgp.device_graph.dependencies dependencies(%[[A]], %[[B]]) {
  // CHECK: } : (!mgp.device_graph.node, !mgp.device_graph.node) ->
  %a = "test.work"() : () -> !mgp.device_graph.node
  %b = "test.work2"() : () -> !mgp.device_graph.node
  %node, %r = mgp.device_graph.dependencies dependencies(%a, %b, %a) {
    %v = "test.body"() : () -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
  } : (!mgp.device_graph.node, !mgp.device_graph.node, !mgp.device_graph.node)
    -> (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>)
  return
}

// -----

// Negative case: all dependency edges already distinct -- left unchanged.
// CHECK-LABEL: func.func @no_change_dependencies
func.func @no_change_dependencies() {
  // CHECK: mgp.device_graph.dependencies dependencies(%{{.+}}, %{{.+}}) {
  // CHECK: } : (!mgp.device_graph.node, !mgp.device_graph.node) ->
  %a = "test.work"() : () -> !mgp.device_graph.node
  %b = "test.work2"() : () -> !mgp.device_graph.node
  %node, %r = mgp.device_graph.dependencies dependencies(%a, %b) {
    %v = "test.body"() : () -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
  } : (!mgp.device_graph.node, !mgp.device_graph.node)
    -> (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>)
  return
}
