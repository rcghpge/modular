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
func.func @deduplicate_dependencies() -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: %[[A:.+]] = "test.work"
  // CHECK: %[[B:.+]] = "test.work2"
  // CHECK: mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[A]], %[[B]]) {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %a = "test.work"() : () -> !mgp.device_graph.node
  %b = "test.work2"() : () -> !mgp.device_graph.node
  %node, %r = mgp.device_graph.dependencies builder(%bldr) dependencies(%a, %b, %a) {
  ^bb0(%builder: !mgp.device_graph.builder):
    %v = "test.body"() : () -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>
  return %r : !mgp.buffer<<"gpu", 0>>
}

// -----

// Negative case: all dependency edges already distinct -- left unchanged.
// CHECK-LABEL: func.func @no_change_dependencies
func.func @no_change_dependencies() -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %a = "test.work"() : () -> !mgp.device_graph.node
  %b = "test.work2"() : () -> !mgp.device_graph.node
  %node, %r = mgp.device_graph.dependencies builder(%bldr) dependencies(%a, %b) {
  ^bb0(%builder: !mgp.device_graph.builder):
    %v = "test.body"() : () -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>
  return %r : !mgp.buffer<<"gpu", 0>>
}

// -----

// An unused `dependencies` body result is removed (along with the matching
// `dependencies.yield` operand); the synthesized `node` result is always kept.
// CHECK-LABEL: func.func @remove_unused_dependency_result
func.func @remove_unused_dependency_result() -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: %[[A:.+]] = "test.dep"
  // CHECK: %[[NODE:.+]], %[[OUT:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[A]]) {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[USED:.+]] = "test.work"
  // CHECK: %[[DEAD:.+]] = "test.dead"
  // CHECK: mgp.device_graph.dependencies.yield %[[USED]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>
  // CHECK: return %[[OUT]]
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %a = "test.dep"() : () -> !mgp.device_graph.node
  %node, %used, %unused = mgp.device_graph.dependencies builder(%bldr) dependencies(%a) {
  ^bb0(%builder: !mgp.device_graph.builder):
    %x = "test.work"() : () -> !mgp.buffer<<"gpu", 0>>
    %y = "test.dead"() : () -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %x, %y : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  return %used : !mgp.buffer<<"gpu", 0>>
}

// -----

// The same value yielded twice collapses to a single body result; both uses
// point at it. The `node` result is preserved.
// CHECK-LABEL: func.func @deduplicate_dependency_results
func.func @deduplicate_dependency_results() {
  // CHECK: %[[NODE:.+]], %[[OUT:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[V:.+]] = "test.work"
  // CHECK: mgp.device_graph.dependencies.yield %[[V]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>
  // CHECK: "test.use"(%[[OUT]], %[[OUT]])
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %node, %first, %second = mgp.device_graph.dependencies builder(%bldr) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %x = "test.work"() : () -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %x, %x : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  "test.use"(%first, %second) : (!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> ()
  return
}

// -----

// Both patterns compose: a duplicated-and-unused body-result set reduces to the
// one distinct used value, keeping the `node` result.
// CHECK-LABEL: func.func @dedup_and_remove_unused_dependency_results
func.func @dedup_and_remove_unused_dependency_results() -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: %[[NODE:.+]], %[[OUT:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[USED:.+]] = "test.work"
  // CHECK: mgp.device_graph.dependencies.yield %[[USED]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>
  // CHECK: return %[[OUT]]
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %node, %a0, %a1, %dead = mgp.device_graph.dependencies builder(%bldr) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %x = "test.work"() : () -> !mgp.buffer<<"gpu", 0>>
    %y = "test.dead"() : () -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %x, %x, %y : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  return %a0 : !mgp.buffer<<"gpu", 0>>
}

// -----

// Negative case: all body results used and distinct -- left unchanged.
// CHECK-LABEL: func.func @no_change_dependency_results
func.func @no_change_dependency_results() {
  // CHECK: %[[NODE:.+]], %[[R:.+]]:2 = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[X:.+]] = "test.work"
  // CHECK: %[[Y:.+]] = "test.work2"
  // CHECK: mgp.device_graph.dependencies.yield %[[X]], %[[Y]] : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  // CHECK: "test.use"(%[[NODE]], %[[R]]#0, %[[R]]#1)
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %node, %x, %y = mgp.device_graph.dependencies builder(%bldr) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"() : () -> !mgp.buffer<<"gpu", 0>>
    %b = "test.work2"() : () -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %a, %b : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  "test.use"(%node, %x, %y) : (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> ()
  return
}

// -----

// Forwarding: a body result whose `dependencies.yield` operand is produced
// OUTSIDE the body (the op is not IsolatedFromAbove) is a pure pass-through.
// Its uses are rewritten to the outer value directly, then the dead result is
// dropped, leaving only the synthesized `node`.
// CHECK-LABEL: func.func @forward_external_result
func.func @forward_external_result(%b: !mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: %[[OUTER:.+]] = "test.outer"
  // The op keeps only the `node` result; its body has no results, so the
  // operandless yield is elided on print.
  // CHECK: %[[NODE:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK-NEXT: }
  // CHECK: "test.keep"(%[[NODE]])
  // CHECK: return %[[OUTER]]
  %outer = "test.outer"() : () -> !mgp.buffer<<"gpu", 0>>
  %node, %r = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    mgp.device_graph.dependencies.yield %outer : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>
  "test.keep"(%node) : (!mgp.device_graph.node) -> ()
  return %r : !mgp.buffer<<"gpu", 0>>
}

// -----

// Mixed: only the externally-produced result is forwarded; the result produced
// inside the body is retained.
// CHECK-LABEL: func.func @forward_external_result_partial
func.func @forward_external_result_partial(%b: !mgp.device_graph.builder) -> (!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) {
  // CHECK: %[[OUTER:.+]] = "test.outer"
  // CHECK: %[[NODE:.+]], %[[R:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[INNER:.+]] = "test.work"
  // CHECK: mgp.device_graph.dependencies.yield %[[INNER]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>
  // CHECK: return %[[OUTER]], %[[R]]
  %outer = "test.outer"() : () -> !mgp.buffer<<"gpu", 0>>
  %node, %ext, %inner = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %w = "test.work"() : () -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %outer, %w : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  return %ext, %inner : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
}
