// RUN: mo-opt %s --mogg-import-packages --mo-to-mogg --mogg-pipeline \
// RUN:   --mogg-to-mgp | FileCheck %s

// An `mo.sequence` tagged with a non-default stream id lowers by binding its
// body to a `mgp.device_context.select_stream` view of the base device context
// and bracketing it with fork/join `mgp.device_wait` ops. The body's compiled
// regions take the VIEW context (not the base block arg), which is also what
// keeps MGP sequence fusion from merging them with main-stream work.

// CHECK-LABEL: mgp.model @seq
// The body's device context is a select_stream view of the base gpu:0 context.
// CHECK: %[[VIEW:.+]] = mgp.device_context.select_stream[1] %{{.*}} : !mgp.device_context<<"gpu", 0>>
// Fork: the side stream (view) waits before the body runs; it consumes VIEW.
// CHECK: mgp.device_wait{{.*}}%[[VIEW]]
// Join: a second device_wait brackets the body on the way out.
// CHECK: mgp.device_wait
mo.graph @seq(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:0>) -> !mo.tensor<[2, 2], f32, gpu:0> {
  %s = mo.sequence[1] (%a : !mo.tensor<[2, 2], f32, gpu:0>) (%arg) -> (!mo.tensor<[2, 2], f32, gpu:0>) {
    %r = mo.relu(%arg) : !mo.tensor<[2, 2], f32, gpu:0>
    %r2 = mo.relu(%r) : !mo.tensor<[2, 2], f32, gpu:0>
    mo.yield %r2 : !mo.tensor<[2, 2], f32, gpu:0>
  }
  %main = mo.relu(%b) : !mo.tensor<[2, 2], f32, gpu:0>
  %out = mo.add(%main, %s) : !mo.tensor<[2, 2], f32, gpu:0>
  mo.output %out : !mo.tensor<[2, 2], f32, gpu:0>
}

// A replicated (multi-device) body binds one select_stream view per device.

// CHECK-LABEL: mgp.model @seq2dev
// CHECK-DAG: mgp.device_context.select_stream[1] %{{.*}} : !mgp.device_context<<"gpu", 0>>
// CHECK-DAG: mgp.device_context.select_stream[1] %{{.*}} : !mgp.device_context<<"gpu", 1>>
mo.graph @seq2dev(%a: !mo.tensor<[2, 2], f32, gpu:0>, %b: !mo.tensor<[2, 2], f32, gpu:1>) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>) {
  %r:2 = mo.sequence[1] (%a, %b : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>) (%x, %y) -> (!mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>) {
    %rx = mo.relu(%x) : !mo.tensor<[2, 2], f32, gpu:0>
    %ry = mo.relu(%y) : !mo.tensor<[2, 2], f32, gpu:1>
    mo.yield %rx, %ry : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>
  }
  mo.output %r#0, %r#1 : !mo.tensor<[2, 2], f32, gpu:0>, !mo.tensor<[2, 2], f32, gpu:1>
}
