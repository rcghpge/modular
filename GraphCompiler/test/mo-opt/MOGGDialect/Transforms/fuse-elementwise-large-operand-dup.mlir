// RUN: mo-opt --mogg-import-packages --mo-to-mogg --mogg-fuse-elementwise --mogg-simplify-kernel-operands --canonicalize --cse --mogg-strip-kernel-decls %s | FileCheck %s
// RUN: mo-opt --mogg-import-packages --mo-to-mogg --mogg-fuse-elementwise=large-tensor-element-threshold=16384 --mogg-simplify-kernel-operands --canonicalize --cse --mogg-strip-kernel-decls %s | FileCheck %s --check-prefix=HIGHTHRESH

// Duplicating a device elementwise kernel into an extra consumer re-reads all of
// its tensor inputs once more, while materializing it costs one result write plus
// one read per consumer. For an op whose tensor-input bytes exceed its result
// bytes -- e.g. a binary `mo.add` of two large (> 4096-element) operands -- the
// duplication is a net memory-traffic loss, and in residual chains it compounds
// quadratically (JSC-30 / MSTDL-2759). The duplication gate therefore refuses to
// clone such an op into multiple consumer subgraphs and materializes it instead.

// A multi-use binary op over TWO large operands must NOT be duplicated: the Add
// is materialized into its own kernel and each consumer reads its result.
mo.graph @binaryAddLargeFanout(%arg0: !mo.tensor<[4096, 2], f32, gpu:0>, %arg1: !mo.tensor<[4096, 2], f32, gpu:0>) -> (!mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) {
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[4096, 2], f32, gpu:0>
  %1 = mo.tanh(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %2 = mo.relu(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  mo.output %1, %2 : !mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @binaryAddLargeFanout
// The Add is materialized exactly once in its own kernel...
// CHECK: %[[ADD:.*]] = mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NOT: mogg.call.elementwise["{{.*}}.Add"]
// ...and each consumer reads the materialized result rather than recomputing it.
// CHECK: mogg.kernel(%[[ADD]]
// CHECK: mogg.call.elementwise["{{.*}}.Tanh"]
// CHECK: mogg.kernel(%[[ADD]]
// CHECK: mogg.call.elementwise["{{.*}}.ReLU"]

// A multi-use UNARY op has at most one large operand, so re-reading it on
// duplication is not a net loss. It is still cloned into each consumer's fused
// subgraph (Exp appears in both kernels), preserving the existing fusion win.
mo.graph @unaryExpLargeFanout(%arg0: !mo.tensor<[4096, 2], f32, gpu:0>) -> (!mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) {
  %0 = mo.exp(%arg0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %1 = mo.tanh(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %2 = mo.relu(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  mo.output %1, %2 : !mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @unaryExpLargeFanout
// First consumer kernel fuses Exp as a prologue...
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Exp"]
// CHECK: mogg.call.elementwise["{{.*}}.Tanh"]
// ...and the second consumer kernel duplicates the Exp prologue too.
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Exp"]
// CHECK: mogg.call.elementwise["{{.*}}.ReLU"]

// With `large-tensor-element-threshold` raised above the operands' 8192-element
// count, the binary Add's operands are no longer "large", so the gate permits
// duplication again: the Add is cloned into each consumer's fused subgraph
// rather than materialized. This exercises the pass option.
// HIGHTHRESH-LABEL: mo.graph @binaryAddLargeFanout
// HIGHTHRESH: mogg.kernel
// HIGHTHRESH: mogg.call.elementwise["{{.*}}.Add"]
// HIGHTHRESH: mogg.call.elementwise["{{.*}}.Tanh"]
// HIGHTHRESH: mogg.kernel
// HIGHTHRESH: mogg.call.elementwise["{{.*}}.Add"]
// HIGHTHRESH: mogg.call.elementwise["{{.*}}.ReLU"]
