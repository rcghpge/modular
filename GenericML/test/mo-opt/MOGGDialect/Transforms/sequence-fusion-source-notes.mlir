// RUN: mo-opt %s --mgp-lower-primitives --mogg-sequence-fusion --mogg-outline-kernels --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mgp-dump-mojo 2>&1 | FileCheck %s

// Test that when ops with different source note metadata (FusedLoc) are
// sequence-fused, source notes are preserved in the emitted
// mogg_async_error call. See GEX-2678.
//
// After --mgp-lower-primitives, each mgp.assert becomes a
// GenericCompiledRegionOp with the op's FusedLoc metadata.
// --mogg-sequence-fusion merges adjacent regions into one kernel, creating a
// FusedLoc whose sub-locations carry the individual source notes.
// AsyncErrorOp::emitMojo collects all unique source notes and emits them
// together, along with a fusion summary tree when multiple ops are fused.

// The emitted source notes use a triple-quoted Mojo string and must contain:
// 1. Source note tracebacks from each fused op.
// 2. A fusion summary noting ops were fused, with a tree of combined ops.
// CHECK:       string.String("""
// CHECK:       trace_beta
// CHECK:       trace_alpha
// CHECK:       fused into a single kernel
// CHECK:       fused op is above
// CHECK:       Sequence
// CHECK:       """)
// CHECK:       mogg_async_error({{.*}},{{.*}},{{.*}})

mgp.model @fused_source_notes : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %cond = index.bool.constant true
  // Chain the second assert through the first to guarantee sequential
  // dependency and sequence fusion.
  %ch1 = mgp.assert %cond, "alpha failed" loc(fused<"  File \22model.py\22, line 10\n    trace_alpha\n">["alpha":0:0])
  %buf = mgp.buffer.constant #M.dense_array<1> : !M.array<1xui8>
  %flag = mgp.buffer.to_bool %ch1, %buf : !mgp.buffer
  %ch2 = mgp.assert %flag, "beta failed" loc(fused<"  File \22model.py\22, line 20\n    trace_beta\n">["beta":0:0])
  mef.output %ch2 : !mef.chain
}
