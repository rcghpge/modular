// RUN: mo-opt %s --mo-to-mogg --mogg-simplify-kernel-operands --mogg-fuse-elementwise --canonicalize --cse -mogg-to-mgp --mogg-outline-kernels --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mgp-dump-mojo 2>&1 | FileCheck %s

// Test that when fused ops have NO source note metadata (no FusedLoc with
// Python tracebacks), the emitted mogg_async_error notes contain the fusion
// tree but NOT a "Source Traceback:" header. This ensures that the runtime
// correctly detects the missing traceback and shows the MODULAR_MAX_DEBUG hint.
// See GEX-2678.
//
// Contrast with elemwise-fusion-source-notes.mlir, which tests the case where
// source notes ARE present and the header IS emitted.

// The notes string should contain the fusion tree but not the traceback header.
// CHECK:       String("""
// CHECK-NOT:   Source Traceback
// CHECK:       Fused ops:
// CHECK:       Elementwise
// CHECK:       """)
// CHECK:       mogg_async_error({{.*}},{{.*}},{{.*}})

mo.graph @elemwise_no_source_notes(
    %arg0: !mo.tensor<[4, 3], f32>
) -> (!mo.tensor<[4, 3], f32>) {
  // These ops will be fused by --mogg-fuse-elementwise, creating FusionInfoAttr,
  // but they have no fused<"..."> location metadata so collectSourceNotes()
  // returns an empty list.
  %0 = mo.relu(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.add(%0, %arg0) : !mo.tensor<[4, 3], f32>
  mo.output %1 : !mo.tensor<[4, 3], f32>
}
