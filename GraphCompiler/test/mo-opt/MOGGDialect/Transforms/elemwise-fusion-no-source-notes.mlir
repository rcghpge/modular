// RUN: mo-opt %s --mogg-import-packages --mo-to-mogg --mogg-simplify-kernel-operands --mogg-fuse-elementwise --canonicalize --cse -mogg-to-mgp --mogg-outline-kernels --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mgp-dump-mojo 2>&1 | FileCheck %s

// Test that when fused ops have NO source note metadata (no FusedLoc with
// Python tracebacks), the emitted mogg_format_kernel_error call contains the
// fusion tree but NOT a "Source Traceback:" section. This ensures that the
// runtime correctly omits the traceback section when no traces are present.
// See GEX-2678.
//
// Contrast with elemwise-fusion-source-notes.mlir, which tests the case where
// source notes ARE present.

// The outer MGP region wraps errors with the region name (mogg_format_region_error).
// The inner MOGG ABI stub embeds the fusion tree in mogg_format_kernel_error
// with an empty traceback argument (no FusedLoc metadata on these ops).
// Output order: outer region stub Mojo first, inner stub Mojo after.
// CHECK:       mogg_format_region_error(
// CHECK:       mogg_async_error({{.*}},{{.*}})
// CHECK:       mogg_format_kernel_error(
// CHECK:       Elementwise
// CHECK-NOT:   Source Traceback

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
