// RUN: mo-opt %s --mo-to-mogg --mogg-simplify-kernel-operands --mogg-fuse-elementwise --canonicalize --cse -mogg-to-mgp --mogg-outline-kernels --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mgp-dump-mojo 2>&1 | FileCheck %s

// Test that when mo ops with FusedLoc source note metadata are fused via
// elementwise fusion, the source notes from ALL fused ops are preserved in
// the emitted mogg_async_error call.
//
// Before the fix, FuseElementwiseKernels.cpp only called setFusionInfoAttr()
// without merging locations. This caused source notes from the fused-in op
// to be lost. Now FusedLoc::get() is used to merge locations from both src
// and dst, so AddStackTracesToAssertsPass / AsyncErrorOp::emitMojo can find
// all source notes.

// The emitted error notes should contain source traces from both fused ops.
// CHECK:       String("""
// CHECK:       trace_add
// CHECK:       trace_relu
// CHECK:       """)
// CHECK:       mogg_async_error({{.*}},{{.*}},{{.*}})

mo.graph @elemwise_source_notes(
    %arg0: !mo.tensor<[4, 3], f32>
) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.relu(%arg0) : !mo.tensor<[4, 3], f32> loc(fused<"  File \22model.py\22, line 10\n    trace_relu\n">["model.py":10:0])
  %1 = mo.add(%0, %arg0) : !mo.tensor<[4, 3], f32> loc(fused<"  File \22model.py\22, line 20\n    trace_add\n">["model.py":20:0])
  mo.output %1 : !mo.tensor<[4, 3], f32>
}
