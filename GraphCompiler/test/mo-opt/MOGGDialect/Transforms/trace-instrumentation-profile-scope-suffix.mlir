// RUN: mo-opt %s --mogg-import-packages --mo-to-mogg --mogg-pipeline --mogg-to-mgp --mogg-outline-kernels --mogg-add-try-except-blocks --mogg-emit-mojo -mgp-dump-mojo 2>&1 | FileCheck %s

// Test that Graph profile scope's ProfileScopeLocationAttr survives
// fusion/outlining and that WithTraceOp::emitMojo appends the formatted
// " [scope]" suffix to the kernel's existing NVTX trace name. The scope goes
// after the kernel's own name here, unlike the `[scope] region_N` ordering
// used for profile_scope range names (the in-region range bracketing mechanism,
// AddTryExceptBlocks.cpp). Both ops carry the same profile_scope("my_scope")
// attribute, simulating what Graph.profile_scope("my_scope") attaches to
// every op created in its `with` block.
// CHECK: with max.runtime.tracing.Trace[max.runtime.tracing.TraceLevel.OP, target = StaticString("cpu")]("Elementwise(relu, add) [my_scope]"{{.*}}):

mo.graph @simple_fusion(
    %x: !mo.tensor<[10], f32>,
    %y: !mo.tensor<[10], f32>
) -> !mo.tensor<[10], f32> {
    %relu = mo.relu(%x): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"my_scope", loc(unknown)>)
    %add = mo.add(%relu, %y): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"my_scope", loc(unknown)>)
    mo.output %add: !mo.tensor<[10], f32>
}
