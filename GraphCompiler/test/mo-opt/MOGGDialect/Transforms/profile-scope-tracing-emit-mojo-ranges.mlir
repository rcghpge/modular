// RUN: mo-opt %s --mogg-import-packages -mo-to-mogg --mogg-pipeline -mogg-to-mgp -mogg-outline-kernels -mogg-add-try-except-blocks="enable-profile-scope-tracing=true" --mogg-emit-mojo -mgp-dump-mojo 2>&1 | FileCheck %s

// Full-pipeline regression for the in-region range bracketing mechanism.
// After outlining to KernelStubOp and enabling profile-scope-tracing, the
// emitted Mojo must contain a Trace call whose name contains the kernel stub
// symbol ("region_..."), confirming mechanism 2 is emitted, not just the
// per-kernel suffix from WithTraceOp (mechanism 1).

// CHECK: var [[RANGE:.*]] = max.runtime.tracing.Trace[max.runtime.tracing.TraceLevel.OP, target = StaticString("cpu")]("[draft_forward] region_0")
// CHECK-NEXT: [[RANGE]].start()
// CHECK: [[RANGE]].end()

mo.graph @simple_fusion(
    %x: !mo.tensor<[10], f32>,
    %y: !mo.tensor<[10], f32>
) -> !mo.tensor<[10], f32> {
    %relu = mo.relu(%x): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"draft_forward", loc(unknown)>)
    %add = mo.add(%relu, %y): !mo.tensor<[10], f32> loc(#mogg.profile_scope<"draft_forward", loc(unknown)>)
    mo.output %add: !mo.tensor<[10], f32>
}
