// RUN: mo-opt %s --mogg-import-packages --mo-to-mogg --mogg-pipeline --mogg-to-mgp --mogg-outline-kernels --mogg-add-try-except-blocks --mogg-emit-mojo -mgp-dump-mojo 2>&1 | FileCheck %s

// Regression test for two issues that only surface when Graph.profile_scope()
// wraps a location whose leaf is a FusedLoc (the shape produced by
// AttachPythonStackTraces when source tracebacks are enabled):
//   1. The per-kernel scope suffix must not be dropped.
//   2. Python source notes inside the FusedLoc must still be emitted as the
//      Trace detail string.

// CHECK: with max.runtime.tracing.Trace[max.runtime.tracing.TraceLevel.OP, target = StaticString("cpu")]("Elementwise(relu, add) [my_scope]", "\nmodel.py:10:forward\n"{{.*}}):

#traceback = loc(fused<"  File \"model.py\", line 10, in forward\n    x = relu(x)\n">["model.py":10:0])
#scoped = loc(#mogg.profile_scope<"my_scope", #traceback>)

mo.graph @simple_fusion(
    %x: !mo.tensor<[10], f32>,
    %y: !mo.tensor<[10], f32>
) -> !mo.tensor<[10], f32> {
    %relu = mo.relu(%x): !mo.tensor<[10], f32> loc(#scoped)
    %add = mo.add(%relu, %y): !mo.tensor<[10], f32> loc(#scoped)
    mo.output %add: !mo.tensor<[10], f32>
}
