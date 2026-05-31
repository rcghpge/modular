// RUN: mo-opt %s --mogg-import-packages -mo-to-mogg --mogg-simplify-kernel-operands -mogg-fuse-elementwise --canonicalize --cse -mogg-fuse-views --canonicalize --cse -mogg-fuse-epilogues -mogg-to-mgp -mogg-outline-kernels -mogg-add-try-except-blocks --mogg-emit-mojo -mgp-dump-mojo 2>&1 | FileCheck %s

// Test that emitting Mojo wraps the body of raising kernels in a Trace context
// manager using the fusion info string as the trace name.
//
// Fused relu+add produces fusion info Elementwise(activation, add) (dialect
// prefixes are stripped from trace descriptions for readability).
// CHECK: with std.runtime.tracing.Trace[std.runtime.tracing.TraceLevel.OP, target = StaticString("cpu")]("Elementwise(activation, add)"{{.*}}):

mo.graph @simple_fusion(
    %x: !mo.tensor<[10], f32>,
    %y: !mo.tensor<[10], f32>
) -> !mo.tensor<[10], f32> {
    %relu = mo.activation(%x) {activation = "relu"}: !mo.tensor<[10], f32>
    %add = mo.add(%relu, %y): !mo.tensor<[10], f32>
    mo.output %add: !mo.tensor<[10], f32>
}
