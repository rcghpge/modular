// RUN: mo-opt %s --mogg-import-packages -mo-to-mogg --mogg-simplify-kernel-operands -mogg-fuse-elementwise --canonicalize --cse -mogg-fuse-views --canonicalize --cse -mogg-fuse-epilogues -mogg-to-mgp -mogg-outline-kernels -mogg-add-try-except-blocks -mogg-emit-mojo -mgp-dump-mojo 2>&1 | FileCheck %s

// Test that toPrettyTree() fusion info tree comments appear in emitted Mojo.
// The tree shows op names with (file:line) locations and FunctionType signatures.
// CHECK:      # FusionInfo:
// CHECK-NEXT: #   Epilogue (operandIndices=[0])
// CHECK-NEXT: #       ├── mo.matmul ({{[^)]+}}) : (!mo.tensor<[20, 30], f32>, !mo.tensor<[30, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #       └── View
// CHECK-NEXT: #           ├── mo.static.broadcast_to ({{[^)]+}}) : (!mo.tensor<[1, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #           ├── mo.static.broadcast_to ({{[^)]+}}) : (!mo.tensor<[1, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #           ├── mo.static.broadcast_to ({{[^)]+}}) : (!mo.tensor<[1, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #           └── Elementwise
// CHECK-NEXT: #               ├── mo.activation ({{[^)]+}}) : (!mo.tensor<[20, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #               ├── mo.add ({{[^)]+}}) : (!mo.tensor<[20, 40], f32>, !mo.tensor<[20, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #               ├── mo.tanh ({{[^)]+}}) : (!mo.tensor<[20, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #               ├── mo.add ({{[^)]+}}) : (!mo.tensor<[20, 40], f32>, !mo.tensor<[20, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #               ├── mo.sqrt ({{[^)]+}}) : (!mo.tensor<[20, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #               ├── mo.mul ({{[^)]+}}) : (!mo.tensor<[20, 40], f32>, !mo.tensor<[20, 40], f32>) -> !mo.tensor<[20, 40], f32>
// CHECK-NEXT: #               ├── mo.cast ({{[^)]+}}) : (!mo.tensor<[20, 40], f32>) -> !mo.tensor<[20, 40], f16>
// CHECK-NEXT: #               └── mo.cast ({{[^)]+}}) : (!mo.tensor<[20, 40], f16>) -> !mo.tensor<[20, 40], f32>

mo.graph @big_fusion(
    %A: !mo.tensor<[20, 30], f32>,
    %B: !mo.tensor<[30, 40], f32>,
    %bias: !mo.tensor<[1, 40], f32>,
    %scale: !mo.tensor<[1, 40], f32>,
    %offset: !mo.tensor<[1, 40], f32>
) -> !mo.tensor<[20, 40], f32> {
    %mm    = mo.matmul(%A, %B): (!mo.tensor<[20, 30], f32>, !mo.tensor<[30, 40], f32>) -> !mo.tensor<[20, 40], f32>
    %bcast_bias = mo.static.broadcast_to(%bias) : !mo.tensor<[1, 40], f32> -> !mo.tensor<[20, 40], f32>
    %bcast_scale = mo.static.broadcast_to(%scale) : !mo.tensor<[1, 40], f32> -> !mo.tensor<[20, 40], f32>
    %bcast_offset = mo.static.broadcast_to(%offset) : !mo.tensor<[1, 40], f32> -> !mo.tensor<[20, 40], f32>
    %relu  = mo.activation(%mm) {activation = "relu"}: !mo.tensor<[20, 40], f32>
    %add   = mo.add(%relu, %bcast_bias): !mo.tensor<[20, 40], f32>
    %tanh  = mo.tanh(%add): !mo.tensor<[20, 40], f32>
    %add2  = mo.add(%tanh, %bcast_offset): !mo.tensor<[20, 40], f32>
    %sqrt  = mo.sqrt(%add2): !mo.tensor<[20, 40], f32>
    %mul   = mo.mul(%sqrt, %bcast_scale): !mo.tensor<[20, 40], f32>
    %cast  = mo.cast(%mul) : (!mo.tensor<[20, 40], f32>) -> !mo.tensor<[20, 40], f16>
    %cast2 = mo.cast(%cast) : (!mo.tensor<[20, 40], f16>) -> !mo.tensor<[20, 40], f32>
    mo.output %cast2: !mo.tensor<[20, 40], f32>
}
