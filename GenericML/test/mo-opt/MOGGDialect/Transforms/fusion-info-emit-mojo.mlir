// RUN: mo-opt %s -mo-to-mogg --mogg-simplify-kernel-operands -mogg-fuse-elementwise --canonicalize --cse -mogg-fuse-views --canonicalize --cse -mogg-fuse-prologues-epilogues -mgp-mo-to-primitives -mogg-outline-kernels -mogg-add-try-except-blocks -mogg-emit-mojo -mgp-dump-mojo 2>&1 | FileCheck %s

// Test that toPrettyTree() fusion info tree comments appear in emitted Mojo.
// CHECK:      # FusionInfo:
// CHECK-NEXT: #   Prologue (operandIndices=[0])
// CHECK-NEXT: #       ├── mo.matmul:{{[0-9]+}}
// CHECK-NEXT: #       └── View
// CHECK-NEXT: #           ├── mo.static.broadcast_to:{{[0-9]+}}
// CHECK-NEXT: #           ├── mo.static.broadcast_to:{{[0-9]+}}
// CHECK-NEXT: #           ├── mo.static.broadcast_to:{{[0-9]+}}
// CHECK-NEXT: #           └── Elementwise
// CHECK-NEXT: #               ├── mo.relu:{{[0-9]+}}
// CHECK-NEXT: #               ├── mo.add:{{[0-9]+}}
// CHECK-NEXT: #               ├── mo.tanh:{{[0-9]+}}
// CHECK-NEXT: #               ├── mo.add:{{[0-9]+}}
// CHECK-NEXT: #               ├── mo.sqrt:{{[0-9]+}}
// CHECK-NEXT: #               ├── mo.mul:{{[0-9]+}}
// CHECK-NEXT: #               ├── mo.cast:{{[0-9]+}}
// CHECK-NEXT: #               └── mo.cast:{{[0-9]+}}

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
    %relu  = mo.relu(%mm): !mo.tensor<[20, 40], f32>
    %add   = mo.add(%relu, %bcast_bias): !mo.tensor<[20, 40], f32>
    %tanh  = mo.tanh(%add): !mo.tensor<[20, 40], f32>
    %add2  = mo.add(%tanh, %bcast_offset): !mo.tensor<[20, 40], f32>
    %sqrt  = mo.sqrt(%add2): !mo.tensor<[20, 40], f32>
    %mul   = mo.mul(%sqrt, %bcast_scale): !mo.tensor<[20, 40], f32>
    %cast  = mo.cast(%mul) : (!mo.tensor<[20, 40], f32>) -> !mo.tensor<[20, 40], f16>
    %cast2 = mo.cast(%cast) : (!mo.tensor<[20, 40], f16>) -> !mo.tensor<[20, 40], f32>
    mo.output %cast2: !mo.tensor<[20, 40], f32>
}
