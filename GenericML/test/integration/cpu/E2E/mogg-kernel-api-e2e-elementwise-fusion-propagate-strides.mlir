// RUN: mo-opt %s -o %t.mlir --mo-to-mgp="constant-fold=false  mogg-debug-dump-fused-tensor-specs=true"
// RUN: mt --execute %t.mlir | FileCheck %s

// Unit tests to ensure that during ViewFusion the static strides are propagated correctly.

// CHECK-LABEL: Running 'fused_broadcast_transpose':
// CHECK: input #0: shape = [5, 4, 3] strides = [0, 1, 0]
// CHECK: input #1: shape = [5, 4, 3] strides = [1, 5, 20]
mo.graph @fused_broadcast_transpose(%x: !mo.tensor<[4, 1], f32>, %y: !mo.tensor<[3, 4, 5], f32>) -> (!mo.tensor<[5, 4, 3], f32>) {
    %0 = mo.static.broadcast_to(%x) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[5, 4, 3], f32>
    %perm = mo.constant {value = #M.dense_array<2, 1, 0> : tensor<3xsi32>} : !mo.tensor<[3], si32>
    %1 = mo.transpose(%y, %perm) : (!mo.tensor<[3, 4, 5], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[5, 4, 3], f32>
    %2 = mo.add(%0, %1) : !mo.tensor<[5, 4, 3], f32>
    mo.output %2 : !mo.tensor<[5, 4, 3], f32>
}
