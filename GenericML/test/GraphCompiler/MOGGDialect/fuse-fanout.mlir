// RUN: mo-opt --mo-to-mogg --fuse-elementwise --canonicalize --cse %s | FileCheck %s

// If we need to materialize a kernel then that should prevent it from being
// folded into subsequent fusions to prevent duplicate compute.

mo.graph @simpleTwoKernels(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.tanh(%0) : !mo.tensor<[4, 3], f32>
  %2 = mo.add(%1, %1) : !mo.tensor<[4, 3], f32>
  %3 = mo.sub(%2, %arg0) : !mo.tensor<[4, 3], f32>
  mo.output %2, %3 : !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>
}


// CHECK: mo.graph @simpleTwoKernels(%[[ARG0:.*]]: !mo.tensor<[4, 3], f32>)
// CHECK-NEXT: %[[KERNEL_1:.*]] = mogg.kernel (%[[ARG0]])
// CHECK: mogg.lambda @[[ELEMWISE_COMPUTE_LAMBDA:.*]] = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN:.*]] = mogg.invoke @[[IN_LAMBDA:.*]](%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT:.*]] = mogg.call "mo.sqrt"(%[[IN]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[TANH:.*]] = mogg.call "mo.tanh"(%[[SQRT]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[ADD:.*]] = mogg.call "mo.add"(%[[TANH]], %[[TANH]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.invoke @[[OUT_LAMBDA:.*]](%[[INDICES]], %[[ADD]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output


// CHECK: %[[KERNEL_2:.*]] = mogg.kernel (%[[KERNEL_1]], %[[ARG0]])
// CHECK: mogg.lambda @[[ELEMWISE_COMPUTE_LAMBDA:.*]]  = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN1:.*]] = mogg.invoke @[[IN_LAMBDA1:.*]](%[[INDICES]])
// CHECK-NEXT:    %[[IN2:.*]] = mogg.invoke @[[IN_LAMBDA2:.*]](%[[INDICES]])
// CHECK-NEXT:    %[[SUB:.*]] = mogg.call "mo.sub"(%[[IN1]], %[[IN2]])
// CHECK-NEXT:    mogg.invoke @[[OUT_LAMBDA:.*]](%[[INDICES]], %[[SUB]])
// CHECK-NEXT:    mogg.output
// CHECK: mo.output %[[KERNEL_1]], %[[KERNEL_2]] : !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>
