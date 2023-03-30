// RUN: mo-opt --mo-to-mogg=use-jit=true --fuse-elementwise --canonicalize --cse %s | FileCheck %s


// Our current greedy upwards algorithm assumes that elementwise is going to be fused into
// so we don't currently allow multiple outputs from fused kernels. This means each end node
// is a full fused graph, so we should create multiple fused kernels rather than avoiding the work
// of the elementwise. This will likely change over time.


mo.graph @simpleTwoKernels(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.tanh(%0) : !mo.tensor<[4, 3], f32>
  %2 = mo.add(%1, %1) : !mo.tensor<[4, 3], f32>
  %3 = mo.add(%1, %arg0) : !mo.tensor<[4, 3], f32>
  mo.output %2, %3 : !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>
}


// CHECK: mo.graph @simpleTwoKernels(%[[ARG0:.*]]: !mo.tensor<[4, 3], f32>)
// CHECK-NEXT: %[[KERNEL_1:.*]] = mogg.kernel (%[[ARG0]])
// CHECK: mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN:.*]] = mogg.invoke @[[IN_LAMBDA:.*]](%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT:.*]] = mogg.call "mo.sqrt"(%[[IN]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[TANH:.*]] = mogg.call "mo.tanh"(%[[SQRT]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[ADD:.*]] = mogg.call "mo.add"(%[[TANH]], %[[TANH]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.invoke @[[OUT_LAMBDA:.*]](%[[INDICES]], %[[ADD]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output
// CHECK: %[[KERNEL_2:.*]] = mogg.kernel (%[[ARG0]])
// CHECK: mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN:.*]] = mogg.invoke @[[IN_LAMBDA:.*]](%[[INDICES]]) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
// CHECK-NEXT:    %[[SQRT:.*]] = mogg.call "mo.sqrt"(%[[IN]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[TANH:.*]] = mogg.call "mo.tanh"(%[[SQRT]]) : !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    %[[ADD:.*]] = mogg.call "mo.add"(%[[TANH]], %[[IN]]) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
// CHECK-NEXT:    mogg.invoke @[[OUT_LAMBDA:.*]](%[[INDICES]], %[[ADD]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output
// CHECK: mo.output %[[KERNEL_1]], %[[KERNEL_2]] : !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>



mo.graph @manyFanOut(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.rsqrt(%0) : !mo.tensor<[4, 3], f32>
  %2 = mo.tanh(%0) : !mo.tensor<[4, 3], f32>
  %3 = mo.sub(%1, %2) : !mo.tensor<[4, 3], f32>
  %4 = mo.sqrt(%3) : !mo.tensor<[4, 3], f32>
  %5 = mo.rsqrt(%3) : !mo.tensor<[4, 3], f32>
  %6 = mo.add(%4, %5) : !mo.tensor<[4, 3], f32>
  %7 = mo.sqrt(%6) : !mo.tensor<[4, 3], f32>
  %8 = mo.rsqrt(%6) : !mo.tensor<[4, 3], f32>
  %9 = mo.mul(%7, %8) : !mo.tensor<[4, 3], f32>
  %10 = mo.sqrt(%9) : !mo.tensor<[4, 3], f32>
  %11 = mo.rsqrt(%9) : !mo.tensor<[4, 3], f32>
  %12 = mo.mul(%10, %11) : !mo.tensor<[4, 3], f32>
  mo.output %3, %6, %9, %12 : !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>, !mo.tensor<[4, 3], f32>
}

// Each chain of elementwise ops will fuse as greedily as possible.

// CHECK: mo.graph @manyFanOut
// CHECK-NEXT: %[[KERNEL1:.*]] = mogg.kernel
// CHECK: mogg.lambda @elemwise_compute_lambda = (%[[INDCIES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN:.*]] = mogg.invoke @[[IN_LAMBDA:.*]](%[[INDCIES]])
// CHECK-NEXT:    = mogg.call "mo.sqrt"
// CHECK-NEXT:    = mogg.call "mo.rsqrt"
// CHECK-NEXT:    = mogg.call "mo.tanh"
// CHECK-NEXT:    %[[FINAL:.*]] = mogg.call "mo.sub"
// CHECK-NEXT:    mogg.invoke @[[OUT_LAMBDA:.*]](%[[INDICES]], %[[FINAL]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output

// CHECK: %[[KERNEL2:.*]] = mogg.kernel
// CHECK:mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN:.*]] = mogg.invoke @[[IN_LAMBDA:.*]](%[[INDCIES]])
// CHECK-NEXT:    = mogg.call "mo.sqrt"
// CHECK-NEXT:    = mogg.call "mo.rsqrt"
// CHECK-NEXT:    = mogg.call "mo.tanh"
// CHECK-NEXT:    = mogg.call "mo.sub"
// CHECK-NEXT:    = mogg.call "mo.sqrt"
// CHECK-NEXT:    = mogg.call "mo.rsqrt"
// CHECK-NEXT:    %[[FINAL:.*]] = mogg.call "mo.add"
// CHECK-NEXT:    mogg.invoke @[[OUT_LAMBDA:.*]](%[[INDICES]], %[[FINAL]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output

// CHECK: %[[KERNEL3:.*]] = mogg.kernel
// CHECK:mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN:.*]] = mogg.invoke @[[IN_LAMBDA:.*]](%[[INDCIES]])
// CHECK-NEXT:    = mogg.call "mo.sqrt"
// CHECK-NEXT:    = mogg.call "mo.rsqrt"
// CHECK-NEXT:    = mogg.call "mo.tanh"
// CHECK-NEXT:    = mogg.call "mo.sub"
// CHECK-NEXT:    = mogg.call "mo.sqrt"
// CHECK-NEXT:    = mogg.call "mo.rsqrt"
// CHECK-NEXT:    = mogg.call "mo.add"
// CHECK-NEXT:    = mogg.call "mo.sqrt"
// CHECK-NEXT:    = mogg.call "mo.rsqrt"
// CHECK-NEXT:    %[[FINAL:.*]] = mogg.call "mo.mul"
// CHECK-NEXT:    mogg.invoke @[[OUT_LAMBDA:.*]](%[[INDICES]], %[[FINAL]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output

// CHECK: %[[KERNEL4:.*]] = mogg.kernel
// CHECK:mogg.lambda @elemwise_compute_lambda = (%[[INDICES:.*]]: !mogg.indices){
// CHECK-NEXT:    %[[IN:.*]] = mogg.invoke @[[IN_LAMBDA:.*]](%[[INDCIES]])
// CHECK-NEXT:    mogg.call "mo.sqrt"
// CHECK-NEXT:    mogg.call "mo.rsqrt"
// CHECK-NEXT:    mogg.call "mo.tanh"
// CHECK-NEXT:    mogg.call "mo.sub"
// CHECK-NEXT:    mogg.call "mo.sqrt"
// CHECK-NEXT:    mogg.call "mo.rsqrt"
// CHECK-NEXT:    mogg.call "mo.add"
// CHECK-NEXT:    mogg.call "mo.sqrt"
// CHECK-NEXT:    mogg.call "mo.rsqrt"
// CHECK-NEXT:    mogg.call "mo.mul"
// CHECK-NEXT:    mogg.call "mo.sqrt"
// CHECK-NEXT:    mogg.call "mo.rsqrt"
// CHECK-NEXT:    %[[FINAL:.*]] = mogg.call "mo.mul"
// CHECK-NEXT:    mogg.invoke @[[OUT_LAMBDA:.*]](%[[INDICES]], %[[FINAL]]) : !mogg.indices, !mogg.simd<f32>
// CHECK-NEXT:    mogg.output

// CHECK: mo.output %[[KERNEL1]], %[[KERNEL2]], %[[KERNEL3]], %[[KERNEL4]]
