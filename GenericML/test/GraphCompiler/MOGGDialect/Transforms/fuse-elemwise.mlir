// RUN: mo-opt --mo-to-mogg --fuse-elementwise --simplify-kernel-operands --canonicalize --cse %s | FileCheck %s

mo.graph @simpleDoubleUse(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.add(%0, %0) : !mo.tensor<[4, 3], f32>
  mo.output %1 : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @simpleDoubleUse
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.add"]
// CHECK-NOT: mogg.experimental.kernel

mo.graph @rightmostTreeFusion(%arg0: !mo.tensor<[4, 3], f32>, %arg1: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.sqrt(%0) : !mo.tensor<[4, 3], f32>
  %2 = mo.sqrt(%1) : !mo.tensor<[4, 3], f32>
  %3 = mo.sqrt(%2) : !mo.tensor<[4, 3], f32>
  %4 = mo.sub(%arg1, %3) : !mo.tensor<[4, 3], f32>
  %5 = mo.add(%arg1, %4) : !mo.tensor<[4, 3], f32>
  mo.output %5 : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @rightmostTreeFusion
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sub"]
// CHECK-NEXT: mogg.call.elementwise["mo.add"]
// CHECK-NOT: mogg.experimental.kernel

mo.graph @leftmostTreeFusion(%arg0: !mo.tensor<[4, 3], f32>, %arg1: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.sqrt(%0) : !mo.tensor<[4, 3], f32>
  %2 = mo.sqrt(%1) : !mo.tensor<[4, 3], f32>
  %3 = mo.sqrt(%2) : !mo.tensor<[4, 3], f32>
  %4 = mo.sub(%3, %arg1) : !mo.tensor<[4, 3], f32>
  %5 = mo.add(%4, %arg1) : !mo.tensor<[4, 3], f32>
  mo.output %5 : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @leftmostTreeFusion
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sub"]
// CHECK-NEXT: mogg.call.elementwise["mo.add"]
// CHECK-NOT: mogg.experimental.kernel

mo.graph @multipleDiamondPattern(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
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
  mo.output %12 : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @multipleDiamondPattern
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.tanh"]
// CHECK-NEXT: mogg.call.elementwise["mo.rsqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sub"]
// CHECK-NEXT: mogg.call.elementwise["mo.rsqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.add"]
// CHECK-NEXT: mogg.call.elementwise["mo.rsqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.mul"]
// CHECK-NEXT: mogg.call.elementwise["mo.rsqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.mul"]
// CHECK-NOT: mogg.experimental.kernel

mo.graph @inversePyramid(%arg0: !mo.tensor<[4, 3], f32>, %arg1: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  // Top level ops.
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.relu(%arg0) : !mo.tensor<[4, 3], f32>
  %2 = mo.sqrt(%arg1) : !mo.tensor<[4, 3], f32>
  %3 = mo.relu(%arg1) : !mo.tensor<[4, 3], f32>
  %4 = mo.add(%arg0, %arg1) : !mo.tensor<[4, 3], f32>
  %5 = mo.mul(%arg0, %arg1) : !mo.tensor<[4, 3], f32>
  %6 = mo.sub(%arg0, %arg1) : !mo.tensor<[4, 3], f32>
  %7 = mo.div(%arg0, %arg1) : !mo.tensor<[4, 3], f32>
  %8 = mo.tanh(%arg0) : !mo.tensor<[4, 3], f32>

  %9 = mo.add(%0, %1) : !mo.tensor<[4, 3], f32>
  %10 = mo.add(%2, %3) : !mo.tensor<[4, 3], f32>
  %11 = mo.add(%5, %6) : !mo.tensor<[4, 3], f32>
  %12 = mo.add(%7, %8) : !mo.tensor<[4, 3], f32>

  %13 = mo.add(%9, %10) : !mo.tensor<[4, 3], f32>
  %14 = mo.add(%11, %12) : !mo.tensor<[4, 3], f32>

  %out = mo.mul(%13, %14) : !mo.tensor<[4, 3], f32>
  mo.output %out : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @inversePyramid
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.tanh"]
// CHECK-NEXT: mogg.call.elementwise["mo.div"]
// CHECK-NEXT: mogg.call.elementwise["mo.sub"]
// CHECK-NEXT: mogg.call.elementwise["mo.mul"]
// CHECK-NEXT: mogg.call.elementwise["mo.relu"]
// CHECK-NEXT: mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT: mogg.call.elementwise["mo.relu"]
// CHECK-NEXT:  mogg.call.elementwise["mo.sqrt"]
// CHECK-NEXT:  mogg.call.elementwise["mo.add"]
// CHECK-NEXT:  mogg.call.elementwise["mo.add"]
// CHECK-NEXT:  mogg.call.elementwise["mo.add"]
// CHECK-NEXT:  mogg.call.elementwise["mo.add"]
// CHECK-NEXT:  mogg.call.elementwise["mo.add"]
// CHECK-NEXT:  mogg.call.elementwise["mo.add"]
// CHECK-NEXT:  mogg.call.elementwise["mo.mul"]
// CHECK-NOT: mogg.experimental.kernel

mo.graph @controlFlow<N, M>(%cond : !mo.tensor<[], bool>, %arg0: !mo.tensor<[N, M], f32>) -> (!mo.tensor<[N, M], f32>) {
  %relu = mo.relu(%arg0) : !mo.tensor<[N, M], f32>

  %0 = mo.if(%cond : !mo.tensor<[], bool>) -> !mo.tensor<[N, M], f32> {
    %1 = mo.add(%relu, %relu) : !mo.tensor<[N, M], f32>
    %2 = mo.tanh(%1) : !mo.tensor<[N, M], f32>
    mo.yield %2 : !mo.tensor<[N, M], f32>
  } else {
    %2 = mo.mul(%relu, %relu) : !mo.tensor<[N, M], f32>
    mo.yield %2 : !mo.tensor<[N, M], f32>
  }
  mo.output %0 : !mo.tensor<[N, M], f32>
}

// Check all kernels are compiled but unfused.
// CHECK-LABEL: mo.graph @controlFlow
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.relu"]
// CHECK: mo.if
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.add"]
// CHECK-NEXT: mogg.call.elementwise["mo.tanh"]
// CHECK: else
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.mul"]


mo.graph @noDuplicationOfCompute() -> (!mo.tensor<[5], f32>,!mo.tensor<[5], f32>) {
  %0 = mo.constant {value = #M.dense_array<-1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %1 = mo.constant {value = #M.dense_array<3.000000e+00, 1.000000e+00, -6.000000e+00, 2.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %3 = mo.tanh(%2) : !mo.tensor<[5], f32>
  %4 = mo.relu(%2) : !mo.tensor<[5], f32>
  mo.output %3, %4 : !mo.tensor<[5], f32>, !mo.tensor<[5], f32>
}

// CHECK-LABEL: mo.graph @noDuplicationOfCompute
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.add"]
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.tanh"]
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.relu"]
// CHECK-NOT: mogg.experimental.kernel

mo.graph @withDuplicationOfCompute() -> (!mo.tensor<[5], f32, gpu:0>,!mo.tensor<[5], f32, gpu:0>) {
  %0 = mo.constant {value = #M.dense_array<-1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32, gpu:0>
  %1 = mo.constant {value = #M.dense_array<3.000000e+00, 1.000000e+00, -6.000000e+00, 2.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32, gpu:0>
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32, gpu:0>
  %3 = mo.tanh(%2) : !mo.tensor<[5], f32, gpu:0>
  %4 = mo.relu(%2) : !mo.tensor<[5], f32, gpu:0>
  mo.output %3, %4 : !mo.tensor<[5], f32, gpu:0>, !mo.tensor<[5], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @withDuplicationOfCompute
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.add"]
// CHECK-NEXT: mogg.call.elementwise["mo.tanh"]
// CHECK: mogg.experimental.kernel
// CHECK: mogg.call.elementwise["mo.add"]
// CHECK-NEXT: mogg.call.elementwise["mo.relu"]
// CHECK-NOT: mogg.experimental.kernel
