// RUN: mo-opt --mogg-import-packages --mo-to-mogg --mogg-fuse-elementwise --mogg-simplify-kernel-operands --canonicalize --cse --mogg-strip-kernel-decls %s | FileCheck %s

mo.graph @simpleDoubleUse(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.add(%0, %0) : !mo.tensor<[4, 3], f32>
  mo.output %1 : !mo.tensor<[4, 3], f32>
}

// CHECK-LABEL: mo.graph @simpleDoubleUse
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NOT: mogg.kernel{{[^.]}}

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
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sub"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NOT: mogg.kernel{{[^.]}}

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
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sub"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NOT: mogg.kernel{{[^.]}}

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
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Tanh"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Rsqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sub"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Rsqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Rsqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Mul"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Rsqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Mul"]
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @inversePyramid(%arg0: !mo.tensor<[4, 3], f32>, %arg1: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[4, 3], f32>) {
  // Top level ops.
  %0 = mo.sqrt(%arg0) : !mo.tensor<[4, 3], f32>
  %1 = mo.activation(%arg0) {activation = "relu"} : !mo.tensor<[4, 3], f32>
  %2 = mo.sqrt(%arg1) : !mo.tensor<[4, 3], f32>
  %3 = mo.activation(%arg1) {activation = "relu"} : !mo.tensor<[4, 3], f32>
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
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Tanh"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Div"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sub"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Mul"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Activation"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Activation"]
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Sqrt"]
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT:  mogg.call.elementwise["{{.*}}.Mul"]
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @controlFlow<N, M>(%cond : !mo.tensor<[], bool>, %arg0: !mo.tensor<[N, M], f32>) -> (!mo.tensor<[N, M], f32>) {
  %relu = mo.activation(%arg0) {activation = "relu"} : !mo.tensor<[N, M], f32>

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
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Activation"]
// CHECK: mo.if
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Tanh"]
// CHECK: else
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Mul"]


mo.graph @noDuplicationOfCompute() -> (!mo.tensor<[5], f32>,!mo.tensor<[5], f32>) {
  %0 = mo.constant {value = #M.dense_array<-1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %1 = mo.constant {value = #M.dense_array<3.000000e+00, 1.000000e+00, -6.000000e+00, 2.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32>
  %3 = mo.tanh(%2) : !mo.tensor<[5], f32>
  %4 = mo.activation(%2) {activation = "relu"} : !mo.tensor<[5], f32>
  mo.output %3, %4 : !mo.tensor<[5], f32>, !mo.tensor<[5], f32>
}

// CHECK-LABEL: mo.graph @noDuplicationOfCompute
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Add"]
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Tanh"]
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Activation"]
// CHECK-NOT: mogg.kernel{{[^.]}}

mo.graph @withDuplicationOfCompute() -> (!mo.tensor<[5], f32, gpu:0>,!mo.tensor<[5], f32, gpu:0>) {
  %0 = mo.constant {value = #M.dense_array<-1.000000e+00, 2.000000e+00, -3.000000e+00, -4.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32, gpu:0>
  %1 = mo.constant {value = #M.dense_array<3.000000e+00, 1.000000e+00, -6.000000e+00, 2.000000e+00, 5.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32, gpu:0>
  %2 = mo.add(%0, %1) : !mo.tensor<[5], f32, gpu:0>
  %3 = mo.tanh(%2) : !mo.tensor<[5], f32, gpu:0>
  %4 = mo.activation(%2) {activation = "relu"} : !mo.tensor<[5], f32, gpu:0>
  mo.output %3, %4 : !mo.tensor<[5], f32, gpu:0>, !mo.tensor<[5], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @withDuplicationOfCompute
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Tanh"]
// CHECK: mogg.kernel
// CHECK: mogg.call.elementwise["{{.*}}.Add"]
// CHECK-NEXT: mogg.call.elementwise["{{.*}}.Activation"]
// CHECK-NOT: mogg.kernel{{[^.]}}
