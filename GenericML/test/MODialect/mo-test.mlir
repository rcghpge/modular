// RUN: mo-opt %s | mo-opt | FileCheck %s

// CHECK: mo.graph @constant() -> !mo.tensor<2x3xf32>
mo.graph @constant() -> !mo.tensor<2x3xf32> {
  // CHECK: %[[VAL_0:.*]] = "mo.constant"() {value = dense<1.0{{.*}}> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
  %0 = "mo.constant"() {value = dense<1.000000e+00> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
  // CHECK: mo.output %[[VAL_0]] : !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// CHECK: mo.graph @two_constants() -> (!mo.tensor<2x3xf32>, !mo.tensor<3x2xf32>)
mo.graph @two_constants() -> (!mo.tensor<2x3xf32>, !mo.tensor<3x2xf32>) {
  // CHECK-DAG: %[[VAL_0:.*]] = "mo.constant"() {value = dense<1.0{{.*}}> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
  // CHECK-DAG: %[[VAL_1:.*]] = "mo.constant"() {value = dense<2.0{{.*}}> : tensor<3x2xf32>} : () -> !mo.tensor<3x2xf32>
  %0 = "mo.constant"() {value = dense<1.000000e+00> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
  %1 = "mo.constant"() {value = dense<2.000000e+00> : tensor<3x2xf32>} : () -> !mo.tensor<3x2xf32>
  // CHECK: mo.output %[[VAL_0]], %[[VAL_1]] : !mo.tensor<2x3xf32>, !mo.tensor<3x2xf32>
  mo.output %0, %1 : !mo.tensor<2x3xf32>, !mo.tensor<3x2xf32>
}
