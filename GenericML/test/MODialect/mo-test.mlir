// RUN: mo-opt %s | mo-opt | FileCheck %s

// CHECK: "mo.graph"()
"mo.graph" () ({
  ^bb0:
    // CHECK: %[[VAL_0:.*]] = "mo.constant"() {value = dense<1.0{{.*}}> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
    %0 = "mo.constant" () {value = dense<1.0> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
    // CHECK mo.output !mo.tensor<2x3xf32>
    mo.output %0 : !mo.tensor<2x3xf32>
}) {function_type = () -> !mo.tensor<2x3xf32>, sym_name = "constant"} : () -> ()

// CHECK: "mo.graph"()
"mo.graph" () ({
  ^bb0:
    // CHECK: %[[VAL_0:.*]] = "mo.constant"() {value = dense<1.0{{.*}}> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
    %0 = "mo.constant" () {value = dense<1.0> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
    // CHECK: %[[VAL_1:.*]] = "mo.constant"() {value = dense<2.0{{.*}}> : tensor<3x2xf32>} : () -> !mo.tensor<3x2xf32>
    %1 = "mo.constant" () {value = dense<2.0> : tensor<3x2xf32>} : () -> !mo.tensor<3x2xf32>
    // CHECK mo.output !mo.tensor<2x3xf32> !mo.tensor<3x2xf32>
    mo.output %0, %1 : !mo.tensor<2x3xf32>, !mo.tensor<3x2xf32>
}) {function_type = () -> (!mo.tensor<2x3xf32>, !mo.tensor<3x2xf32>), sym_name = "two_constants"} : () -> ()
