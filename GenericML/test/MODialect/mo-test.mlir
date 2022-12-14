// RUN: mo-opt -cse %s | mo-opt | FileCheck %s

// CHECK: mo.graph @dyn_rank(%[[ARG0:.*]]: !mo.tensor<*f32>) -> !mo.tensor<*f32> {
mo.graph @dyn_rank(%arg0: !mo.tensor<*f32>) -> (!mo.tensor<*f32>) {
  mo.output %arg0 : !mo.tensor<*f32>
}

// CHECK: mo.graph @dyn_dims(%[[ARG0:.*]]: !mo.tensor<2x?x1xf32>) -> !mo.tensor<2x?x1xf32> {
mo.graph @dyn_dims(%arg0: !mo.tensor<2x?x1xf32>) -> (!mo.tensor<2x?x1xf32>) {
  mo.output %arg0 : !mo.tensor<2x?x1xf32>
}

// CHECK: mo.graph @scalar(%[[ARG0:.*]]: !mo.tensor<f32>) -> !mo.tensor<f32> {
mo.graph @scalar(%arg0: !mo.tensor<f32>) -> (!mo.tensor<f32>) {
  mo.output %arg0 : !mo.tensor<f32>
}

// CHECK: mo.graph @invalid_type(%[[ARG0:.*]]: !mo.tensor<2x3xinvalid>) -> !mo.tensor<2x3xinvalid> {
mo.graph @invalid_type(%arg0: !mo.tensor<2x3xinvalid>) -> (!mo.tensor<2x3xinvalid>) {
  mo.output %arg0 : !mo.tensor<2x3xinvalid>
}

// CHECK: mo.graph @constant() -> !mo.tensor<2x3xf32>
mo.graph @constant() -> !mo.tensor<2x3xf32> {
  // CHECK: %[[VAL_0:.*]] = "mo.constant"() {value = #M.dense_array<1.{{0+}}e+00, 2.{{0+}}e+00, 3.{{0+}}e+00, 4.{{0+}}e+00, 5.{{0+}}e+00, 6.{{0+}}e+00> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
  %0 = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
  // CHECK: mo.output %[[VAL_0]] : !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// CHECK: mo.graph @constant_cse() -> (!mo.tensor<2xui32>, !mo.tensor<2xui32>)
mo.graph @constant_cse() -> (!mo.tensor<2xui32>, !mo.tensor<2xui32>) {
  // CHECK: %[[VAL0:.*]] = "mo.constant"() {value = #M.dense_array<1, 2> : tensor<2xui32>} : () -> !mo.tensor<2xui32>
  %0 = "mo.constant"() {value = #M.dense_array<1, 2> : tensor<2xui32>} : () -> !mo.tensor<2xui32>
  %1 = "mo.constant"() {value = #M.dense_array<1, 2> : tensor<2xui32>} : () -> !mo.tensor<2xui32>
  // CHECK: mo.output %[[VAL0]], %[[VAL0]] : !mo.tensor<2xui32>, !mo.tensor<2xui32>
  mo.output %0, %1 : !mo.tensor<2xui32>, !mo.tensor<2xui32>
}

// CHECK: mo.graph @two_constants() -> (!mo.tensor<2x3xf32>, !mo.tensor<4xsi8>)
mo.graph @two_constants() -> (!mo.tensor<2x3xf32>, !mo.tensor<4xsi8>) {
  // CHECK-DAG: %[[VAL_0:.*]] = "mo.constant"() {value = #M.dense_array<0.{{0+}}e+00, 1.{{0+}}e+00, 2.{{0+}}e+00, 3.{{0+}}e+00, 4.{{0+}}e+00, 5.{{0+}}e+00> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
  // CHECK-DAG: %[[VAL_1:.*]] = "mo.constant"() {value = dense_resource<mtfl_buffer> : tensor<4xsi8>} : () -> !mo.tensor<4xsi8>
  %0 = "mo.constant"() {value = #M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0> : tensor<2x3xf32>} : () -> !mo.tensor<2x3xf32>
  %1 = "mo.constant"() {value = dense_resource<mtfl_buffer> : tensor<4xsi8>} : () -> !mo.tensor<4xsi8>
  // CHECK: mo.output %[[VAL_0]], %[[VAL_1]] : !mo.tensor<2x3xf32>, !mo.tensor<4xsi8>
  mo.output %0, %1 : !mo.tensor<2x3xf32>, !mo.tensor<4xsi8>
}

{-#
  dialect_resources: {
    builtin: {
      mtfl_buffer: "0x4000000087654321"
    }
  }
#-}
