// RUN: mo-opt -mo-shape-inference -mo-symbolic-optimize -mogg-pattern-fusion %s | FileCheck %s --check-prefix=CHECK-IR

// CHECK-IR: mo.graph @tf_layer_norm(%[[ARG0:.*]]:  !mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 2], f32>
mo.graph @tf_layer_norm(%arg0 : !mo.tensor<[4, 3, 2], f32>) -> (!mo.tensor<[4, 3, 2], f32>) {
  // CHECK-IR: %[[EPS:.*]] = mo.constant {{{.*}}} : !mo.tensor<[], f32>
  // CHECK-IR: %[[GAM:.*]] = mo.constant {value = #M.dense_array<1.100{{.*}}} : !mo.tensor<[2], f32>
  // CHECK-IR: %[[BET:.*]] = mo.constant {value = #M.dense_array<-1.000{{.*}}} : !mo.tensor<[2], f32>
  // CHECK-IR: %[[VAL1:.*]] = mo.reduce.layer_norm(%[[ARG0]], %[[GAM]], %[[BET]], %[[EPS]])
  // CHECK-IR: mo.output %[[VAL1]]
  %sh = "mo.constant"() {value = #M.dense_array<4, 3, 2> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %sh2 = "mo.constant"() {value = #M.dense_array<4, 3, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f32>} : () -> !mo.tensor<[], f32>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %1 = mo.reduce.mean(%arg0) {axis = 2 : index} : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %2 = mo.broadcast_to(%1, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %3 = mo.sub(%arg0, %2) : !mo.tensor<[4, 3, 2], f32>
  %4 = mo.mul(%3, %3) : !mo.tensor<[4, 3, 2], f32>
  %5 = mo.reduce.mean(%4) {axis = 2 : index} : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %6 = mo.broadcast_to(%eps, %sh2) : (!mo.tensor<[], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 1], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 3, 1], f32>
  %8 = mo.rsqrt(%7) : !mo.tensor<[4, 3, 1], f32>
  %9 = mo.broadcast_to(%8, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %10 = mo.broadcast_to(%gamma, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %11 = mo.mul(%9, %10) : !mo.tensor<[4, 3, 2], f32>
  %12 = mo.mul(%arg0, %11) : !mo.tensor<[4, 3, 2], f32>
  %13 = mo.mul(%2, %11) : !mo.tensor<[4, 3, 2], f32>
  %14 = mo.broadcast_to(%beta, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %15 = mo.sub(%14, %13) : !mo.tensor<[4, 3, 2], f32>
  %16 = mo.add(%12, %15) : !mo.tensor<[4, 3, 2], f32>
  mo.output %16 : !mo.tensor<[4, 3, 2], f32>
}

// CHECK-IR: mo.graph @tf_layer_norm_skip(%[[ARG0:.*]]:  !mo.tensor<[4, 3, 2], f32>, %[[ARG1:.*]]: !mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 2], f32>
mo.graph @tf_layer_norm_skip(%arg0 : !mo.tensor<[4, 3, 2], f32>, %arg1 : !mo.tensor<[4, 3, 2], f32>) -> (!mo.tensor<[4, 3, 2], f32>) {
  // CHECK-IR: %[[EPS:.*]] = mo.constant {{{.*}}} : !mo.tensor<[], f32>
  // CHECK-IR: %[[GAM:.*]] = mo.constant {value = #M.dense_array<1.100{{.*}}} : !mo.tensor<[2], f32>
  // CHECK-IR: %[[BET:.*]] = mo.constant {value = #M.dense_array<-1.000{{.*}}} : !mo.tensor<[2], f32>
  // CHECK-IR: %[[VAL:.*]] = mo.add(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[4, 3, 2], f32>
  // CHECK-IR: %[[VAL1:.*]] = mo.reduce.layer_norm(%[[VAL]], %[[GAM]], %[[BET]], %[[EPS]])
  // CHECK-IR: mo.output %[[VAL1]]
  %sh = "mo.constant"() {value = #M.dense_array<4, 3, 2> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %sh2 = "mo.constant"() {value = #M.dense_array<4, 3, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f32>} : () -> !mo.tensor<[], f32>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[4, 3, 2], f32>
  %1 = mo.reduce.mean(%0) {axis = 2 : index} : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %2 = mo.broadcast_to(%1, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %3 = mo.sub(%0, %2) : !mo.tensor<[4, 3, 2], f32>
  %4 = mo.mul(%3, %3) : !mo.tensor<[4, 3, 2], f32>
  %5 = mo.reduce.mean(%4) {axis = 2 : index} : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %6 = mo.broadcast_to(%eps, %sh2) : (!mo.tensor<[], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 1], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 3, 1], f32>
  %8 = mo.rsqrt(%7) : !mo.tensor<[4, 3, 1], f32>
  %9 = mo.broadcast_to(%8, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %10 = mo.broadcast_to(%gamma, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %11 = mo.mul(%9, %10) : !mo.tensor<[4, 3, 2], f32>
  %12 = mo.mul(%0, %11) : !mo.tensor<[4, 3, 2], f32>
  %13 = mo.mul(%2, %11) : !mo.tensor<[4, 3, 2], f32>
  %14 = mo.broadcast_to(%beta, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %15 = mo.sub(%14, %13) : !mo.tensor<[4, 3, 2], f32>
  %16 = mo.add(%12, %15) : !mo.tensor<[4, 3, 2], f32>
  mo.output %16 : !mo.tensor<[4, 3, 2], f32>
}

// CHECK-IR: mo.graph @pt_layer_norm(%[[ARG0:.*]]:  !mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 2], f32>
mo.graph @pt_layer_norm(%arg0 : !mo.tensor<[4, 3, 2], f32>) -> (!mo.tensor<[4, 3, 2], f32>) {
  // CHECK-IR: %[[EPS:.*]] = mo.constant {{{.*}}} : !mo.tensor<[], f32>
  // CHECK-IR: %[[GAM:.*]] = mo.constant {value = #M.dense_array<1.100{{.*}}} : !mo.tensor<[2], f32>
  // CHECK-IR: %[[BET:.*]] = mo.constant {value = #M.dense_array<-1.000{{.*}}} : !mo.tensor<[2], f32>
  // CHECK-IR: %[[VAL1:.*]] = mo.reduce.layer_norm(%[[ARG0]], %[[GAM]], %[[BET]], %[[EPS]])
  // CHECK-IR: mo.output %[[VAL1]]
  %sh = "mo.constant"() {value = #M.dense_array<4, 3, 2> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %sh2 = "mo.constant"() {value = #M.dense_array<4, 3, 1> : tensor<3xsi64>} : () -> !mo.tensor<[3], si64>
  %eps = "mo.constant"() {value = #M.dense_array<0.00001> : tensor<f32>} : () -> !mo.tensor<[], f32>
  %gamma = "mo.constant"() {value = #M.dense_array<1.1, 2.1> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %beta = "mo.constant"() {value = #M.dense_array<-1.0, -2.0> : tensor<2xf32>} : () -> !mo.tensor<[2], f32>
  %1 = mo.reduce.mean(%arg0) {axis = 2 : index} : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %2 = mo.broadcast_to(%1, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %3 = mo.sub(%arg0, %2) : !mo.tensor<[4, 3, 2], f32>
  %4 = mo.mul(%3, %3) : !mo.tensor<[4, 3, 2], f32>
  %5 = mo.reduce.mean(%4) {axis = 2 : index} : (!mo.tensor<[4, 3, 2], f32>) -> !mo.tensor<[4, 3, 1], f32>
  %6 = mo.broadcast_to(%eps, %sh2) : (!mo.tensor<[], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 1], f32>
  %7 = mo.add(%5, %6) : !mo.tensor<[4, 3, 1], f32>
  %8 = mo.sqrt(%7) : !mo.tensor<[4, 3, 1], f32>
  %9 = mo.broadcast_to(%8, %sh) : (!mo.tensor<[4, 3, 1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %10 = mo.div(%3, %9) : !mo.tensor<[4, 3, 2], f32>
  %11 = mo.broadcast_to(%gamma, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %12 = mo.mul(%10, %11) : !mo.tensor<[4, 3, 2], f32>
  %13 = mo.broadcast_to(%beta, %sh) : (!mo.tensor<[2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[4, 3, 2], f32>
  %14 = mo.add(%12, %13) : !mo.tensor<[4, 3, 2], f32>
  mo.output %14 : !mo.tensor<[4, 3, 2], f32>
}
