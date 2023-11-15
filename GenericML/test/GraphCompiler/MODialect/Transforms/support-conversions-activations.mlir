// RUN: mo-opt %s --support-conversions-test --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: support_convert_prelu
// CHECK-NEXT: [[ARG0:%.*]] = mo.constant {{.*}}-2.0{{.*}}-1.0{{.*}}0.0{{.*}}1.0{{.*}}2.0{{.*}} : !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG1:%.*]] = mo.constant {{.*}}2.0{{.*}} : !mo.tensor<[1], f32>
// CHECK-NEXT: [[ARG2:%.*]] = mo.constant {{.*}}0.0{{.*}} : !mo.tensor<[], f32>
// CHECK-NEXT: [[ARG3:%.*]] = mo.shape_of([[ARG0]]) : (!mo.tensor<[5], f32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: [[ARG4:%.*]] = mo.broadcast_to([[ARG2]], [[ARG3]]) : (!mo.tensor<[], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG5:%.*]] = mo.shape_of([[ARG0]]) : (!mo.tensor<[5], f32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: [[ARG6:%.*]] = mo.broadcast_to([[ARG1]], [[ARG5]]) : (!mo.tensor<[1], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG7:%.*]] = mo.min([[ARG0]], [[ARG4]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG8:%.*]] = mo.mul([[ARG6]], [[ARG7]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG9:%.*]] = mo.max([[ARG0]], [[ARG4]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG10:%.*]] = mo.add([[ARG9]], [[ARG8]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output [[ARG10]] : !mo.tensor<[5], f32>
mo.graph @support_convert_prelu() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant {value = #M.dense_array<-2.0, -1.0, 0.0, 1.0, 2.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %1 = mo.constant {value = #M.dense_array<0.2> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = "prelu"(%0, %1) : (!mo.tensor<[5], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}

// CHECK-LABEL: support_convert_prelu_mixed_types
// CHECK-NEXT: [[ARG0:%.*]] = mo.constant {{.*}}2{{.*}}1{{.*}}3{{.*}} : !mo.tensor<[3], si32>
// CHECK-NEXT: [[ARG1:%.*]] = mo.constant {{.*}}3.0{{.*}} : !mo.tensor<[1], f32>
// CHECK-NEXT: [[ARG2:%.*]] = mo.constant {{.*}}0.0{{.*}} : !mo.tensor<[], f32>
// CHECK-NEXT: [[ARG3:%.*]] = mo.cast([[ARG0]]) : (!mo.tensor<[3], si32>) -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG4:%.*]] = mo.shape_of([[ARG3]]) : (!mo.tensor<[3], f32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: [[ARG5:%.*]] = mo.broadcast_to([[ARG2]], [[ARG4]]) : (!mo.tensor<[], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG6:%.*]] = mo.shape_of([[ARG3]]) : (!mo.tensor<[3], f32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: [[ARG7:%.*]] = mo.broadcast_to([[ARG1]], [[ARG6]]) : (!mo.tensor<[1], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG8:%.*]] = mo.min([[ARG3]], [[ARG5]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG9:%.*]] = mo.mul([[ARG7]], [[ARG8]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG10:%.*]] = mo.max([[ARG3]], [[ARG5]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG11:%.*]] = mo.add([[ARG10]], [[ARG9]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG12:%.*]] = mo.cast([[ARG11]]) : (!mo.tensor<[3], f32>) -> !mo.tensor<[3], si32>
// CHECK-NEXT: mo.output [[ARG12]] : !mo.tensor<[3], si32>
mo.graph @support_convert_prelu_mixed_types() -> (!mo.tensor<[3], si32>) {
  %0 = mo.constant {value = #M.dense_array<2, 1, 3> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %1 = mo.constant {value = #M.dense_array<0.3> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = "prelu"(%0, %1) : (!mo.tensor<[3], si32>, !mo.tensor<[1], f32>) -> !mo.tensor<[3], si32>
  mo.output %2 : !mo.tensor<[3], si32>
}

// CHECK-LABEL: support_convert_relu
// CHECK-NEXT: [[ARG0:%.*]] = mo.constant {{.*}}-3.0{{.*}}0.0{{.*}}5.0{{.*}} : !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG1:%.*]] = mo.constant {{.*}}0.0{{.*}} : !mo.tensor<[], f32>
// CHECK-NEXT: [[ARG2:%.*]] = mo.shape_of([[ARG0]]) : (!mo.tensor<[3], f32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: [[ARG3:%.*]] = mo.broadcast_to([[ARG1]], [[ARG2]]) : (!mo.tensor<[], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG4:%.*]] = mo.max([[ARG0]], [[ARG3]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: mo.output [[ARG4]] : !mo.tensor<[3], f32>
mo.graph @support_convert_relu() -> (!mo.tensor<[3], f32>) {
  %0 = mo.constant {value = #M.dense_array<-3.0, 0.0, 5.0> : tensor<3xf32>} : !mo.tensor<[3], f32>
  %1 = "relu"(%0) : (!mo.tensor<[3], f32>) -> !mo.tensor<[3], f32>
  mo.output %1 : !mo.tensor<[3], f32>
}

// CHECK-LABEL: support_convert_relu_mixed_types
// CHECK-NEXT: [[ARG0:%.*]] = mo.constant {value = #M.dense_array<3, 0, 5> : tensor<3xsi32>} : !mo.tensor<[3], si32>
// CHECK-NEXT: [[ARG1:%.*]] = mo.constant {{.*}}0.0{{.*}} : !mo.tensor<[], f32>
// CHECK-NEXT: [[ARG2:%.*]] = mo.cast([[ARG0]]) : (!mo.tensor<[3], si32>) -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG3:%.*]] = mo.shape_of([[ARG2]]) : (!mo.tensor<[3], f32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: [[ARG4:%.*]] = mo.broadcast_to([[ARG1]], [[ARG3]]) : (!mo.tensor<[], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG5:%.*]] = mo.max([[ARG2]], [[ARG4]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[ARG6:%.*]] = mo.cast([[ARG5]]) : (!mo.tensor<[3], f32>) -> !mo.tensor<[3], si32>
// CHECK-NEXT: mo.output [[ARG6]] : !mo.tensor<[3], si32>
mo.graph @support_convert_relu_mixed_types() -> (!mo.tensor<[3], si32>) {
  %0 = mo.constant {value = #M.dense_array<3, 0, 5> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %1 = "relu"(%0) : (!mo.tensor<[3], si32>) -> !mo.tensor<[3], si32>
  mo.output %1 : !mo.tensor<[3], si32>
}

// CHECK-LABEL: support_convert_relu6
// CHECK-NEXT: [[ARG0:%.*]] = mo.constant {{.*}}-2.0{{.*}}1.8{{.*}}0.0{{.*}}5.0{{.*}}6.0{{.*}} : !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG1:%.*]] = mo.constant {{.*}}6.0{{.*}} : !mo.tensor<[], f32>
// CHECK-NEXT: [[ARG2:%.*]] = mo.constant {{.*}}0.0{{.*}} : !mo.tensor<[], f32>
// CHECK-NEXT: [[ARG3:%.*]] = mo.shape_of([[ARG0]]) : (!mo.tensor<[5], f32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: [[ARG4:%.*]] = mo.broadcast_to([[ARG1]], [[ARG3]]) : (!mo.tensor<[], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG5:%.*]] = mo.shape_of([[ARG0]]) : (!mo.tensor<[5], f32>) -> !mo.tensor<[1], si32>
// CHECK-NEXT: [[ARG6:%.*]] = mo.broadcast_to([[ARG2]], [[ARG5]]) : (!mo.tensor<[], f32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG7:%.*]] = mo.min([[ARG0]], [[ARG4]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: [[ARG8:%.*]] = mo.max([[ARG6]], [[ARG7]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output [[ARG8]] : !mo.tensor<[5], f32>
mo.graph @support_convert_relu6() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant {value = #M.dense_array<-2.0, 18.0, 0.0, 5.0, 6.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %1 = "relu6"(%0) : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %1 : !mo.tensor<[5], f32>
}

    // %0 = mo.constant {value = #M.dense_array<7, 8> : tensor<2xsi32>} : !mo.tensor<[2], si32>
    // %1 = mo.constant {value = #M.dense_array<6.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
    // %2 = mo.constant {value = #M.dense_array<0.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
    // %3 = mo.cast(%1) : (!mo.tensor<[], f32>) -> !mo.tensor<[], si32>
    // %4 = mo.shape_of(%0) : (!mo.tensor<[2], si32>) -> !mo.tensor<[1], si32>
    // %5 = mo.broadcast_to(%3, %4) : (!mo.tensor<[], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[2], si32>
    // %6 = mo.cast(%2) : (!mo.tensor<[], f32>) -> !mo.tensor<[], si32>
    // %7 = mo.shape_of(%0) : (!mo.tensor<[2], si32>) -> !mo.tensor<[1], si32>
    // %8 = mo.broadcast_to(%6, %7) : (!mo.tensor<[], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[2], si32>
    // %9 = mo.min(%0, %5) : !mo.tensor<[2], si32>
    // %10 = mo.max(%8, %9) : !mo.tensor<[2], si32>
    // mo.output %10 : !mo.tensor<[2], si32>

// CHECK-LABEL: support_convert_relu6_mixed_types
// CHECK-NEXT: [[ARG0:%.*]] = mo.constant {{.*}}7{{.*}}8{{.*}} : !mo.tensor<[2], si32>
// CHECK-NEXT: [[ARG1:%.*]] = mo.constant {{.*}}6.0{{.*}} : !mo.tensor<[], f32>
// CHECK-NEXT: [[ARG2:%.*]] = mo.constant {{.*}}0.0{{.*}} : !mo.tensor<[], f32>
mo.graph @support_convert_relu6_mixed_types() -> (!mo.tensor<[2], si32>) {
  %0 = mo.constant {value = #M.dense_array<7, 8> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %1 = "relu6"(%0) : (!mo.tensor<[2], si32>) -> !mo.tensor<[2], si32>
  mo.output %1 : !mo.tensor<[2], si32>
}
