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
// CHECK-NOT: module
mo.graph @support_convert_prelu() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant {value = #M.dense_array<-2.0, -1.0, 0.0, 1.0, 2.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %1 = mo.constant {value = #M.dense_array<0.2> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = "prelu"(%0, %1) : (!mo.tensor<[5], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}
