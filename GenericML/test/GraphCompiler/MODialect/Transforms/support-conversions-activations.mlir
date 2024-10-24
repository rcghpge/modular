// RUN: mo-opt %s --split-input-file --support-conversions-test --legalize-rmo-operators --constant-fold --shape-inference="constant-fold=False" --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_prelu() -> !mo.tensor<[5], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<-2.000000e+00, -1.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<2.000000e-01> : tensor<1xf32>} : !mo.tensor<[1], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant {value = #M.dense_array<0.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.broadcast_to([[V2]]) : !mo.tensor<[], f32> -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[1], f32> -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.min([[V3]], [[V0]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.mul([[V4]], [[V5]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: [[V7:[^ ]*]] = mo.max([[V3]], [[V0]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: [[V8:[^ ]*]] = mo.add([[V7]], [[V6]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output [[V8]] : !mo.tensor<[5], f32>
mo.graph @support_convert_prelu() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant {value = #M.dense_array<-2.0, -1.0, 0.0, 1.0, 2.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %1 = mo.constant {value = #M.dense_array<0.2> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = "prelu"(%0, %1) : (!mo.tensor<[5], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[5], f32>
  mo.output %2 : !mo.tensor<[5], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_prelu_mixed_types() -> !mo.tensor<[3], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<3.000000e-01> : tensor<1xf32>} : !mo.tensor<[1], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<0.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant {value = #M.dense_array<2.000000e+00, 1.000000e+00, 3.000000e+00> : !M.array<3xf32>} : !mo.tensor<[3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[], f32> -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[1], f32> -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.min([[V3]], [[V2]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.mul([[V4]], [[V5]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[V7:[^ ]*]] = mo.max([[V3]], [[V2]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[V8:[^ ]*]] = mo.add([[V7]], [[V6]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[V9:[^ ]*]] = mo.cast([[V8]]) : (!mo.tensor<[3], f32>) -> !mo.tensor<[3], si32>
// CHECK-NEXT: mo.output [[V9]] : !mo.tensor<[3], si32>
mo.graph @support_convert_prelu_mixed_types() -> (!mo.tensor<[3], si32>) {
  %0 = mo.constant {value = #M.dense_array<2, 1, 3> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %1 = mo.constant {value = #M.dense_array<0.3> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = "prelu"(%0, %1) : (!mo.tensor<[3], si32>, !mo.tensor<[1], f32>) -> !mo.tensor<[3], si32>
  mo.output %2 : !mo.tensor<[3], si32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_relu() -> !mo.tensor<[3], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<-3.000000e+00, 0.000000e+00, 5.000000e+00> : tensor<3xf32>} : !mo.tensor<[3], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<0.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[], f32> -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.max([[V2]], [[V0]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: mo.output [[V3]] : !mo.tensor<[3], f32>
mo.graph @support_convert_relu() -> (!mo.tensor<[3], f32>) {
  %0 = mo.constant {value = #M.dense_array<-3.0, 0.0, 5.0> : tensor<3xf32>} : !mo.tensor<[3], f32>
  %1 = "relu"(%0) : (!mo.tensor<[3], f32>) -> !mo.tensor<[3], f32>
  mo.output %1 : !mo.tensor<[3], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_relu_mixed_types() -> !mo.tensor<[3], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<0.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<3.000000e+00, 0.000000e+00, 5.000000e+00> : !M.array<3xf32>} : !mo.tensor<[3], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.static.broadcast_to([[V0]]) : !mo.tensor<[], f32> -> !mo.tensor<[3], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.max([[V2]], [[V1]]) : !mo.tensor<[3], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.cast([[V3]]) : (!mo.tensor<[3], f32>) -> !mo.tensor<[3], si32>
// CHECK-NEXT: mo.output [[V4]] : !mo.tensor<[3], si32>
mo.graph @support_convert_relu_mixed_types() -> (!mo.tensor<[3], si32>) {
  %0 = mo.constant {value = #M.dense_array<3, 0, 5> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %1 = "relu"(%0) : (!mo.tensor<[3], si32>) -> !mo.tensor<[3], si32>
  mo.output %1 : !mo.tensor<[3], si32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_relu6() -> !mo.tensor<[5], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<-2.000000e+00, 1.800000e+01, 0.000000e+00, 5.000000e+00, 6.000000e+00> : tensor<5xf32>} : !mo.tensor<[5], f32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<6.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant {value = #M.dense_array<0.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[], f32> -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.static.broadcast_to([[V2]]) : !mo.tensor<[], f32> -> !mo.tensor<[5], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.min([[V3]], [[V0]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.max([[V4]], [[V5]]) : !mo.tensor<[5], f32>
// CHECK-NEXT: mo.output [[V6]] : !mo.tensor<[5], f32>
mo.graph @support_convert_relu6() -> (!mo.tensor<[5], f32>) {
  %0 = mo.constant {value = #M.dense_array<-2.0, 18.0, 0.0, 5.0, 6.0> : tensor<5xf32>} : !mo.tensor<[5], f32>
  %1 = "relu6"(%0) : (!mo.tensor<[5], f32>) -> !mo.tensor<[5], f32>
  mo.output %1 : !mo.tensor<[5], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @support_convert_relu6_mixed_types() -> !mo.tensor<[2], si32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant {value = #M.dense_array<7, 8> : tensor<2xsi32>} : !mo.tensor<[2], si32>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant {value = #M.dense_array<6> : !M.array<1xsi32>} : !mo.tensor<[], si32>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant {value = #M.dense_array<0> : !M.array<1xsi32>} : !mo.tensor<[], si32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.broadcast_to([[V1]]) : !mo.tensor<[], si32> -> !mo.tensor<[2], si32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.static.broadcast_to([[V2]]) : !mo.tensor<[], si32> -> !mo.tensor<[2], si32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.min([[V3]], [[V0]]) : !mo.tensor<[2], si32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.max([[V4]], [[V5]]) : !mo.tensor<[2], si32>
// CHECK-NEXT: mo.output [[V6]] : !mo.tensor<[2], si32>
mo.graph @support_convert_relu6_mixed_types() -> (!mo.tensor<[2], si32>) {
  %0 = mo.constant {value = #M.dense_array<7, 8> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %1 = "relu6"(%0) : (!mo.tensor<[2], si32>) -> !mo.tensor<[2], si32>
  mo.output %1 : !mo.tensor<[2], si32>
}
