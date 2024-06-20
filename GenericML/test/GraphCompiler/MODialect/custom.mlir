// RUN: mo-opt --canonicalize %s| FileCheck %s

// CHECK:      mo.graph @custom_dynamic(%[[ARG0:.*]]: !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
// CHECK-NEXT:  %[[CUSTOM:.*]] = mo.custom {symbol = "test_custom_op"}(%[[ARG0]])
// CHECK-NEXT:  mo.output %[[CUSTOM]]

mo.graph @custom_dynamic(%arg0 : !mo.tensor<[?], f32>) -> !mo.tensor<[?], f32> {
  %0 = mo.custom {symbol = "test_custom_op"}(%arg0) : (!mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
  mo.output %0 : !mo.tensor<[?], f32>
}
