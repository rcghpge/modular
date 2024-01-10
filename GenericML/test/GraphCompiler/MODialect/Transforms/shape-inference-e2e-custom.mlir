// RUN: mo-opt --shape-inference="experimental=true kernel-library-paths=%extra_test_lib" %s | FileCheck %s


// CHECK-LABEL: mo.graph @custom_op
// CHECK-SAME: <[[D0:.*]], [[D1:.*]]>(
// CHECK-SAME:   %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], f32>
// CHECK-SAME: ) -> !mo.tensor<[[[D0]], [[D1]]], f32> no_inline {
mo.graph @custom_op(
  %input: !mo.tensor<[?, ?], f32>
) -> !mo.tensor<[?, ?], f32> {

  // CHECK-NEXT:  %[[RES:.*]] = mo.custom {symbol = "test_custom_op"}(%[[INPUT]])
  // CHECK-SAME:    (!mo.tensor<[[[D0]], [[D1]]], f32>) -> !mo.tensor<[[[D0]], [[D1]]], f32>
  // CHECK-NEXT:  mo.output %[[RES]] : !mo.tensor<[[[D0]], [[D1]]], f32>
  %res = mo.custom {symbol = "test_custom_op"}(%input) (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

