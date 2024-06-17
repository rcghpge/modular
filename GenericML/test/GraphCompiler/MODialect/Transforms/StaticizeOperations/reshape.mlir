// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @reshape
//  CHECK-SAME:   <[[N:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], f32>,
mo.graph @reshape<N>(
  %input: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[N], f32>
) -> !mo.tensor<[1], si64> {

  // CHECK: [[MOSH_SHAPE:%.*]] = mosh.param.to_value: !mosh.ape = <[[[N]]]>
  // CHECK: [[RES:%.*]] = mo.shape.to_tensor(%0) -> !mo.tensor<[1], si64>
  %n = mosh.param.to_value = <N>
  %n_scalar = mo.index.to_tensor(%n) -> !mo.tensor<[], si64>
  %one = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %res = mo.reshape(%n_scalar, %one) : (!mo.tensor<[], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output %res : !mo.tensor<[1], si64>
}
