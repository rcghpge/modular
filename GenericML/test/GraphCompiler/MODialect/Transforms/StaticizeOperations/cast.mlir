// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s

// CHECK-LABEL: mo.graph @cast
//  CHECK-SAME:   <[[N:.*]]>(
//  CHECK-SAME:   [[INPUT:%.*]]: !mo.tensor<[], f32>,
mo.graph @cast<N>(
  %input: !mo.tensor<[], f32>,
  %dummy: !mo.tensor<[N], f32>
) -> !mo.tensor<[1], si64> {

  // CHECK: [[MOSH_SHAPE:%.*]] = mosh.param.to_value: !mosh.ape = <[[[N]]]>
  // CHECK: [[RES:%.*]] = mo.shape.to_tensor(%0) -> !mo.tensor<[1], si64>
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N], f32>) -> !mo.tensor<[1], si32>
  %res = mo.cast(%dummy_shape) : (!mo.tensor<[1], si32>) -> !mo.tensor<[1], si64>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output %res : !mo.tensor<[1], si64>
}
