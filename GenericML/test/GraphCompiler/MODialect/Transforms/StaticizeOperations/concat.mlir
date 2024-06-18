// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s


// CHECK-LABEL: mo.graph @concat
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]]>(
mo.graph @concat<N1, N2>(
  %dummy1: !mo.tensor<[N1], f32>,
  %dummy2: !mo.tensor<[N2], f32>
) -> !mo.tensor<[3], si32> {

  %axis = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %one = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>

  // CHECK: [[MOSH_SHAPE:%.*]] = mosh.param.to_value: !mosh.ape = <[[[N1]], 1, [[N2]]]>
  // CHECK: [[RES:%.*]] = mo.shape.to_tensor([[MOSH_SHAPE]]) -> !mo.tensor<[3], si32>
  %n1 = mo.shape_of(%dummy1) : (!mo.tensor<[N1], f32>) -> !mo.tensor<[1], si32>
  %n2 = mo.shape_of(%dummy2) : (!mo.tensor<[N2], f32>) -> !mo.tensor<[1], si32>
  %res = mo.concat[%axis: !mo.tensor<[], si32>](%n1, %one, %n2) : (
    !mo.tensor<[1], si32>, !mo.tensor<[1], si32>, !mo.tensor<[1], si32>
  ) -> !mo.tensor<[3], si32>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output %res : !mo.tensor<[3], si32>
}
