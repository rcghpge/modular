// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s


// CHECK-LABEL: mo.graph @reduce_mul
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]]>(
mo.graph @reduce_mul<N1, N2>(
  %dummy: !mo.tensor<[N1, 10, N2], f32>
) -> !mo.tensor<[1], si32> {

  %axis = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>

  // CHECK: [[MOSH_SHAPE:%.*]] = mosh.param.to_value: !mosh.ape = <[mul_nuw([[N1]], [[N2]], 10)]>
  // CHECK: [[RES:%.*]] = mo.shape.to_tensor([[MOSH_SHAPE]]) -> !mo.tensor<[1], si32>
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, 10, N2], f32>) -> !mo.tensor<[3], si32>
  %res = mo.reduce.mul(%dummy_shape, %axis) : (
    !mo.tensor<[3], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[1], si32>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output %res : !mo.tensor<[1], si32>
}
