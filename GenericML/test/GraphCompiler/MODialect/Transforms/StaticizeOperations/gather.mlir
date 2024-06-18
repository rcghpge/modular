// RUN: mo-opt %s --propagate-shapes --canonicalize | FileCheck %s


// CHECK-LABEL: mo.graph @gather_scalar_indices
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]]>(
mo.graph @gather_scalar_indices<N1, N2>(
  %dummy: !mo.tensor<[N1, 10, N2], f32>
) -> !mo.tensor<[], si32> {

  %axis = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %idx = mo.constant {value = #M.dense_array<-1> : tensor<si32>} : !mo.tensor<[], si32>

  // CHECK: [[IDX:%.*]] = mosh.param.to_value = <[[N2]]>
  // CHECK: [[RES:%.*]] = mo.index.to_tensor([[IDX]]) -> !mo.tensor<[], si32>
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, 10, N2], f32>) -> !mo.tensor<[3], si32>
  %res = mo.gather(%dummy_shape, %idx, %axis) : (
    !mo.tensor<[3], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[], si32>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output %res : !mo.tensor<[], si32>
}


// CHECK-LABEL: mo.graph @gather_1d_indices
//  CHECK-SAME:   <[[N1:.*]], [[N2:.*]]>(
mo.graph @gather_1d_indices<N1, N2>(
  %dummy: !mo.tensor<[N1, 10, N2], f32>
) -> !mo.tensor<[2], si32> {

  %axis = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %indices = mo.constant {value = #M.dense_array<-1, 1> : tensor<2xsi32>} : !mo.tensor<[2], si32>

  // CHECK: [[MOSH_SHAPE:%.*]] = mosh.param.to_value: !mosh.ape = <[[[N2]], 10]>
  // CHECK: [[RES:%.*]] = mo.shape.to_tensor([[MOSH_SHAPE]]) -> !mo.tensor<[2], si32>
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, 10, N2], f32>) -> !mo.tensor<[3], si32>
  %res = mo.gather(%dummy_shape, %indices, %axis) : (
    !mo.tensor<[3], si32>, !mo.tensor<[2], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[2], si32>

  // CHECK-NEXT: mo.output [[RES]]
  mo.output %res : !mo.tensor<[2], si32>
}
