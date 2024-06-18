// RUN: mo-opt %s --propagate-shapes --canonicalize --verify-diagnostics


mo.graph @gather_error<N1, N2>(
  %dummy: !mo.tensor<[N1, 10, N2], f32>
) -> !mo.tensor<[2], si32> {

  %axis = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %indices = mo.constant {value = #M.dense_array<-1, 3> : tensor<2xsi32>} : !mo.tensor<[2], si32>
  %dummy_shape = mo.shape_of(%dummy) : (!mo.tensor<[N1, 10, N2], f32>) -> !mo.tensor<[3], si32>

  // expected-error@+1 {{[gather] index out of bound: 3 for input size: 3}}
  %res = mo.gather(%dummy_shape, %indices, %axis) : (
    !mo.tensor<[3], si32>, !mo.tensor<[2], si32>, !mo.tensor<[], si32>
  ) -> !mo.tensor<[2], si32>

  mo.output %res : !mo.tensor<[2], si32>
}
