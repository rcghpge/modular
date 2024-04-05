// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @scatter_nd_ok(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @scatter_nd_static_ok(%arg0: !mo.tensor<[4, 2], f32>, %updates: !mo.tensor<[1, 3, 2], f32>, %indices: !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[4, 2], f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[4, 2], f32>
  mo.output %0 : !mo.tensor<[4, 2], f32>
}

// -----

mo.graph @scatter_nd_bad_updaes(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f16>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.scatter_nd' op failed to verify that all of {input, updates, result} have same dtype}}
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f16>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @scatter_nd_bad_indices(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, f32>) -> !mo.tensor<?, f32> {
  // expected-error @+1 {{'rmo.mo.scatter_nd' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<?, f32>'}}
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, f32>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @scatter_nd_bad_result(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f16> {
  // expected-error @+1{{'rmo.mo.scatter_nd' op failed to verify that all of {input, updates, result} have same dtype}}
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f16>
  mo.output %0 : !mo.tensor<?, f16>
}

// -----

mo.graph @scatter_nd_add_ok(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd.add(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @scatter_nd_max_ok(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd.max(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @scatter_nd_min_ok(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd.min(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}

// -----

mo.graph @scatter_nd_mul_ok(%arg0: !mo.tensor<?, f32>, %updates: !mo.tensor<?, f32>, %indices: !mo.tensor<?, si64>) -> !mo.tensor<?, f32> {
  %0 = rmo.mo.scatter_nd.mul(%arg0, %updates, %indices) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>, !mo.tensor<?, si64>) -> !mo.tensor<?, f32>
  mo.output %0 : !mo.tensor<?, f32>
}
