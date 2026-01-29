// RUN: mo-opt %s --split-input-file --verify-diagnostics

mo.graph @scatter_nd_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd<() -> Sh3: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @scatter_nd_static_ok(%arg0: !mo.tensor<[4, 2], f32>, %updates: !mo.tensor<[1, 3, 2], f32>, %indices: !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[4, 2], f32> {
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>) -> !mo.tensor<[4, 2], f32>
  mo.output %0 : !mo.tensor<[4, 2], f32>
}

// -----

mo.graph @scatter_nd_bad_updaes<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f16>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  // expected-error @+1 {{'rmo.mo.scatter_nd' op failed to verify that all of {input, updates, result} have same dtype}}
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f16>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @scatter_nd_bad_indices<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32> {
  // expected-error @+1 {{'rmo.mo.scatter_nd' op operand #2 must be tensor with integer or index elements, but got '!mo.tensor<Sh2, f32>'}}
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, f32>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @scatter_nd_bad_result<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f16> {
  // expected-error @+1{{'rmo.mo.scatter_nd' op failed to verify that all of {input, updates, result} have same dtype}}
  %0 = rmo.mo.scatter_nd(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f16>
  mo.output %0 : !mo.tensor<Sh3, f16>
}

// -----

mo.graph @scatter_nd_add_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd.add<() -> Sh3: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @scatter_nd_max_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd.max<() -> Sh3: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @scatter_nd_min_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd.min<() -> Sh3: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

// -----

mo.graph @scatter_nd_mul_ok<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape>(%arg0: !mo.tensor<Sh0, f32>, %updates: !mo.tensor<Sh1, f32>, %indices: !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32> {
  %0 = rmo.mo.scatter_nd.mul<() -> Sh3: !mosh.ape>(%arg0, %updates, %indices) : (!mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>, !mo.tensor<Sh2, si64>) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}
