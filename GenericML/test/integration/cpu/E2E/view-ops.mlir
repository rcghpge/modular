// RUN: mo-opt %s --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'broadcast_to_fallback_shape_func':
// CHECK: 'broadcast_to_fallback_shape_func' returned tensor<1x2x3x4xsi32> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mo.graph @broadcast_to_fallback_shape_func(%0 : !mo.tensor<[], si32>) -> !mo.tensor<[1, 2, 3, 4], si32> {
  %1 = mo.static.broadcast_to(%0) : !mo.tensor<[], si32> -> !mo.tensor<[1, 2, 3, 4], si32>
  mo.output %1 : !mo.tensor<[1, 2, 3, 4], si32>
}

// CHECK-LABEL: Running 'broadcast_constant':
// CHECK: 'broadcast_constant' returned tensor<1x2x3x4xsi32> [1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2, 1, -1, -2, 2]
mo.graph @broadcast_constant() -> !mo.tensor<[1, 2, 3, 4], si32> {
  %0 = "mo.constant"() {value = #M.dense_array<1, -1, -2, 2> : tensor<4xsi32>} : () -> !mo.tensor<[4], si32>
  %1 = mo.static.broadcast_to(%0) : !mo.tensor<[4], si32> -> !mo.tensor<[1, 2, 3, 4], si32>
  mo.output %1 : !mo.tensor<[1, 2, 3, 4], si32>
}

// CHECK-LABEL: Running 'transpose_cst':
// CHECK: 'transpose_cst' returned tensor<1x3x2xui32> [1, 4, 2, 5, 3, 6]
mo.graph @transpose_cst() -> (!mo.tensor<[1, 3, 2], ui32>) {
  %0 = mo.constant {value = #M.dense_array<1, 2, 3, 4, 5, 6> : tensor<6xui32>} : !mo.tensor<[1, 2, 3], ui32>
  %axis = mo.constant {value = #M.dense_array<2> : tensor<ui32>} : !mo.tensor<[], ui32>
  %perm  = mo.constant {value = #M.dense_array<0, 2, 1> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  %1 = mo.transpose(%0, %perm) : (!mo.tensor<[1, 2, 3], ui32>, !mo.tensor<[3], si64>) -> !mo.tensor<[1, 3, 2], ui32>
  mo.output %1 : !mo.tensor<[1, 3, 2], ui32>
}

// CHECK-LABEL: Running 'slice_dim_cst':
// CHECK: 'slice_dim_cst' returned tensor<2x3xf32> [6.000000e+00, 8.000000e+00, 1.000000e+01, 1.100000e+01, 1.300000e+01, 1.500000e+01]
mo.graph @slice_dim_cst() -> !mo.tensor<[2, 3], f32> {
  %zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %one = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %two = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %three = mo.constant {value = #M.dense_array<3> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %five = mo.constant {value = #M.dense_array<5> : tensor<1xsi64>} : !mo.tensor<[], si64>

  %in = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0> : tensor<15xf32>} : !mo.tensor<[3, 5], f32>
  %res = mo.slice_dim[0](%in, %one, %three, %one) : (
    !mo.tensor<[3, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 5], f32>
  %res2 = mo.slice_dim[1](%res, %zero, %five, %two) : (
    !mo.tensor<[2, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 3], f32>
  mo.output %res2 : !mo.tensor<[2, 3], f32>
}
