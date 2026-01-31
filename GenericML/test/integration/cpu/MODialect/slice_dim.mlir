// RUN: mo-opt %s --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

mo.graph @basic_slice_dim(%arg0: !mo.tensor<[3, 5], f32>) -> !mo.tensor<[2, 3], f32> {
  %input = mo.constant {value = #M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>} : !mo.tensor<[3, 5], f32>

  %zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %one = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %two = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %three = mo.constant {value = #M.dense_array<3> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %five = mo.constant {value = #M.dense_array<5> : tensor<1xsi64>} : !mo.tensor<[], si64>

  %in = mo.add(%input, %arg0):  !mo.tensor<[3, 5], f32>

  %res = mo.slice_dim[0](%in, %one, %three, %one) : (
    !mo.tensor<[3, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 5], f32>
  %res2 = mo.slice_dim[1](%res, %zero, %five, %two) : (
    !mo.tensor<[2, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 3], f32>
  mo.output %res2 : !mo.tensor<[2, 3], f32>
}

// CHECK{LITERAL}: 'basic_slice_dim' returned tensor<2x3xf32> [5.000000e+00, 7.000000e+00, 9.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01]

mo.graph @slice_dim_neg_step(%arg0: !mo.tensor<[5, 3], f32>) -> !mo.tensor<[2, 2], f32> {
  %input = mo.constant {value = #M.dense_array<0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0> : tensor<15xf32>} : !mo.tensor<[5, 3], f32>
  %in = mo.add(%input, %arg0):  !mo.tensor<[5, 3], f32>

  %neg_two = mo.constant {value = #M.dense_array<-2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %neg_one = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %zero = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>

  %res = mo.slice_dim[0](%in, %neg_one, %zero, %neg_two) : (
    !mo.tensor<[5, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 3], f32>
  %res2 = mo.slice_dim[1](%res, %neg_one, %zero, %neg_one) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 2], f32>
  mo.output %res2 : !mo.tensor<[2, 2], f32>
}

// CHECK{LITERAL}: 'slice_dim_neg_step' returned tensor<2x2xf32> [1.400000e+01, 1.300000e+01, 8.000000e+00, 7.000000e+00]
