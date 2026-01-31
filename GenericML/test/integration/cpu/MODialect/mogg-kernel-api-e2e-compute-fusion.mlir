// RUN: mo-opt %s -o %t.mlir --mo-to-mgp="enable-compute-lambdas=true constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir --result-output-style=full | FileCheck %s

// CHECK: has_compute_lambda = True
// CHECK-NEXT: has_store_lambda = False
// CHECK-LABEL: Running 'test_compute_out_no_fusion':
// CHECK-NEXT: 'test_compute_out_no_fusion' returned tensor<1xf32> [4.000000e+00]
mo.graph @test_compute_out_no_fusion() -> !mo.tensor<[1], f32> {
  %0 = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %1 = mo.custom {symbol = "my_square_compute_out_fusion", device = #M.device_ref<"cpu", 0>}(%0) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  mo.output %1 : !mo.tensor<[1], f32>
}

// CHECK: has_compute_lambda = True
// CHECK-NEXT: has_store_lambda = False
// CHECK-LABEL: Running 'test_compute_out_simple_fusion':
// CHECK-NEXT: 'test_compute_out_simple_fusion' returned tensor<1xf32> [-5.000000e+00]
mo.graph @test_compute_out_simple_fusion() -> !mo.tensor<[1], f32> {
  %0 = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %1 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = mo.custom {symbol = "my_square_compute_out_fusion", device = #M.device_ref<"cpu", 0>}(%0) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  %3 = mo.add(%2, %1) :  !mo.tensor<[1], f32>
  %4 = mo.negative(%3) :  !mo.tensor<[1], f32>
  mo.output %4 : !mo.tensor<[1], f32>
}

// CHECK: has_compute_lambda = True
// CHECK-NEXT: has_store_lambda = False
// CHECK-LABEL: Running 'test_compute_out_split_dim':
// CHECK-NEXT: 'test_compute_out_split_dim' returned tensor<1x1xf32> [4.000000e+00]
mo.graph @test_compute_out_split_dim() -> !mo.tensor<[1, 1], f32> {
  %0 = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %1 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = mo.custom {symbol = "my_square_compute_out_fusion", device = #M.device_ref<"cpu", 0>}(%0) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  %3 = mo.split_dim[0](%2): (!mo.tensor<[1], f32>) -> !mo.tensor<[1, 1], f32>
  mo.output %3 : !mo.tensor<[1, 1], f32>
}


// CHECK: has_compute_lambda = True
// CHECK-NEXT: has_store_lambda = False
// CHECK-LABEL: Running 'test_compute_out_split_dim_fused':
// CHECK-NEXT: 'test_compute_out_split_dim_fused' returned tensor<1x1xf32> [-5.000000e+00]
mo.graph @test_compute_out_split_dim_fused() -> !mo.tensor<[1, 1], f32> {
  %0 = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %1 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>} : !mo.tensor<[1, 1], f32>
  %2 = mo.custom {symbol = "my_square_compute_out_fusion", device = #M.device_ref<"cpu", 0>}(%0) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  %3 = mo.split_dim[0](%2): (!mo.tensor<[1], f32>) -> !mo.tensor<[1, 1], f32>
  %4 = mo.add(%3, %1) :  !mo.tensor<[1, 1], f32>
  %5 = mo.negative(%4) :  !mo.tensor<[1, 1], f32>
  mo.output %5 : !mo.tensor<[1, 1], f32>
}

// Cast can't be fused with compute fusion so we generate the classic output lambda.
// CHECK: has_compute_lambda = False
// CHECK-NEXT: has_store_lambda = True
// CHECK-LABEL: Running 'test_compute_out_fusion_with_cast':
// CHECK-NEXT: 'test_compute_out_fusion_with_cast' returned tensor<1xsi32> [-5]
mo.graph @test_compute_out_fusion_with_cast() -> !mo.tensor<[1], si32> {
  %0 = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %1 = mo.constant {value = #M.dense_array<1.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
  %2 = mo.custom {symbol = "my_square_compute_out_fusion", device = #M.device_ref<"cpu", 0>}(%0) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], f32>
  %3 = mo.add(%2, %1) :  !mo.tensor<[1], f32>
  %4 = mo.cast(%3) : (!mo.tensor<[1], f32>) -> !mo.tensor<[1], si32>
  %5 = mo.negative(%4) :  !mo.tensor<[1], si32>
  mo.output %5 : !mo.tensor<[1], si32>
}
