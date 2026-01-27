// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" --allow-unregistered-dialect --mogg-fuse-elementwise --mogg-fuse-views= --mogg-fuse-prologues-epilogues --cse --canonicalize --mogg-simplify-kernel-operands --mlir-disable-threading | FileCheck %s --check-prefix=CHECK-IR



// COM: One single elementwise kernel, doesn't fuse anything
// CHECK-IR-LABEL: mo.graph @elementwise1
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'elementwise1':
// CHECK: 'elementwise1' returned tensor<2x3xf32> [4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]
mo.graph @elementwise1() -> !mo.tensor<[2, 3], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%0, %1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %2 : !mo.tensor<[2, 3], f32>
}

// COM: 3 elementwise kernels, should be fused in one
// CHECK-IR-LABEL: mo.graph @elementwise3
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}}, {{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'elementwise3':
// CHECK: 'elementwise3' returned tensor<2x3xf32> [1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.800000e+01, 2.000000e+01]
mo.graph @elementwise3() -> !mo.tensor<[2, 3], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %2 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %3 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%0, %1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %4 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%3, %2) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  %5 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%4, %4) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %5 : !mo.tensor<[2, 3], f32>
}

// COM: matmul kernel must be fused with add
// CHECK-IR-LABEL: mo.graph @matmul_add
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}}, {{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK: lambdas_have_fusion = True
// CHECK-LABEL: Running 'matmul_add':
// CHECK: 'matmul_add' returned tensor<2x2xf32> [6.500000e+00, 5.500000e+00, 1.350000e+01, 1.250000e+01]
mo.graph @matmul_add() -> !mo.tensor<[2, 2], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<5.000000e-01, 2.000000e+00, 2.500000e+00, 1.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %2 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %3 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "matmul_fuse_out"}(%0, %1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %4 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%3, %2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %4 : !mo.tensor<[2, 2], f32>
}

// broadcast and add fused (FuseBroadcast)
// CHECK-IR-LABEL: mo.graph @broadcast_add
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'broadcast_add':
// CHECK: 'broadcast_add' returned tensor<3x4xf32> [2.000000e+00, 3.500000e+00, 5.000000e+00, 6.500000e+00, 4.000000e+00, 5.500000e+00, 7.000000e+00, 8.500000e+00, 6.000000e+00, 7.500000e+00, 9.000000e+00, 1.050000e+01]
mo.graph @broadcast_add() -> !mo.tensor<[3, 4], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00, 6.000000e+00, 6.500000e+00> : tensor<12xf32>} : !mo.tensor<[3, 4], f32>
  %2 = mo.static.broadcast_to(%0) {device = #M.device_ref<"cpu", 0>} : !mo.tensor<[4], f32> -> !mo.tensor<[3, 4], f32>
  %3 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%2, %1) : (!mo.tensor<[3, 4], f32>, !mo.tensor<[3, 4], f32>) -> !mo.tensor<[3, 4], f32>
  mo.output %3 : !mo.tensor<[3, 4], f32>
}

// transpose and add fused (FuseBroadcast)
// CHECK-IR-LABEL: mo.graph @transpose_add
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}}, {{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'transpose_add':
// CHECK: 'transpose_add' returned tensor<3x4xf32> [1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00, 6.000000e+00, 6.500000e+00]
mo.graph @transpose_add(%arg0: !mo.tensor<[4, 3], f32>) -> !mo.tensor<[3, 4], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00, 3.000000e+00, 3.500000e+00, 4.000000e+00, 4.500000e+00, 5.000000e+00, 5.500000e+00, 6.000000e+00, 6.500000e+00> : tensor<3x4xf32>} : !mo.tensor<[3, 4], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %2 = mo.transpose(%arg0, %1) {device = #M.device_ref<"cpu", 0>} : (!mo.tensor<[4, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 4], f32>
  %3 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%2, %0) : (!mo.tensor<[3, 4], f32>, !mo.tensor<[3, 4], f32>) -> !mo.tensor<[3, 4], f32>
  mo.output %3 : !mo.tensor<[3, 4], f32>
}

// matmul (not fused)
// CHECK: lambdas_have_fusion = False
// CHECK-LABEL: Running 'matmul_no_fusion':
// CHECK: 'matmul_no_fusion' returned tensor<2x2xf32> [5.500000e+00, 4.000000e+00, 1.150000e+01, 1.000000e+01]
mo.graph @matmul_no_fusion() -> !mo.tensor<[2, 2], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<5.000000e-01, 2.000000e+00, 2.500000e+00, 1.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "matmul_fuse_out"}(%0, %1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %2 : !mo.tensor<[2, 2], f32>
}

// both add fused
// CHECK-IR-LABEL: mo.graph @elemwise_input_fusion
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}}, {{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'elemwise_input_fusion':
// CHECK: 'elemwise_input_fusion' returned tensor<4xf32> [3.500000e+00, 5.500000e+00, 7.500000e+00, 9.500000e+00]
mo.graph @elemwise_input_fusion() -> !mo.tensor<[4], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %2 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %3 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %4 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_lhs"}(%3, %2) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %4 : !mo.tensor<[4], f32>
}

// firsts add fused by FuseElementwise, then fused with last one with FuseInOutLambdas
// CHECK-IR-LABEL: mo.graph @elemwise_input_fusion_dbl
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}}, {{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'elemwise_input_fusion_dbl':
// CHECK: 'elemwise_input_fusion_dbl' returned tensor<4xf32> [6.500000e+00, 1.000000e+01, 1.350000e+01, 1.700000e+01]
mo.graph @elemwise_input_fusion_dbl() -> !mo.tensor<[4], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %2 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %3 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %4 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%3, %3) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %5 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_lhs"}(%4, %2) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %5 : !mo.tensor<[4], f32>
}

// both add_elementwise fused in add_fuse_inputs by FuseInOutLambdas
// CHECK-IR-LABEL: mo.graph @elemwise_input_fusion_multi_inputs
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}}, {{.+}}, {{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'elemwise_input_fusion_multi_inputs':
// CHECK: 'elemwise_input_fusion_multi_inputs' returned tensor<4xf32> [3.500000e+00, 5.500000e+00, 7.500000e+00, 9.500000e+00]
mo.graph @elemwise_input_fusion_multi_inputs() -> !mo.tensor<[4], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %2 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 5.000000e-01, 1.250000e+00, 1.500000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %3 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<-5.000000e-01, 5.000000e-01, 2.500000e-01, 5.000000e-01> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %4 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%2, %3) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %5 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %6 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_fuse_inputs"}(%4, %5) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %6 : !mo.tensor<[4], f32>
}

// add_elementwise and variadic_add fused by FuseInOutLambdas
// CHECK-IR-LABEL: mo.graph @variadic_fusion

// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}}, {{.+}}, {{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'variadic_fusion':
// CHECK: 'variadic_fusion' returned tensor<4xf32> [3.500000e+00, 5.500000e+00, 7.500000e+00, 9.500000e+00]
mo.graph @variadic_fusion() -> !mo.tensor<[4], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.500000e+00, 1.250000e+00, 1.000000e+00, 2.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<-5.000000e-01, 2.500000e-01, 1.000000e+00, 5.000000e-01> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %2 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %3 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %4 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %5 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "variadic_add"}(%4, %2, %3) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %5 : !mo.tensor<[4], f32>
}

// both add_elementwise and variadic_add fused by FuseInOutLambdas
// CHECK-IR-LABEL: mo.graph @variadic_fusion_multiple

// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}})
// COM: CHECK-IR: mo.output %[[OUT]]

// CHECK-LABEL: Running 'variadic_fusion_multiple':
// CHECK: 'variadic_fusion_multiple' returned tensor<4xf32> [3.500000e+00, 5.500000e+00, 7.500000e+00, 9.500000e+00]
mo.graph @variadic_fusion_multiple() -> !mo.tensor<[4], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.500000e+00, 1.250000e+00, 1.000000e+00, 2.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %1 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<-5.000000e-01, 2.500000e-01, 1.000000e+00, 5.000000e-01> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %2 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %3 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.500000e+00, -5.000000e-01, 1.000000e+00, 5.000000e-01> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %4 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<-1.000000e+00, 1.500000e+00, 5.000000e-01, 1.500000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %5 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %6 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "imposter_add_elementwise"}(%3, %4) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %7 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "variadic_add"}(%5, %2, %6) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %7 : !mo.tensor<[4], f32>
}

// Both ops fused by FuseViews
// CHECK-IR-LABEL: mo.graph @view_fusion_print_specs
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK: x.shape = [3, 2]
// CHECK: x.strides = [1, 3]
// CHECK: x.address_space = AddressSpace.GENERIC
// CHECK: x.exclusive = False
// CHECK-LABEL: Running 'view_fusion_print_specs':
// CHECK: 'view_fusion_print_specs' returned tensor<3x2xf32> [1.000000e+00, 4.000000e+00, 2.000000e+00, 5.000000e+00, 3.000000e+00, 6.000000e+00]
mo.graph @view_fusion_print_specs() -> !mo.tensor<[3, 2], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %1 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "transpose_2d"}(%0) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32>
  %2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "print_tensor_spec_fused"}(%1) : (!mo.tensor<[3, 2], f32>) -> !mo.tensor<[3, 2], f32>
  mo.output %2 : !mo.tensor<[3, 2], f32>
}

// All ops fuse together.
// The strides should be propagated through both transposes.
// CHECK-IR-LABEL: mo.graph @double_view_fusion_print_specs
// CHECK-IR: mo.constant
// CHECK-IR: %[[OUT:.*]] = mogg.kernel({{.+}})
// CHECK-IR: mo.output %[[OUT]]

// CHECK: x.shape = [2, 3]
// CHECK: x.strides = [3, 1]
// CHECK: x.address_space = AddressSpace.GENERIC
// CHECK: x.exclusive = False
// CHECK-LABEL: Running 'double_view_fusion_print_specs':
// CHECK: 'double_view_fusion_print_specs' returned tensor<2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]
mo.graph @double_view_fusion_print_specs() -> !mo.tensor<[2, 3], f32> {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00> : tensor<6xf32>} : !mo.tensor<[2, 3], f32>
  %1 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "transpose_2d"}(%0) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32>
  %2 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "transpose_2d"}(%1) : (!mo.tensor<[3, 2], f32>) -> !mo.tensor<[2, 3], f32>
  %3 = mo.custom {device = #M.device_ref<"cpu", 0>, symbol = "print_tensor_spec_fused"}(%2) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %3 : !mo.tensor<[2, 3], f32>
}
