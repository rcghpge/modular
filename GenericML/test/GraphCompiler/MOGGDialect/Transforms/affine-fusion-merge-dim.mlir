// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" -o %t.mlir --mlir-print-ir-after=fuse-prologues-epilogues  &> %t.debug_ir.mlir


// This runs a heuristic check that all operations are fused together (and so affine fusion works).
// RUN: cat %t.debug_ir.mlir | FileCheck %s --check-prefix=CHECK-IR

// The kernel summary string should reflect an accurate fusion record.
// DAG because the order things are printed in the middle of the pipeline is
// not predictable.
// CHECK-IR-DAG: mo.graph @simple_merge_dim_rank2_merge_0
// CHECK-IR-DAG: summary = "mo.matmul:{{[0-9]+}}+mo.merge_dim:{{[0-9]+}}+mo.add:{{[0-9]+}}"

// CHECK-LABEL: 'simple_merge_dim_rank2_merge_0' returned tensor<32xf32>
// CHECK-SAME: 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01
// CHECK-SAME: 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01
// CHECK-SAME: 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01
// CHECK-SAME: 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01, 4.000000e+01, 4.100000e+01, 4.200000e+01
mo.graph @simple_merge_dim_rank2_merge_0(%arg0: !mo.tensor<[4, 10], f32>, %arg1: !mo.tensor<[10, 8], f32>) -> !mo.tensor<[32], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
       14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
       27., 28., 29., 30., 31., 32.> : tensor<32xf32>} : () -> !mo.tensor<[32], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[4, 10], f32>, !mo.tensor<[10, 8], f32>) -> !mo.tensor<[4, 8], f32>
    %out = mo.merge_dim[0](%res): (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[32], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[32], f32>
    mo.output %final: !mo.tensor<[32], f32>
}

// CHECK-IR-DAG: mo.graph @conv_merge_dim_rank3_merge_1
// CHECK-IR-DAG: summary = "mo.conv:{{[0-9]+}}+mo.merge_dim:{{[0-9]+}}+mo.add:{{[0-9]+}}"

// CHECK-LABEL: 'conv_merge_dim_rank3_merge_1' returned tensor<1x4x3xf32>
// CHECK-SAME: 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
// CHECK-SAME: 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01
mo.graph @conv_merge_dim_rank3_merge_1(%input : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 4, 3], f32>) {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.> : tensor<12xf32>} : () -> !mo.tensor<[1, 4, 3], f32>
    %st = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %di = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %pa = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
    %ng = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
    %filter = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.0, 1.0, 1.0> : tensor<1x1x1x3xf32>} : () -> !mo.tensor<[1, 1, 1, 3], f32>
    %0 = mo.add(%filter, %filter) : !mo.tensor<[1, 1, 1, 3], f32>
    %conv = mo.conv(%input, %0) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[1, 2, 2, 3], f32>, !mo.tensor<[1, 1, 1, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
    ) -> !mo.tensor<[1, 2, 2, 3], f32>
    %merged = mo.merge_dim[1](%conv): (!mo.tensor<[1, 2, 2, 3], f32>) -> !mo.tensor<[1, 4, 3], f32>
    %final = mo.add(%merged, %bias): !mo.tensor<[1, 4, 3], f32>
    mo.output %final : !mo.tensor<[1, 4, 3], f32>
}

// CHECK-IR-DAG: mo.graph @conv_merge_dim_rank3_merge_2
// CHECK-IR-DAG: summary = "mo.conv:{{[0-9]+}}+mo.merge_dim:{{[0-9]+}}+mo.add:{{[0-9]+}}"

// CHECK-LABEL: 'conv_merge_dim_rank3_merge_2' returned tensor<1x2x6xf32>
// CHECK-SAME: 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
// CHECK-SAME: 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01
mo.graph @conv_merge_dim_rank3_merge_2(%input : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 2, 6], f32>) {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.> : tensor<12xf32>} : () -> !mo.tensor<[1, 2, 6], f32>
    %st = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %di = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : () -> !mo.tensor<[2], si64>
    %pa = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : () -> !mo.tensor<[4], si64>
    %ng = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3> : tensor<si64>} : () -> !mo.tensor<[], si64>
    %filter = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.0, 1.0, 1.0> : tensor<1x1x1x3xf32>} : () -> !mo.tensor<[1, 1, 1, 3], f32>
    %0 = mo.add(%filter, %filter) : !mo.tensor<[1, 1, 1, 3], f32>
    %conv = mo.conv(%input, %0) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[1, 2, 2, 3], f32>, !mo.tensor<[1, 1, 1, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
    ) -> !mo.tensor<[1, 2, 2, 3], f32>
    %merged = mo.merge_dim[2](%conv): (!mo.tensor<[1, 2, 2, 3], f32>) -> !mo.tensor<[1, 2, 6], f32>
    %final = mo.add(%merged, %bias): !mo.tensor<[1, 2, 6], f32>
    mo.output %final : !mo.tensor<[1, 2, 6], f32>
}
