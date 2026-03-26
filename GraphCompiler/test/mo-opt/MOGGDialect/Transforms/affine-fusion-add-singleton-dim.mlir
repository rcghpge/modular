// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir --mlir-print-ir-after=mogg-fuse-epilogues  &> %t.debug_ir.mlir


// This runs a heuristic check that all operations are fused together (and so affine fusion works).
// RUN: cat %t.debug_ir.mlir | FileCheck %s --check-prefix=CHECK-IR

// The kernel summary string should reflect an accurate fusion record.
// DAG because the order things are printed in the middle of the pipeline is
// not predictable.
// CHECK-IR-DAG: mo.graph @simple_add_singleton_dim_rank3_insert1_middle
// CHECK-IR-DAG: fusionInfo = Epilogue(Affine("mo.matmul":{{[0-9]+}}, "mo.add_singleton_dim":{{[0-9]+}}), "mo.add":{{[0-9]+}}, operandIndices=[0])

// CHECK-LABEL: 'simple_add_singleton_dim_rank3_insert1_middle' returned tensor<2x1x3xf32>
// CHECK-SAME: 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01
mo.graph @simple_add_singleton_dim_rank3_insert1_middle(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>} : () -> !mo.tensor<[2, 1, 3], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[2, 10], f32>, !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3], f32>
    %out = mo.add_singleton_dim[1](%res): (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[2, 1, 3], f32>
    mo.output %final: !mo.tensor<[2, 1, 3], f32>
}

// CHECK-IR-DAG: mo.graph @simple_add_singleton_dim_rank3_insert1_start
// CHECK-IR-DAG: fusionInfo = Epilogue(Affine("mo.matmul":{{[0-9]+}}, "mo.add_singleton_dim":{{[0-9]+}}), "mo.add":{{[0-9]+}}, operandIndices=[0])

// CHECK-LABEL: 'simple_add_singleton_dim_rank3_insert1_start' returned tensor<1x2x3xf32>
// CHECK-SAME: 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01
mo.graph @simple_add_singleton_dim_rank3_insert1_start(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[10, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>} : () -> !mo.tensor<[1, 2, 3], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[2, 10], f32>, !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3], f32>
    %out = mo.add_singleton_dim[0](%res): (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[1, 2, 3], f32>
    mo.output %final: !mo.tensor<[1, 2, 3], f32>
}

// CHECK-IR-DAG: mo.graph @simple_add_singleton_dim_rank3_insert1_end
// CHECK-IR-DAG: fusionInfo = Epilogue(Affine("mo.matmul":{{[0-9]+}}, "mo.add_singleton_dim":{{[0-9]+}}), "mo.add":{{[0-9]+}}, operandIndices=[0])

// CHECK-LABEL: 'simple_add_singleton_dim_rank3_insert1_end' returned tensor<2x3x1xf32>
// CHECK-SAME: 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01
mo.graph @simple_add_singleton_dim_rank3_insert1_end(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3, 1], f32> {
    %bias = "mo.constant"() {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>} : () -> !mo.tensor<[2, 3, 1], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[2, 10], f32>, !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3], f32>
    %out = mo.add_singleton_dim[2](%res): (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3, 1], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[2, 3, 1], f32>
    mo.output %final: !mo.tensor<[2, 3, 1], f32>
}
