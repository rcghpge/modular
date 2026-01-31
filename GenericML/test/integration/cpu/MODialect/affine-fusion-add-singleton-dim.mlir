// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones %t.mlir | FileCheck %s

// CHECK-LABEL: 'simple_add_singleton_dim_rank3_insert1_middle' returned tensor<2x1x3xf32>
// CHECK-SAME: 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01
mo.graph @simple_add_singleton_dim_rank3_insert1_middle(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
    %bias = "mo.constant"() {value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>} : () -> !mo.tensor<[2, 1, 3], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[2, 10], f32>, !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3], f32>
    %out = mo.add_singleton_dim[1](%res): (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[2, 1, 3], f32>
    mo.output %final: !mo.tensor<[2, 1, 3], f32>
}

// CHECK-LABEL: 'simple_add_singleton_dim_rank3_insert1_start' returned tensor<1x2x3xf32>
// CHECK-SAME: 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01
mo.graph @simple_add_singleton_dim_rank3_insert1_start(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[10, 3], f32>) -> !mo.tensor<[1, 2, 3], f32> {
    %bias = "mo.constant"() {value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>} : () -> !mo.tensor<[1, 2, 3], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[2, 10], f32>, !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3], f32>
    %out = mo.add_singleton_dim[0](%res): (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[1, 2, 3], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[1, 2, 3], f32>
    mo.output %final: !mo.tensor<[1, 2, 3], f32>
}

// CHECK-LABEL: 'simple_add_singleton_dim_rank3_insert1_end' returned tensor<2x3x1xf32>
// CHECK-SAME: 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01
mo.graph @simple_add_singleton_dim_rank3_insert1_end(%arg0: !mo.tensor<[2, 10], f32>, %arg1: !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3, 1], f32> {
    %bias = "mo.constant"() {value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<6xf32>} : () -> !mo.tensor<[2, 3, 1], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[2, 10], f32>, !mo.tensor<[10, 3], f32>) -> !mo.tensor<[2, 3], f32>
    %out = mo.add_singleton_dim[2](%res): (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3, 1], f32>
    %final = mo.add(%out, %bias): !mo.tensor<[2, 3, 1], f32>
    mo.output %final: !mo.tensor<[2, 3, 1], f32>
}
