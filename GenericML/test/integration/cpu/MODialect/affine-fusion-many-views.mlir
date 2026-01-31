// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones %t.mlir | FileCheck %s

// CHECK-LABEL: 'crazy_views' returned tensor<1x1x8x1x4x1xf32>
// CHECK-SAME: 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01
// CHECK-SAME: 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01
// CHECK-SAME: 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01
// CHECK-SAME: 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01, 4.000000e+01, 4.100000e+01, 4.200000e+01
mo.graph @crazy_views(%arg0: !mo.tensor<[4, 10], f32>, %arg1: !mo.tensor<[10, 8], f32>) -> !mo.tensor<[1, 1, 8, 1, 4, 1], f32> {
    %bias = "mo.constant"() {value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
       14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
       27., 28., 29., 30., 31., 32.> : tensor<32xf32>} : () -> !mo.tensor<[1, 1, 8, 1, 4, 1], f32>
    %res = mo.matmul(%arg0, %arg1): (!mo.tensor<[4, 10], f32>, !mo.tensor<[10, 8], f32>) -> !mo.tensor<[4, 8], f32>

    %view1 = mo.add_singleton_dim[0](%res): (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[1, 4, 8], f32>
    %view2 = mo.add_singleton_dim[3](%view1): (!mo.tensor<[1, 4, 8], f32>) -> !mo.tensor<[1, 4, 8, 1], f32>
    %view3 = mo.split_dim[2](%view2): (!mo.tensor<[1, 4, 8, 1], f32>) -> !mo.tensor<[1, 4, 2, 4, 1], f32>
    %view4 = mo.add_singleton_dim[1](%view3): (!mo.tensor<[1, 4, 2, 4, 1], f32>) -> !mo.tensor<[1, 1, 4, 2, 4, 1], f32>
    %view5 = mo.merge_dim[2](%view4): (!mo.tensor<[1, 1, 4, 2, 4, 1], f32>) -> !mo.tensor<[1, 1, 8, 4, 1], f32>
    %view6 = mo.add_singleton_dim[3](%view5): (!mo.tensor<[1, 1, 8, 4, 1], f32>) -> !mo.tensor<[1, 1, 8, 1, 4, 1], f32>

    %final = mo.add(%view6, %bias): !mo.tensor<[1, 1, 8, 1, 4, 1], f32>
    mo.output %final: !mo.tensor<[1, 1, 8, 1, 4, 1], f32>
}
