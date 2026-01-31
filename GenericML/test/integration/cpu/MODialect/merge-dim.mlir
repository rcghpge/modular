// RUN: mo-opt %s --mo-to-mgp="constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones %t.mlir | FileCheck %s

// CHECK-LABEL: 'trivial_merge_dim' returned tensor<1x2x16x1xf32>
// CHECK-SAME: 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00
// CHECK-SAME: 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01
// CHECK-SAME: 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01
// CHECK-SAME: 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01
mo.graph @trivial_merge_dim() -> !mo.tensor<[1,2,16,1], f32> {
    %bias = "mo.constant"() {value = #M.dense_array<1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.,
       14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26.,
       27., 28., 29., 30., 31., 32.> : tensor<32xf32>} : () -> !mo.tensor<[1,2,2,8,1], f32>
    %out = mo.merge_dim[2](%bias): (!mo.tensor<[1,2,2,8,1], f32>) -> !mo.tensor<[1,2,16,1], f32>
    mo.output %out: !mo.tensor<[1,2,16,1], f32>
}
