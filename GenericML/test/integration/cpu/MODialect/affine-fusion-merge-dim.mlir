// RUN: mo-opt %s --mo-to-mgp="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.mlir
// RUN: mt --execute --result-output-style=full --model-inputs=ones %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'merge_dim_rank3_merge_1':
// We should get the correct shape / strides before merge_dim.
// CHECK-NEXT: output._static_shape = [1, 2, 2, 3]
// CHECK-NEXT: output._static_strides = [12, 6, 3, 1]
// CHECK-NEXT: output._runtime_shape = (1, 2, 2, 3)
// CHECK-NEXT: output._runtime_strides = (12, 6, 3, 1)
// CHECK-NEXT: tensor<1x4x3xf32> [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]

mo.graph @merge_dim_rank3_merge_1(%0 : !mo.tensor<[1, 2, 2, 3], f32>) -> (!mo.tensor<[1, 4, 3], f32>) {
    %1 = mo.custom {symbol = "print_output_shape_strides_fused", device = #M.device_ref<"cpu", 0>}(%0) : (!mo.tensor<[1, 2, 2, 3], f32>) -> !mo.tensor<[1, 2, 2, 3], f32>
    %2 = mo.merge_dim[1](%1): (!mo.tensor<[1, 2, 2, 3], f32>) -> !mo.tensor<[1, 4, 3], f32>
    mo.output %2 : !mo.tensor<[1, 4, 3], f32>
}
