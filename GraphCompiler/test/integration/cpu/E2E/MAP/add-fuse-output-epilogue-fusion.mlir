// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.adv.mlir
// RUN: mt --execute --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.legacy.mlir
// RUN: mt --execute --result-output-style=full %t.legacy.mlir | FileCheck %s

// End-to-end coverage for TensorCreateEpilogueOp::emitMojo
// (GraphCompiler/lib/MOGGDialect/MOGGOps.cpp) with a real (non-empty) kernel
// body and an elementwise op actually fused into its epilogue. See
// add-fuse-output-no-fusion.mlir for the same kernel with nothing fused in.

// CHECK-LABEL: Running 'add_fuse_output_add':
// CHECK: 'add_fuse_output_add' returned tensor<2x2xf32> [7.000000e+00, 9.000000e+00, 1.100000e+01, 1.300000e+01]
mo.graph @add_fuse_output_add() -> !mo.tensor<[2, 2], f32> {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %1 = mo.constant {value = #M.dense_array<5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %2 = mo.constant {value = #M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %3 = mo.custom {symbol = "imposter_add_fuse_output", device = #M.device_ref<"cpu", 0>}(%0, %1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %4 = mo.custom {symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>}(%3, %2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %4 : !mo.tensor<[2, 2], f32>
}
