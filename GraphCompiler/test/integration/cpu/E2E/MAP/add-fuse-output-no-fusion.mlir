// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.adv.mlir
// RUN: mt --execute --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.legacy.mlir
// RUN: mt --execute --result-output-style=full %t.legacy.mlir | FileCheck %s

// Isolates TensorCreateEpilogueOp::emitMojo (GraphCompiler/lib/MOGGDialect/
// MOGGOps.cpp) from any fusion-pass logic: calls the fusable kernel alone,
// with zero fusion candidates present, so `map.iter.opaque` still routes its
// output through the epilogue machinery even though nothing gets absorbed
// into it. See add-fuse-output-epilogue-fusion.mlir for the fused case.

// CHECK-LABEL: Running 'add_fuse_output_no_fusion':
// CHECK: 'add_fuse_output_no_fusion' returned tensor<2x2xf32> [6.000000e+00, 8.000000e+00, 1.000000e+01, 1.200000e+01]
mo.graph @add_fuse_output_no_fusion() -> !mo.tensor<[2, 2], f32> {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %1 = mo.constant {value = #M.dense_array<5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %2 = mo.custom {symbol = "imposter_add_fuse_output", device = #M.device_ref<"cpu", 0>}(%0, %1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %2 : !mo.tensor<[2, 2], f32>
}
