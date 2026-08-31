// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.adv.mlir
// RUN: mt --execute --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.legacy.mlir
// RUN: mt --execute --result-output-style=full %t.legacy.mlir | FileCheck %s

// End-to-end coverage for a dtype-changing `mo.cast` fused into a TRAIT-lane
// kernel's epilogue: `EpilogueFuser` retypes the store path to si32 while the
// epilogue's incoming value stays f32, and `TensorCreateEpilogueOp::emitMojo`
// rebinds that incoming value against the INPUT dtype rather than the output's
// (a single `rebind` cannot bridge f32 -> si32). The si32 result below is what
// proves the retype and the emit-side rebind agree.

// CHECK-LABEL: Running 'add_fuse_output_add_cast':
// CHECK: 'add_fuse_output_add_cast' returned tensor<2x2xsi32> [7, 9, 11, 13]
mo.graph @add_fuse_output_add_cast() -> !mo.tensor<[2, 2], si32> {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %1 = mo.constant {value = #M.dense_array<5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %2 = mo.constant {value = #M.dense_array<1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %3 = mo.custom {symbol = "imposter_add_fuse_output", device = #M.device_ref<"cpu", 0>}(%0, %1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %4 = mo.custom {symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>}(%3, %2) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %5 = mo.cast(%4) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], si32>
  mo.output %5 : !mo.tensor<[2, 2], si32>
}
