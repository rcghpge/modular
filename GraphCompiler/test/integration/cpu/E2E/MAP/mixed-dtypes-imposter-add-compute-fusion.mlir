// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s -o %t.adv.mlir --gc-pipeline="enable-compute-lambdas=true constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.adv.mlir --result-output-style=full | FileCheck %s
// RUN: mo-opt %s -o %t.legacy.mlir --gc-pipeline="enable-compute-lambdas=true constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.legacy.mlir --result-output-style=full | FileCheck %s

// Same fusion shape as imposter-add-compute-fusion.mlir, but the trailing
// `mo.cast` makes the fused epilogue mixed-precision (f32 in, si32 out). A
// compute lambda returns its value to the kernel instead of storing it, so it
// cannot change dtype -- the classifier falls back to the classic store lambda
// (`OutputLegacyForCompute`) on both pipelines even though
// `imposter_add_compute_fusion`'s output arg is `_FusedComputeOutputTensor`
// and compute lambdas are enabled.
// CHECK: has_compute_lambda = False
// CHECK-LABEL: Running 'mixed_dtypes_imposter_add_compute_fusion_add_cast':
// CHECK-NEXT: 'mixed_dtypes_imposter_add_compute_fusion_add_cast' returned tensor<2x2xsi32> [-12, -22, -33, -45]
mo.graph @mixed_dtypes_imposter_add_compute_fusion_add_cast() -> (!mo.tensor<[2, 2], si32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %b = mo.constant {value = #M.dense_array<1.0, 0.0, 0.0, 1.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %bias = mo.constant {value = #M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %sum = mo.custom {symbol = "imposter_add_compute_fusion", device = #M.device_ref<"cpu", 0>}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %added = mo.add(%sum, %bias) : !mo.tensor<[2, 2], f32>
  %cast = mo.cast(%added) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], si32>
  %out = mo.negative(%cast) : !mo.tensor<[2, 2], si32>
  mo.output %out : !mo.tensor<[2, 2], si32>
}
