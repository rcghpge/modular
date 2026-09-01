// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s -o %t.adv.mlir --gc-pipeline="enable-compute-lambdas=true constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.adv.mlir --result-output-style=full | FileCheck %s
// RUN: mo-opt %s -o %t.legacy.mlir --gc-pipeline="enable-compute-lambdas=true constant-fold=false kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.legacy.mlir --result-output-style=full | FileCheck %s

// `imposter_add_compute_fusion` (GraphCompiler/test/kernels/testing_kernels.mojo)
// declares its output as `_FusedComputeOutputTensor`, so fusing the trailing
// `mo.add` into it should pick a genuine compute lambda (not a classic store
// lambda) since dtypes match and compute lambdas are enabled above.
// CHECK: has_compute_lambda = True
// CHECK-LABEL: Running 'imposter_add_compute_fusion_add':
// CHECK-NEXT: 'imposter_add_compute_fusion_add' returned tensor<2x2xf32> [1.200000e+01, 2.200000e+01, 3.300000e+01, 4.500000e+01]
mo.graph @imposter_add_compute_fusion_add() -> (!mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %b = mo.constant {value = #M.dense_array<1.0, 0.0, 0.0, 1.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %bias = mo.constant {value = #M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %sum = mo.custom {symbol = "imposter_add_compute_fusion", device = #M.device_ref<"cpu", 0>}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  %out = mo.add(%sum, %bias) : !mo.tensor<[2, 2], f32>
  mo.output %out : !mo.tensor<[2, 2], f32>
}
