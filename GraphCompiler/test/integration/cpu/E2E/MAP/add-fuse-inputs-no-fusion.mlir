// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.adv.mlir
// RUN: mt --execute --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.legacy.mlir
// RUN: mt --execute --result-output-style=full %t.legacy.mlir | FileCheck %s

// Baseline for add-fuse-inputs-prologue-fusion.mlir: the same kernel with both
// `FusedInputTensor` operands fed directly by constants, so there is nothing to
// fuse. `MOToMAP` still builds a `map.iter.prologue` for every fusible input,
// so this pins the pass-through lowering of one (the callback just loads the
// captured tensor) independently of any actual fusion.

// CHECK-LABEL: Running 'add_fuse_inputs_no_fusion':
// CHECK: 'add_fuse_inputs_no_fusion' returned tensor<4xf32> [1.500000e+00, 2.500000e+00, 3.500000e+00, 4.500000e+00]
mo.graph @add_fuse_inputs_no_fusion() -> !mo.tensor<[4], f32> {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %1 = mo.constant {value = #M.dense_array<5.000000e-01, 1.000000e+00, 1.500000e+00, 2.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %2 = mo.custom {symbol = "imposter_add_fuse_inputs", device = #M.device_ref<"cpu", 0>}(%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %2 : !mo.tensor<[4], f32>
}
