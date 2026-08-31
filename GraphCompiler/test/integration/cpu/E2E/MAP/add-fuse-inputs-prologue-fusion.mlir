// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.adv.mlir
// RUN: mt --execute --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.legacy.mlir
// RUN: mt --execute --result-output-style=full %t.legacy.mlir | FileCheck %s

// Prologue (input-load) fusion counterpart of matmul-add-epilogue-fusion.mlir.
// Both `FusedInputTensor` operands of `imposter_add_fuse_inputs` are fed by an
// `imposter_add_elementwise` producer, so both producers are fused into its
// load path: the advanced-fusion path splices each into a `map.iter.prologue`,
// which lowers to a `mogg._tensor.create.prologue` and emits an `InputFusion`
// struct bound onto the operand via `_bind_to_fused_input`. Producers yield
// [0.5, 1, 1.5, 2] and [3, 4.5, 6, 7.5], which the kernel sums.

// CHECK-LABEL: Running 'add_fuse_inputs_prologue':
// CHECK: 'add_fuse_inputs_prologue' returned tensor<4xf32> [3.500000e+00, 5.500000e+00, 7.500000e+00, 9.500000e+00]
mo.graph @add_fuse_inputs_prologue() -> !mo.tensor<[4], f32> {
  %0 = mo.constant {value = #M.dense_array<1.000000e+00, 1.500000e+00, 2.000000e+00, 2.500000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %1 = mo.constant {value = #M.dense_array<2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %2 = mo.constant {value = #M.dense_array<1.000000e+00, 5.000000e-01, 1.250000e+00, 1.500000e+00> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %3 = mo.constant {value = #M.dense_array<-5.000000e-01, 5.000000e-01, 2.500000e-01, 5.000000e-01> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %4 = mo.custom {symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>}(%2, %3) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %5 = mo.custom {symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>}(%0, %1) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  %6 = mo.custom {symbol = "imposter_add_fuse_inputs", device = #M.device_ref<"cpu", 0>}(%4, %5) : (!mo.tensor<[4], f32>, !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %6 : !mo.tensor<[4], f32>
}
