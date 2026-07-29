// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.adv.mlir
// RUN: mt --execute --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths="%kernel_api_test_lib"" -o %t.legacy.mlir
// RUN: mt --execute --result-output-style=full %t.legacy.mlir | FileCheck %s

// End-to-end execution of the advanced-fusion MAP-ABI path for multi-output
// (two plain `OutputTensor`s, two `foreach`) opaque kernels. Same dual-run
// equivalence proof as `adv-fusion-map-execute.mlir`: both the adv (env-gated)
// and legacy runs must produce identical output against these patterns.
// Per-output lines are matched loosely (substring) so they hold regardless of
// the full-result-style separator printed between the two returned tensors.

// 1 input / 2 outputs: `res0 = a + 1`, `res1 = a * 2`.
// CHECK-LABEL: Running 'two_outputs':
// CHECK: 'two_outputs' returned
// CHECK: tensor<2x2xf32> [2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]
// CHECK: tensor<2x2xf32> [2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00]
mo.graph @two_outputs() -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %r0, %r1 = mo.custom {symbol = "map_lower_two_outputs", device = #M.device_ref<"cpu", 0>}(%a) : (!mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %r0, %r1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// 2 inputs / 2 outputs: `res0 = a + b`, `res1 = a - b`.
// CHECK-LABEL: Running 'add_sub':
// CHECK: 'add_sub' returned
// CHECK: tensor<2x2xf32> [1.100000e+01, 2.200000e+01, 3.300000e+01, 4.400000e+01]
// CHECK: tensor<2x2xf32> [9.000000e+00, 1.800000e+01, 2.700000e+01, 3.600000e+01]
mo.graph @add_sub() -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<10.0, 20.0, 30.0, 40.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %b = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[2, 2], f32>
  %r0, %r1 = mo.custom {symbol = "map_lower_add_sub", device = #M.device_ref<"cpu", 0>}(%a, %b) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>)
  mo.output %r0, %r1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
