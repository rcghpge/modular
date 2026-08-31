// RUN: env MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.adv.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x2 --input-shapes=2x2 --input-shapes=2x2 --result-output-style=full %t.adv.mlir | FileCheck %s
// RUN: mo-opt %s --gc-pipeline="constant-fold=false kernel-library-paths=%kernel_api_test_pkg" -o %t.legacy.mlir
// RUN: mt --execute --model-inputs=ones --input-shapes=2x2 --input-shapes=2x2 --input-shapes=2x2 --result-output-style=full %t.legacy.mlir | FileCheck %s

// Same coverage as add-fuse-output-epilogue-fusion.mlir, with dynamic shapes
// to confirm TensorCreateEpilogueOp::emitMojo also handles the runtime
// (non-compile-time-constant) index path correctly.

// CHECK-LABEL: Running 'add_fuse_output_add_dynamic':
// CHECK: 'add_fuse_output_add_dynamic' returned tensor<2x2xf32> [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]
mo.graph @add_fuse_output_add_dynamic<M, N>(%a: !mo.tensor<[M, N], f32>, %b: !mo.tensor<[M, N], f32>, %bias: !mo.tensor<[M, N], f32>) -> (!mo.tensor<[M, N], f32>) {
  %sum = mo.custom {symbol = "imposter_add_fuse_output", device = #M.device_ref<"cpu", 0>}(%a, %b) : (!mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32>
  %out = mo.custom {symbol = "imposter_add_elementwise", device = #M.device_ref<"cpu", 0>}(%sum, %bias) : (!mo.tensor<[M, N], f32>, !mo.tensor<[M, N], f32>) -> !mo.tensor<[M, N], f32>
  mo.output %out : !mo.tensor<[M, N], f32>
}
