// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" | FileCheck %s --check-prefix=ASYNC
// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib force-sync=true" | FileCheck %s --check-prefix=SYNC

// Verify that GPU kernels lower to async by default (to avoid blocking), but
// can be forced synchronous via --force-sync for debugging.

mo.graph @gpu_mul(%a: !mo.tensor<[2], f32, gpu:0>, %b: !mo.tensor<[2], f32, gpu:0>) -> !mo.tensor<[2], f32, gpu:0> {
  %0 = mo.mul(%a, %b) : !mo.tensor<[2], f32, gpu:0>
  mo.output %0 : !mo.tensor<[2], f32, gpu:0>
}

// ASYNC-LABEL: mo.graph @gpu_mul
// ASYNC: mogg.experimental.kernel
// ASYNC: device = #M.device_ref<"gpu", 0>
// ASYNC: isAsync = true

// SYNC-LABEL: mo.graph @gpu_mul
// SYNC: mogg.experimental.kernel
// SYNC: device = #M.device_ref<"gpu", 0>
// SYNC: isAsync = false
