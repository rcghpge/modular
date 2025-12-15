// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" | FileCheck %s

// Verify mo.transpose is launched synchronously (isAsync = false) on both CPU
// and GPU until the model init high water mark is enforced (GEX-2569).

mo.graph @transpose_sync_cpu(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[3, 4], f32>) {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %out = mo.transpose(%arg0, %perm) : (!mo.tensor<[4, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 4], f32>
  mo.output %out : !mo.tensor<[3, 4], f32>
}

// CHECK-LABEL: mo.graph @transpose_sync_cpu
// CHECK: %[[KERNEL_OUT:.*]] = mogg.experimental.kernel(
// CHECK: mogg.call.view["mo.transpose"]
// CHECK: isAsync = false
// CHECK: summary = "mo.transpose
// CHECK: mo.output %[[KERNEL_OUT]] : !mo.tensor<[3, 4], f32>

mo.graph @transpose_sync_gpu(%arg0: !mo.tensor<[4, 3], f32, gpu:0>) -> (!mo.tensor<[3, 4], f32, gpu:0>) {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %out = mo.transpose(%arg0, %perm) : (!mo.tensor<[4, 3], f32, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 4], f32, gpu:0>
  mo.output %out : !mo.tensor<[3, 4], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @transpose_sync_gpu
// CHECK: mogg.experimental.kernel(
// CHECK: mogg.call.view["mo.transpose"]
// CHECK: device = #M.device_ref<"gpu", 0>
// CHECK: isAsync = false
// CHECK: summary = "mo.transpose
