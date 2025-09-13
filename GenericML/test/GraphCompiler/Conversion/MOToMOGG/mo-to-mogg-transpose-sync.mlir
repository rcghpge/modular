// RUN: mo-opt %s --mo-to-mogg="kernel-library-paths=%kernel_api_test_lib" | FileCheck %s

// Verify mo.transpose is launched synchronously (isAsync = false) until the
// model init high water mark is enforced (GEX-2569).

mo.graph @transpose_sync(%arg0: !mo.tensor<[4, 3], f32>) -> (!mo.tensor<[3, 4], f32>) {
  %perm = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %out = mo.transpose(%arg0, %perm) : (!mo.tensor<[4, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 4], f32>
  mo.output %out : !mo.tensor<[3, 4], f32>
}

// CHECK-LABEL: mo.graph @transpose_sync
// CHECK: %[[KERNEL_OUT:.*]] = mogg.experimental.kernel(
// CHECK: mogg.call.view["mo.transpose"]
// CHECK: isAsync = false
// CHECK: summary = "mo.transpose
// CHECK: mo.output %[[KERNEL_OUT]] : !mo.tensor<[3, 4], f32>
