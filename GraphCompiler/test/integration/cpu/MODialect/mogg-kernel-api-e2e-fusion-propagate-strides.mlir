// RUN: mo-opt %s -o %t.mlir --gc-pipeline="constant-fold=false  mogg-debug-dump-fused-tensor-specs=true kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir | FileCheck %s

// Unit tests to ensure that during ViewFusion the static strides and the
// static alignment are propagated correctly.

// CHECK-LABEL: Running 'broadcast_to':
// CHECK: input #0: shape = (5, 4, 3) strides = (0, 1, 0) alignment = 64
mo.graph @broadcast_to(%x: !mo.tensor<[4, 1], f32>) -> (!mo.tensor<[5, 4, 3], f32>) {
    %0 = mo.static.broadcast_to(%x) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[5, 4, 3], f32>
    %1 = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%0) : (!mo.tensor<[5, 4, 3], f32>) -> !mo.tensor<[5, 4, 3], f32>
    mo.output %1 : !mo.tensor<[5, 4, 3], f32>
}


// CHECK-LABEL: Running 'reshape':
// CHECK: input #0: shape = (3, 4, 2) strides = (8, 2, 1) alignment = 64
mo.graph @reshape(%x: !mo.tensor<[24], f32>) -> (!mo.tensor<[3, 4, 2], f32>) {
  %0 = mo.static.reshape(%x) : !mo.tensor<[24], f32> -> !mo.tensor<[3, 4, 2], f32>
  %1 = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%0) : (!mo.tensor<[3, 4, 2], f32>) -> !mo.tensor<[3, 4, 2], f32>
  mo.output %1 : !mo.tensor<[3, 4, 2], f32>
}

// CHECK-LABEL: Running 'transpose':
// CHECK: input #0: shape = (3, 4, 2) strides = (4, 1, 12) alignment = 64
mo.graph @transpose(%x: !mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[3, 4, 2], f32>) {
  %perm = mo.constant {value = #M.dense_array<1, 2, 0> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %0 = mo.transpose(%x, %perm) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[3, 4, 2], f32>
  %1 = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%0) : (!mo.tensor<[3, 4, 2], f32>) -> !mo.tensor<[3, 4, 2], f32>
  mo.output %1 : !mo.tensor<[3, 4, 2], f32>
}

// CHECK-LABEL: Running 'slice_dim_pos_step':
// CHECK: input #0: shape = (2, 3) strides = (5, 2) alignment = 4
mo.graph @slice_dim_pos_step(%x: !mo.tensor<[3, 5], f32>) -> !mo.tensor<[2, 3], f32> {
  %start = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<3, 5> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%x, %start, %stop, %step) : (
    !mo.tensor<[3, 5], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 3], f32>
  %out = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%res) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %out : !mo.tensor<[2, 3], f32>
}

// CHECK-LABEL: Running 'slice_dim_neg_step':
// CHECK: input #0: shape = (2, 2) strides = (-6, -1) alignment = 1
mo.graph @slice_dim_neg_step(%x: !mo.tensor<[5, 3], f32>) -> !mo.tensor<[2, 2], f32> {
  %start = mo.constant {value = #M.dense_array<-1, -1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<-1, -1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<-2, -1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%x, %start, %stop, %step) : (
    !mo.tensor<[5, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[2, 2], f32>
  %out = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%res) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %out : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: Running 'reshape_and_transpose':
// CHECK: input #0: shape = (3, 4, 2) strides = (4, 1, 12) alignment = 64
mo.graph @reshape_and_transpose(%x: !mo.tensor<[24], f32>) -> (!mo.tensor<[3, 4, 2], f32>) {
  %0 = mo.static.reshape(%x) : !mo.tensor<[24], f32> -> !mo.tensor<[2, 3, 4], f32>
  %perm = mo.constant {value = #M.dense_array<1, 2, 0> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %1 = mo.transpose(%0, %perm) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[3, 4, 2], f32>
  %2 = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%1) : (!mo.tensor<[3, 4, 2], f32>) -> !mo.tensor<[3, 4, 2], f32>
  mo.output %2 : !mo.tensor<[3, 4, 2], f32>
}

// Every start offset is zero, so only the source's 40-byte row pitch bounds
// what a consumer may assume. Regression test for GEX-4058.
// CHECK-LABEL: Running 'slice_column_zero_start':
// CHECK: input #0: shape = (4, 8) strides = (10, 1) alignment = 8
mo.graph @slice_column_zero_start(%x: !mo.tensor<[4, 10], f32>) -> !mo.tensor<[4, 8], f32> {
  %start = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %stop = mo.constant {value = #M.dense_array<4, 8> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res = mo.slice(%x, %start, %stop, %step) : (
    !mo.tensor<[4, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[4, 8], f32>
  %out = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%res) : (!mo.tensor<[4, 8], f32>) -> !mo.tensor<[4, 8], f32>
  mo.output %out : !mo.tensor<[4, 8], f32>
}
