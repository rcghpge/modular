// RUN: mo-opt %s -o %t.mlir --mo-to-mgp="constant-fold=false  mogg-debug-dump-fused-tensor-specs=true kernel-library-paths=%kernel_api_test_pkg"
// RUN: mt --execute %t.mlir | FileCheck %s

// Unit tests to ensure that during ViewFusion the static strides are propagated correctly.

// CHECK-LABEL: Running 'broadcast_to':
// CHECK: input #0: shape = [5, 4, 3] strides = [0, 1, 0]
mo.graph @broadcast_to(%x: !mo.tensor<[4, 1], f32>) -> (!mo.tensor<[5, 4, 3], f32>) {
    %0 = mo.static.broadcast_to(%x) : !mo.tensor<[4, 1], f32> -> !mo.tensor<[5, 4, 3], f32>
    %1 = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%0) : (!mo.tensor<[5, 4, 3], f32>) -> !mo.tensor<[5, 4, 3], f32>
    mo.output %1 : !mo.tensor<[5, 4, 3], f32>
}


// CHECK-LABEL: Running 'reshape':
// CHECK: input #0: shape = [3, 4, 2] strides = [8, 2, 1]
mo.graph @reshape(%x: !mo.tensor<[24], f32>) -> (!mo.tensor<[3, 4, 2], f32>) {
  %0 = mo.static.reshape(%x) : !mo.tensor<[24], f32> -> !mo.tensor<[3, 4, 2], f32>
  %1 = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%0) : (!mo.tensor<[3, 4, 2], f32>) -> !mo.tensor<[3, 4, 2], f32>
  mo.output %1 : !mo.tensor<[3, 4, 2], f32>
}

// CHECK-LABEL: Running 'transpose':
// CHECK: input #0: shape = [3, 4, 2] strides = [4, 1, 12]
mo.graph @transpose(%x: !mo.tensor<[2, 3, 4], f32>) -> (!mo.tensor<[3, 4, 2], f32>) {
  %perm = mo.constant {value = #M.dense_array<1, 2, 0> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %0 = mo.transpose(%x, %perm) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[3, 4, 2], f32>
  %1 = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%0) : (!mo.tensor<[3, 4, 2], f32>) -> !mo.tensor<[3, 4, 2], f32>
  mo.output %1 : !mo.tensor<[3, 4, 2], f32>
}

// CHECK-LABEL: Running 'slice_dim_pos_step':
// CHECK: input #0: shape = [2, 3] strides = [5, 2]
mo.graph @slice_dim_pos_step(%x: !mo.tensor<[3, 5], f32>) -> !mo.tensor<[2, 3], f32> {
  %zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %one = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %two = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %three = mo.constant {value = #M.dense_array<3> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %five = mo.constant {value = #M.dense_array<5> : tensor<1xsi64>} : !mo.tensor<[], si64>

  %res = mo.slice_dim[0](%x, %one, %three, %one) : (
    !mo.tensor<[3, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 5], f32>
  %res2 = mo.slice_dim[1](%res, %zero, %five, %two) : (
    !mo.tensor<[2, 5], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 3], f32>
  %out = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%res2) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %out : !mo.tensor<[2, 3], f32>
}

// CHECK-LABEL: Running 'slice_dim_neg_step':
// CHECK: input #0: shape = [2, 2] strides = [-6, -1]
mo.graph @slice_dim_neg_step(%x: !mo.tensor<[5, 3], f32>) -> !mo.tensor<[2, 2], f32> {
  %neg_two = mo.constant {value = #M.dense_array<-2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %neg_one = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %zero = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>

  %res = mo.slice_dim[0](%x, %neg_one, %zero, %neg_two) : (
    !mo.tensor<[5, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 3], f32>
  %res2 = mo.slice_dim[1](%res, %neg_one, %zero, %neg_one) : (
    !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[2, 2], f32>
  %out = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%res2) : (!mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %out : !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: Running 'reshape_and_transpose':
// CHECK: input #0: shape = [3, 4, 2] strides = [4, 1, 12]
mo.graph @reshape_and_transpose(%x: !mo.tensor<[24], f32>) -> (!mo.tensor<[3, 4, 2], f32>) {
  %0 = mo.static.reshape(%x) : !mo.tensor<[24], f32> -> !mo.tensor<[2, 3, 4], f32>
  %perm = mo.constant {value = #M.dense_array<1, 2, 0> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %1 = mo.transpose(%0, %perm) : (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[3, 4, 2], f32>
  %2 = mo.custom {symbol = "print_tensor_spec_fused", device = #M.device_ref<"cpu", 0>} (%1) : (!mo.tensor<[3, 4, 2], f32>) -> !mo.tensor<[3, 4, 2], f32>
  mo.output %2 : !mo.tensor<[3, 4, 2], f32>
}
