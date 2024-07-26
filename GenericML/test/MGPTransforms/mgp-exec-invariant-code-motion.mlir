// RUN: mo-opt %s --allow-unregistered-dialect --exec-invariant-code-motion --canonicalize | FileCheck %s

//
// @kgen_execute_kernel
//

// CHECK-LABEL: mgp.model @kgen_execute_kernel : mof
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// Context has been extended to hold the cached buffer
// CHECK:       %[[CTX:.*]] = mgp.context.create 3
// Lifted ops
// CHECK-DAG:   %[[PERM_BUF:.*]] = mgp.buffer.constant %{{.*}}, %[[CTX]][1], #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16> : !M.array<32xui8>
// CHECK-DAG:   %[[WEIGHT_BUF:.*]] = mgp.buffer.constant %{{.*}}, %[[CTX]][1], dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
// CHECK-DAG:   %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0:.*]], %[[CTX]][1]
// CHECK-DAG:   %[[OUT_BUF:.*]] = mgp.buffer.alloc.static %[[CH1]], %[[CTX]][1], align 16 : !mgp.buffer<37632>
// CHECK-DAG:   %[[KERNEL_CH:.*]] = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (in %[[WEIGHT_BUF]] : {{.*}}, in %[[PERM_BUF]] : {{.*}}, out %[[OUT_BUF]] : {{.*}}), summary "kernel"
// Chain merge to cache buffer only after written to
// CHECK-DAG:   %[[BUF_ACTION_CH:.*]] = mgp.chain.device_to_host %[[KERNEL_CH]], %[[CTX]][1]
// Cache the buffer
// CHECK-DAG:   %[[CACHE_CH:.*]] = mgp.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1, 2], %[[OUT_BUF]]
// Chain merge to combine original and cache ops
// CHECK-DAG:   %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH]])
// CHECK:       mef.output %[[OUT_CH]], %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// Retrieve the buffer
// CHECK-DAG:   %[[BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 2]
// Reference to lifted chain got updated to input chain
// CHECK:       mef.output %[[CH0]], %[[BUF]]
// CHECK-NEXT:  }

mgp.model @kgen_execute_kernel : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.buffer<37632>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %perm_buf = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16>
  %weight_buf = mgp.buffer.constant %ch0, %ctx[1], dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
  %perm_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<4xsi64>
  %weight_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<64x3x7x7xf32>
  %out_buf = mgp.buffer.alloc.static %ch1, %ctx[1], align 16 : !mgp.buffer<37632>
  %out_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<7x7x3x64xf32>
  %ch2 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (in %weight_buf : !mgp.buffer<37632>, in %perm_buf : !mgp.buffer<32>, out %out_buf : !mgp.buffer<37632>), summary "kernel"
  %ch3 = mgp.chain.device_to_host %ch2, %ctx[1]
  mef.output %ch3, %out_buf : !mef.chain, !mgp.buffer<37632>
}

//
// @matmul_packb
//

// CHECK-LABEL: mgp.model @matmul_packb : mof
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// CHECK:       %[[CTX:.*]] = mgp.context.create 3
// Lifted ops
// CHECK-DAG:   %[[CONST_BUF:.*]] = mgp.buffer.constant %{{.*}}, %[[CTX]][1], #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16> : !M.array<16xui8>
// CHECK-DAG:   %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0:.*]], %[[CTX]][1]
// CHECK-DAG:   %[[IGNORED_CHAIN:.*]], %[[RES:.*]]:2 = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (), summary "matmul_pack_shape_func" out_dims 2
// CHECK-DAG:   %[[BYTES:.*]] = mip.nary.mul(%[[RES]]#0, %[[RES]]#1) {constInt = 8 : si64}
// CHECK-DAG:   %[[PACKED_BUF:.*]] = mgp.buffer.alloc.dynamic %[[CH1]], %[[CTX]][1], %[[BYTES]], align 16 : !mgp.buffer<>
// CHECK-DAG:   %[[PACK_CH:.*]] = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (in %[[CONST_BUF]] : {{.*}}, out %[[PACKED_BUF]] : {{.*}}), summary "pack_b_ndbuffer"
// Chain merge to cache buffer only after written to
// CHECK-DAG:   %[[BUF_ACTION_CH:.*]] = mgp.chain.device_to_host %[[PACK_CH]], %[[CTX]][1]
// Buffer caching
// CHECK-DAG:   %[[CACHE_CH:.*]] = mgp.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1, 2], %[[PACKED_BUF]]
// Chain merge to combine original and cache ops
// CHECK-DAG:   %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH]])
// CHECK:       mef.output %[[OUT_CH]], %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG2:.*]]: !mgp.tensor<?x2xf32>)
// Inserted buffer lookup
// CHECK-DAG:   %[[PACKED_BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 2]
// Remaining ops
// CHECK-DAG:   %{{.*}} = mgp.kernel.execute %{{.*}}, %[[CTX]][1, 1], (0, 0), (in %{{.*}} : {{.*}}, in %[[PACKED_BUF]] : {{.*}}, out %{{.*}} : {{.*}}), summary "matmul"
// CHECK:       mef.output %[[CH0]], %{{.*}}
// CHECK:       }

mgp.model @matmul_packb : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">, %arg2: !mgp.tensor<?x2xf32>) -> (!mef.chain, !mgp.tensor<?x2xf32>) {
  %const_buf = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
  %const_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<2x2xf32>
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %res:3 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (), summary "matmul_pack_shape_func" out_dims 2
  %bytes = mip.nary.mul(%res#1, %res#2) {constInt = 8 : si64}
  %packed_buf = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %bytes, align 16 : !mgp.buffer<>
  %ch2 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (in %const_buf : !mgp.buffer<16>, out %packed_buf : !mgp.buffer<>), summary "pack_b_ndbuffer"
  %arg2_buf = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x2xf32>
  %arg2_spec = mgp.tensor.extract.tensor_spec %ctx[1], %arg2 : !mgp.tensor<?x2xf32>
  %arg2_dim0 = mgp.tensor_spec.dims.dynamic %arg2_spec : !mgp.tensor_spec<?x2xf32>
  %bytes2 = mip.nary.mul(%arg2_dim0) {constInt = 16 : si64}
  %out_spec = mgp.tensor_spec.create %arg2_dim0 : !mgp.tensor_spec<?x2xf32>
  %out_buf = mgp.buffer.alloc.dynamic %ch2, %ctx[1], %bytes2, align 16 : !mgp.buffer<>
  %ch3 = mgp.kernel.execute %ch2, %ctx[1, 1], (0, 0), (in %arg2_buf : !mgp.buffer<>, in %packed_buf : !mgp.buffer<>, out %out_buf : !mgp.buffer<>), summary "matmul"
  %out_tensor = mgp.tensor.create %ch3, %ctx[1], %out_buf, %out_spec  : !mgp.tensor<?x2xf32>
  mef.output %ch0, %out_tensor : !mef.chain, !mgp.tensor<?x2xf32>
}

//
// @non_pure_op
//

// CHECK-LABEL: mgp.model @non_pure_op : mof
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// Nothing was lifted.
// CHECK:       %[[CTX:.*]] = mgp.context.create 2

mgp.model @non_pure_op : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %const_buf = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
  %const_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<2x2xf32>
  %out_spec = "foo.side_effect_op"(%const_spec) : (!mgp.tensor_spec<2x2xf32>) -> !mgp.tensor_spec<?x?xf32>
  %out_dim0, %out_dim1 = mgp.tensor_spec.dims.dynamic %out_spec : !mgp.tensor_spec<?x?xf32>
  %bytes = mip.nary.mul(%out_dim0, %out_dim1) {constInt = 8 : si64}
  %out_buf = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %bytes, align 16 : !mgp.buffer<>
  %ch2 = "bar.side_effect_op"(%ch1, %const_buf, %const_spec, %out_buf, %out_spec) : (!mgp.chain, !mgp.buffer<16>, !mgp.tensor_spec<2x2xf32>, !mgp.buffer<>, !mgp.tensor_spec<?x?xf32>) -> !mgp.chain
  %out_tensor = mgp.tensor.create %ch2, %ctx[1], %out_buf, %out_spec  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %out_tensor : !mef.chain, !mgp.tensor<?x?xf32>
}

//
// @dont_lift_if_no_computation_saved
//

// CHECK-LABEL: mgp.model @dont_lift_if_no_computation_saved : mof
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// Nothing changed
// CHECK:       %[[CTX:.*]] = mgp.context.create 2
// CHECK:       mef.output %{{.*}}, %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK:       mef.output %[[CH0]], %{{.*}}
// CHECK-NEXT:  }

mgp.model @dont_lift_if_no_computation_saved : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.buffer<0>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %out_buf = mgp.buffer.alloc.static %ch1, %ctx[1], align 16 : !mgp.buffer<0>
  mef.output %ch0, %out_buf : !mef.chain, !mgp.buffer<0>
}

//
// @intermediate_buffer
//

// CHECK-LABEL: mgp.model @intermediate_buffer : mof
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// CHECK:       %[[CTX:.*]] = mgp.context.create 3
// Lifted ops
//   constant permutation
// CHECK-DAG:   %[[PERM_BUF:.*]] = mgp.buffer.constant %{{.*}}, %[[CTX]][1], #M.aligned_bytes<"0x10000000000000000000000000000000", align 16> : !M.array<16xui8>
//   constant weights
// CHECK-DAG:   %[[WEIGHT_BUF:.*]] = mgp.buffer.constant %{{.*}}, %[[CTX]][1], dense_resource<onnx__Conv_497> : !M.aligned_bytes<1600, align 16>
//   transpose
// CHECK-DAG:   %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0:.*]], %[[CTX]][1]
// CHECK-DAG:   %[[TRANSPOSE_BUF:.*]] = mgp.buffer.alloc.static %[[CH1]], %[[CTX]][1], align 16 : !mgp.buffer<1600>
// CHECK-DAG:   %[[TRANSPOSE_CH:.*]] = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (in %[[WEIGHT_BUF]] : {{.*}}, in %[[PERM_BUF]] : {{.*}}, out %[[TRANSPOSE_BUF]] : {{.*}}), summary "transpose"
//   matmul weights packing
// CHECK-DAG:   %[[IGNORED_CH:.*]], %[[RES:.*]]:2 = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (), summary "matmul_pack_shape_func" out_dims 2
// CHECK-DAG:   %[[BYTES:.*]] = mip.nary.mul(%[[RES]]#0, %[[RES]]#1) {constInt = 8 : si64}
// CHECK-DAG:   %[[PACK_BUF:.*]] = mgp.buffer.alloc.dynamic %[[CH1]], %[[CTX]][1], %[[BYTES]], align 16 : !mgp.buffer<>
// CHECK-DAG:   %[[PACK_CH:.*]] = mgp.kernel.execute %[[TRANSPOSE_CH]], %[[CTX]][1, 1], (0, 0), (in %[[TRANSPOSE_BUF]] : {{.*}}, out %[[PACK_BUF]] : {{.*}}), summary "pack_b_ndbuffer"
// Chain merge to preserve control dependencies
// CHECK-DAG:   %[[BUF_ACTION_CH:.*]] = mgp.chain.device_to_host %[[PACK_CH]], %[[CTX]][1]
// Buffer caching (NOTE: transpose buffer isn't cached, since it's used inside init phase only)
// CHECK-DAG:   %[[CACHE_CH:.*]] = mgp.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1, 2], %[[PACK_BUF]]
// Chain merge to preserve control dependencies
// CHECK-DAG:   %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH]])
// CHECK:       mef.output %[[OUT_CH]], %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// Inserted buffer lookup
// CHECK-DAG:   %[[BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 2]
// Reference to lifted chain got updated to input chain
// CHECK-NEXT:  mef.output %[[CH0]], %[[BUF]] : !mef.chain, !mgp.buffer<>
// CHECK-NEXT: }

mgp.model @intermediate_buffer : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.buffer<>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]

  // constant permutation
  %perm_buf = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x10000000000000000000000000000000", align 16>
  %perm_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<2xsi64>

  // constant weights
  %weight_buf = mgp.buffer.constant %ch0, %ctx[1], dense_resource<onnx__Conv_497> : !M.aligned_bytes<1600, align 16>
  %weight_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<10x20xf32>

  // transpose
  %transpose_buf = mgp.buffer.alloc.static %ch1, %ctx[1], align 16 : !mgp.buffer<1600>
  %transpose_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<20x10xf32>
  %transpose_ch = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (in %weight_buf : !mgp.buffer<1600>, in %perm_buf : !mgp.buffer<16>, out %transpose_buf : !mgp.buffer<1600>), summary "transpose"

  // matmul weights packing
  %res:3 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (), summary "matmul_pack_shape_func" out_dims 2
  %bytes = mip.nary.mul(%res#1, %res#2) {constInt = 8 : si64}
  %packed_buf = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %bytes, align 16 : !mgp.buffer<>
  %packed_ch = mgp.kernel.execute %transpose_ch, %ctx[1, 1], (0, 0), (in %transpose_buf : !mgp.buffer<1600>, out %packed_buf : !mgp.buffer<>), summary "pack_b_ndbuffer"
  %ch2 = mgp.chain.device_to_host %packed_ch, %ctx[1]
  mef.output %ch2, %packed_buf : !mef.chain, !mgp.buffer<>
}


//
// @constant_buffer_used_in_both_init_and_exec
//

// CHECK-LABEL: mgp.model @constant_buffer_used_in_both_init_and_exec : mof
// Lifted constant buffers
// Old chains and context ops
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// CHECK:       %[[CTX:.*]] = mgp.context.create 4
// Lifted ops
//   constant permutation (note that this buffer is used in both init and exec)
// CHECK-DAG:   %[[PERM_BUF:.*]] = mgp.buffer.constant %{{.*}}, %[[CTX]][1], #M.aligned_bytes<"0x10000000000000000000000000000000", align 16> : !M.array<16xui8>
//   constant weights
// CHECK-DAG:   %[[WEIGHT_BUF:.*]] = mgp.buffer.constant %{{.*}}, %[[CTX]][1], dense_resource<onnx__Conv_497> : !M.aligned_bytes<1600, align 16>
//   transpose
// CHECK-DAG:   %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0:.*]], %[[CTX]][1]
// CHECK-DAG:   %[[WEIGHT_T_BUF:.*]] = mgp.buffer.alloc.static %[[CH1]], %[[CTX]][1], align 16 : !mgp.buffer<1600>
// CHECK-DAG:   %[[WEIGHT_T_CH:.*]] = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (in %[[WEIGHT_BUF]] : {{.*}}, in %[[PERM_BUF]] : {{.*}}, out %[[WEIGHT_T_BUF]] : {{.*}}), summary "transpose"
// Chain merge to preserve control dependencies
// CHECK-DAG:   %[[BUF_ACTION_CH:.*]] = mgp.chain.device_to_host %[[WEIGHT_T_CH]], %[[CTX]][1]
// Buffer caching
// CHECK-DAG:   %[[PERM_CACHE_CH:.*]] = mgp.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1, 2], %[[PERM_BUF]]
// CHECK-DAG:   %[[WEIGHT_T_CACHE_CH:.*]] = mgp.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1, 3], %[[WEIGHT_T_BUF]]
// Chain merge to preserve control dependencies
// CHECK-DAG:   %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[PERM_CACHE_CH]], %[[WEIGHT_T_CACHE_CH]])
// CHECK:       mef.output %[[OUT_CH]], %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[ARG:.*]]: !mgp.tensor<1x2xf32>) -> (!mef.chain, !mgp.buffer<1600>, !mgp.buffer<8>) {
// Inserted buffer lookup
// CHECK-DAG:   %[[PERM_BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 2]
// CHECK-DAG:   %[[WEIGHT_T_BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 3]
// CHECK-DAG:   %{{.*}} = mgp.kernel.execute %{{.*}}, %[[CTX]][1, 1], (0, 0), (in %{{.*}} : {{.*}}, in %[[PERM_BUF]] : {{.*}}, out %{{.*}} : {{.*}}), summary "transpose"
// CHECK:       mef.output %{{.*}}, %[[WEIGHT_T_BUF]], %{{.*}}
// CHECK-NEXT:  }

mgp.model @constant_buffer_used_in_both_init_and_exec : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">, %arg: !mgp.tensor<1x2xf32>) -> (!mef.chain, !mgp.buffer<1600>, !mgp.buffer<8>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %arg_buf = mgp.tensor.extract.buffer %ctx[1], %arg : !mgp.tensor<1x2xf32>

  // constant permutation
  %perm_buf = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x10000000000000000000000000000000", align 16>
  %perm_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<2xsi64>

  // constant weights
  %weight_buf = mgp.buffer.constant %ch0, %ctx[1], dense_resource<onnx__Conv_497> : !M.aligned_bytes<1600, align 16>
  %weight_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<10x20xf32>

  // constant weight transpose
  %weight_t_buf = mgp.buffer.alloc.static %ch1, %ctx[1], align 16 : !mgp.buffer<1600>
  %weight_t_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<20x10xf32>
  %weight_t_ch = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (in %weight_buf : !mgp.buffer<1600>, in %perm_buf : !mgp.buffer<16>, out %weight_t_buf : !mgp.buffer<1600>), summary "transpose"

  // non-constant arg transpose
  %arg_t_spec = mgp.tensor_spec.create  : !mgp.tensor_spec<2x1xf32>
  %arg_t_buf = mgp.buffer.alloc.static %ch1, %ctx[1], align 16 : !mgp.buffer<8>
  %arg_t_ch = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (in %arg_buf : !mgp.buffer<8>, in %perm_buf : !mgp.buffer<16>, out %arg_t_buf : !mgp.buffer<8>), summary "transpose"

  %ch2 = mgp.chain.create %ctx[1], (%weight_t_ch, %arg_t_ch)
  %ch3 = mgp.chain.device_to_host %ch2, %ctx[1]
  mef.output %ch3, %weight_t_buf, %arg_t_buf : !mef.chain, !mgp.buffer<1600>, !mgp.buffer<8>
}

//
// @lift_non_trivial_buffer_op_without_liftable_user
//

// CHECK-LABEL: mgp.model @lift_non_trivial_buffer_op_without_liftable_user : mof
// Old chains and context ops
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// CHECK:       %[[CTX:.*]] = mgp.context.create 3
// Lifted ops
// CHECK-DAG:   %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0:.*]], %[[CTX]][1]
// CHECK-DAG:   %[[IGNORED_CH:.*]], %[[RES:.*]]:2 = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (), summary "foo" out_dims 2
// CHECK-DAG:   %[[BYTES:.*]] = mip.nary.mul(%[[RES]]#0, %[[RES]]#1) {constInt = 16 : si64}
// CHECK-DAG:   %[[BUF:.*]] = mgp.buffer.alloc.dynamic %[[CH1]], %[[CTX]][1], %[[BYTES]], align 32 : !mgp.buffer<>
// (Redundant) Chain merge to preserve control dependencies
// CHECK-DAG:   %[[BUF_ACTION_CH:.*]] = grt.chain.create ()
// Buffer caching
// CHECK-DAG:   %[[CACHE_CH:.*]] = mgp.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1, 2], %[[BUF]]
// Chain merge to preserve control dependencies
// CHECK-DAG:   %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH]])
// CHECK:       mef.output %[[OUT_CH]], %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// Inserted buffer lookup
// CHECK-DAG:   %[[BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 2]
// CHECK:       mef.output %[[CH0]], %[[BUF]]
// CHECK-NEXT:  }

mgp.model @lift_non_trivial_buffer_op_without_liftable_user : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.buffer<>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %res:3 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (), summary "foo" out_dims 2
  %bytes = mip.nary.mul(%res#1, %res#2) {constInt = 16 : si64}
  %buf = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %bytes, align 32 : !mgp.buffer<>
  mef.output %ch0, %buf : !mef.chain, !mgp.buffer<>
}

//
// @collect_write_chains_for_lifted_buffer
//

// CHECK-LABEL: mgp.model @collect_write_chains_for_lifted_buffer : mof
// Old chains and context ops
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// CHECK:       %[[CTX:.*]] = mgp.context.create 4
// Lifted ops
// CHECK-DAG:   %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0:.*]], %[[CTX]][1]
// CHECK-DAG:   %[[BUF:.*]] = mgp.buffer.alloc.static %[[CH1]], %[[CTX]][1], align 16 : !mgp.buffer<100>
// CHECK-DAG:   %[[OP_CH:.*]], %[[DIM:.*]] = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (out %[[BUF]] : {{.*}}), summary "foo" out_dims 1
// CHECK-DAG:   %[[BYTES:.*]] = mip.nary.mul(%[[DIM]]) {constInt = 8 : si64}
// CHECK-DAG:   %[[OUT_BUF:.*]] = mgp.buffer.alloc.dynamic %[[CH1]], %[[CTX]][1], %[[BYTES]], align 32 : !mgp.buffer<>
// Chain merge to preserve control dependencies (NOTE that it took the kernel output chain into account)
// CHECK-DAG:   %[[BUF_ACTION_CH:.*]] = mgp.chain.device_to_host %[[OP_CH]], %[[CTX]][1]
// Buffer caching
// CHECK-DAG:   %[[CACHE_CH1:.*]] = mgp.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1, 2], %[[BUF]]
// CHECK-DAG:   %[[CACHE_CH2:.*]] = mgp.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1, 3], %[[OUT_BUF]]
// Chain merge to preserve control dependencies
// CHECK-DAG:   %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH1]], %[[CACHE_CH2]])
// CHECK-NEXT:  mef.output %[[OUT_CH]], %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// Inserted buffer lookup
// CHECK-DAG:   %[[BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 2]
// CHECK-DAG:   %[[OUT_BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 3]
// CHECK-NEXT:  mef.output %[[CH0]], %[[BUF]], %[[OUT_BUF]]
// CHECK-NEXT:  }

mgp.model @collect_write_chains_for_lifted_buffer : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.buffer<100>, !mgp.buffer<>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %buf = mgp.buffer.alloc.static %ch1, %ctx[1], align 16 : !mgp.buffer<100>
  %res:2 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (out %buf : !mgp.buffer<100>), summary "foo" out_dims 1
  %bytes = mip.nary.mul(%res#1) {constInt = 8 : si64}
  %out_buf = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %bytes, align 32 : !mgp.buffer<>
  mef.output %ch0, %buf, %out_buf : !mef.chain, !mgp.buffer<100>, !mgp.buffer<>
}

//
// @dont_lift_if_mutated_by_unliftable
//

// CHECK-LABEL: mgp.model @dont_lift_if_mutated_by_unliftable : mof
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// Nothing was lifted
// CHECK:       %{{.*}} = mgp.context.create 2

mgp.model @dont_lift_if_mutated_by_unliftable : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.buffer<100>, !mgp.buffer<>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %buf = mgp.buffer.alloc.static %ch1, %ctx[1], align 16 : !mgp.buffer<100>
  %ch2, %dim0 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (out %buf : !mgp.buffer<100>), summary "foo" out_dims 1
  %ch3 = "foo.side_effect_op"(%dim0) : (index) -> !mgp.chain
  %ch4, %dim1 = mgp.kernel.execute %ch3, %ctx[1, 1], (0, 0), (out %buf : !mgp.buffer<100>), summary "foo" out_dims 1
  %bytes = mip.nary.mul(%dim1) {constInt = 16 : si64}
  %out_buf = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %bytes, align 32 : !mgp.buffer<>
  mef.output %ch0, %buf, %out_buf : !mef.chain, !mgp.buffer<100>, !mgp.buffer<>
}

//
// @cloned_need_indirectly
//

// CHECK-LABEL: mgp.model @cloned_needed_indirectly : mof
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// CHECK:       %[[CTX:.*]] = mgp.context.create 3
// CHECK-DAG:   %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0:.*]], %[[CTX]][1]
// CHECK-DAG:   %[[BUF0:.*]] = mgp.buffer.alloc.static %[[CH1]], %[[CTX]][1], align 64 : !mgp.buffer<2359296>
// CHECK-DAG:   %[[CH2:.*]] = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (out %[[BUF0]] : !mgp.buffer<2359296>), summary "kernelA"
// CHECK-DAG:   %[[CH3:.*]], %[[DIMS:.*]]:2 = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (), summary "shapeFunction" out_dims 2
// CHECK-DAG:   %[[BYTES:.*]] = mip.nary.mul(%[[DIMS]]#0, %[[DIMS]]#1) {constInt = 8 : si64}
// CHECK-DAG:   %[[CH4:.*]] = mgp.chain.create %[[CTX]][1], (%[[CH1]], %[[CH2]])
// CHECK-DAG:   %[[BUF1:.*]] = mgp.buffer.alloc.dynamic %[[CH4]], %[[CTX]][1], %[[BYTES]], align 64 : !mgp.buffer<>
// CHECK-DAG:   %[[CH5:.*]] = mgp.kernel.execute %[[CH2]], %[[CTX]][1, 1], (0, 0), (in %[[BUF0]] : !mgp.buffer<2359296>, out %[[BUF1]] : !mgp.buffer<>), (%outDimensions#0, %outDimensions#1), summary "kernelB"
// CHECK-DAG:   %[[CH6:.*]] = mgp.chain.device_to_host %[[CH5]], %[[CTX]][1]
// CHECK-DAG:   %[[CH7:.*]] = mgp.buffer.cache %[[CH6]], %[[CTX]][1, 2], %[[BUF1]]
// CHECK-DAG:   %[[CH8:.*]] = grt.chain.create (%[[CH0]], %[[CH7]])
// CHECK:       mef.output %[[CH8]], %[[CTX]]
//
// CHECK-NEXT:  } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// Though not directly needed by an unliftable op, this computation nevertheless must be cloned.
// CHECK-DAG:   %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK-DAG:   %[[CH2:.*]], %[[DIMS:.*]]:2 = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (), summary "shapeFunction" out_dims 2
// CHECK-DAG:   %[[SPEC:.*]] = mgp.tensor_spec.create %[[DIMS]]#0, %[[DIMS]]#1 : !mgp.tensor_spec<?x?xf32>
// CHECK-DAG:   %[[BUF:.*]] = mgp.buffer.get_cached %[[CH0]], %[[CTX]][1, 2] : !mgp.buffer<>
// CHECK-DAG:   %[[CH1_DUP:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK-DAG:   %[[TEN:.*]] = mgp.tensor.create %[[CH1_DUP]], %[[CTX]][1], %[[BUF]], %[[SPEC]]
// CHECK:       mef.output %[[CH0]], %[[TEN]]
// CHECK-NEXT:  }

mgp.model @cloned_needed_indirectly : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.tensor_spec.create  : !mgp.tensor_spec<768x768xf32>
  %1 = mgp.buffer.alloc.static %ch1, %ctx[1], align 64 : !mgp.buffer<2359296>
  %ch2 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (out %1 : !mgp.buffer<2359296>), summary "kernelA"
  %ch3, %outDimensions:2 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (), summary "shapeFunction" out_dims 2
  %2 = mgp.tensor_spec.create %outDimensions#0, %outDimensions#1 : !mgp.tensor_spec<?x?xf32>
  %3 = mip.nary.mul(%outDimensions#0, %outDimensions#1) {constInt = 8 : si64}
  %4 = mgp.chain.create %ctx[1], (%ch1, %ch2)
  %5 = mgp.buffer.alloc.dynamic %4, %ctx[1], %3, align 64 : !mgp.buffer<>
  %ch4 = mgp.kernel.execute %ch2, %ctx[1, 1], (0, 0), (in %1 : !mgp.buffer<2359296>, out %5 : !mgp.buffer<>), (%outDimensions#0, %outDimensions#1), summary "kernelB"
  %6 = mgp.tensor.create %ch4, %ctx[1], %5, %2 : !mgp.tensor<?x?xf32>
  %7 = mgp.tensor_spec.create : !mgp.tensor_spec<ui64>
  mef.output %ch0, %6 : !mef.chain, !mgp.tensor<?x?xf32>
}


//
// @while
//

// CHECK-LABEL: mgp.model @while : mof
// Nothing was lifted
// CHECK-NEXT:  %{{.*}} = grt.chain.create ()
// CHECK:       %{{.*}} = mgp.context.create 2

mgp.model @while : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.tensor<4xui8>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x00000004", align 8> : !M.array<4xui8>
  %1 = mgp.tensor_spec.create  : !mgp.tensor_spec<ui32>
  %2 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>
  %4 = mgp.tensor_spec.create  : !mgp.tensor_spec<4xui8>
  %5 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x01010101", align 8> : !M.array<4xui8>
  %7:11 = grt.while (%ch1 as %arg2: !mgp.chain, %ch1 as %arg3: !mgp.chain, %0 as %arg4: !mgp.buffer<4>, %1 as %arg5: !mgp.tensor_spec<ui32>, %ch1 as %arg6: !mgp.chain, %2 as %arg7: !mgp.buffer<4>, %4 as %arg8: !mgp.tensor_spec<4xui8>, %ch1 as %arg9: !mgp.chain, %5 as %arg10: !mgp.buffer<4>, %4 as %arg11: !mgp.tensor_spec<4xui8>, %ctx as %arg12: !mgp.context) {
    %ch1h = mgp.chain.device_to_host %arg2, %arg12[1]
    %9 = mgp.buffer.constant %ch1h, %arg12[1], #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>
    %10 = mgp.buffer.alloc.static %arg2, %arg12[1], align 32 : !mgp.buffer<1>
    %11 = mgp.tensor_spec.create  : !mgp.tensor_spec<bool>
    %12 = mgp.kernel.execute %arg3, %arg12[1, 1], (0, 0), (in %arg4: !mgp.buffer<4>, in %9: !mgp.buffer<4>, out %10: !mgp.buffer<1>), summary "mo.greater"
    %13 = mgp.buffer.to_bool %12, %arg12[1], %10 : !mgp.buffer<1>
    mef.output %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %13 : !mgp.chain, !mgp.chain, !mgp.buffer<4>, !mgp.tensor_spec<ui32>, !mgp.chain, !mgp.buffer<4>, !mgp.tensor_spec<4xui8>, !mgp.chain, !mgp.buffer<4>, !mgp.tensor_spec<4xui8>, !mgp.context, i1
  } do {
    %9 = mgp.buffer.alloc.static %arg2, %arg12[1] : !mgp.buffer<4>
    %10 = mgp.tensor_spec.create  : !mgp.tensor_spec<4xui8>
    %11 = mgp.chain.create %arg12[1], (%arg6, %arg9)
    %12 = mgp.kernel.execute %11, %arg12[1, 1], (0, 0), (in %arg7: !mgp.buffer<4>, in %arg10: !mgp.buffer<4>, out %9: !mgp.buffer<4>), summary "add"
    %13 = "some.op"(%12, %9, %10) : (!mgp.chain, !mgp.buffer<4>, !mgp.tensor_spec<4xui8>) -> !mgp.tensor<4xui8>
    %bufferResult = mgp.tensor.extract.buffer %arg12[1], %13 : !mgp.tensor<4xui8>
    %tensorSpecResult = mgp.tensor.extract.tensor_spec %arg12[1], %13 : !mgp.tensor<4xui8>
    %ch1h = mgp.chain.device_to_host %arg2, %arg12[1]
    %14 = mgp.buffer.constant %ch1h, %arg12[1], #M.aligned_bytes<"0x00000001", align 8> : !M.array<4xui8>
    %15 = mgp.buffer.alloc.static %arg2, %arg12[1], align 32 : !mgp.buffer<4>
    %16 = mgp.kernel.execute %arg3, %arg12[1, 1], (0, 0), (in %arg4: !mgp.buffer<4>, in %14: !mgp.buffer<4>, out %15: !mgp.buffer<4>), summary "sub"
    mef.output %arg2, %16, %15, %arg5, %arg2, %bufferResult, %tensorSpecResult, %arg9, %arg10, %arg11, %arg12 : !mgp.chain, !mgp.chain, !mgp.buffer<4>, !mgp.tensor_spec<ui32>, !mgp.chain, !mgp.buffer<4>, !mgp.tensor_spec<4xui8>, !mgp.chain, !mgp.buffer<4>, !mgp.tensor_spec<4xui8>, !mgp.context
  }
  %8 = mgp.tensor.create %7#4, %ctx[1], %7#5, %7#6 : !mgp.tensor<4xui8>
  %ch2 = mgp.chain.device_to_host %7#0, %ctx[1]
  mef.output %ch2, %8 : !mef.chain, !mgp.tensor<4xui8>
}
