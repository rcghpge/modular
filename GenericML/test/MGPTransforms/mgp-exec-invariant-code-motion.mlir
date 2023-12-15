// RUN: mgp-opt %s --allow-unregistered-dialect --exec-invariant-code-motion | FileCheck %s

//
// @kgen_execute_kernel
//

// CHECK-LABEL: mgp.model @kgen_execute_kernel : mof init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create ()
// Context has been extended to hold the cached buffer
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 2
// Lifted ops
// CHECK-DAG:    %[[PERM_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16>
// CHECK-DAG:    %[[WEIGHT_BUF:.*]] = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
// CHECK-DAG:    %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 37632, align 16
// CHECK-DAG:    %[[TRANSPOSE_CH:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (in %[[WEIGHT_BUF]] : {{.*}}, in %[[PERM_BUF]] : {{.*}}, out %[[OUT_BUF]] : {{.*}}), summary "foo" {numBackOutputBuffers = 1 : ui32}
// Chain merge to cache buffer only after written to
// CHECK-DAG:    %[[BUF_ACTION_CH:.*]] = grt.chain.create (%[[TRANSPOSE_CH]])
// Cache the buffer
// CHECK-DAG:    %[[CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[OUT_BUF]] : !mgp.cpu.buffer<37632>
// Chain merge to combine original and cache ops
// CHECK-DAG:    %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH]])
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<37632>) {
// Retrieve the buffer
// CHECK-DAG:    %[[BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer<37632>
// Reference to lifted chain got updated to input chain
// CHECK:        mef.output %[[CH0]], %[[BUF]] : !mef.chain, !mgp.cpu.buffer<37632>
// CHECK-NEXT: }

mgp.model @kgen_execute_kernel : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  // Don't bother setting up the compiled kernel context
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<37632>) {
  %perm_buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16>
  %weight_buf = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
  %perm_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<4xsi64>
  %weight_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<64x3x7x7xf32>
  %out_buf = mgp.cpu.buffer.alloc.static %ch0, 37632, align 16
  %out_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<7x7x3x64xf32>
  %out_ch = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (in %weight_buf : !mgp.cpu.buffer<37632>, in %perm_buf : !mgp.cpu.buffer<32>, out %out_buf : !mgp.cpu.buffer<37632>), summary "foo" {numBackOutputBuffers = 1 : ui32}
  mef.output %out_ch, %out_buf : !mef.chain, !mgp.cpu.buffer<37632>
}

//
// @matmul_packb
//

// CHECK-LABEL: mgp.model @matmul_packb : mof init {
// CHECK-NEXT:  %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:  %[[CTX:.*]] = mgp.context.create 2
// Lifted ops
// CHECK-DAG:   %[[CONST_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
// CHECK-DAG:   %[[IGNORED_CHAIN:.*]], %[[RES:.*]]:2 = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (), summary "matmul_pack_shape_func" out_dims 2
// CHECK-DAG:   %[[PACKED_SPEC:.*]] = mgp.cpu.tensor_spec.create %[[RES]]#0, %[[RES]]#1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:   %[[PACKED_BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH0]], %[[PACKED_SPEC]], align 16 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:   %[[PACK_CH:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (in %[[CONST_BUF]] : {{.*}}, out %[[PACKED_BUF]] : {{.*}}), summary "pack_b_ndbuffer" {numBackOutputBuffers = 1 : ui32}
// Chain merge to preserve control dependencies
// CHECK-DAG:   %[[BUF_ACTION_CH:.*]] = grt.chain.create (%[[PACK_CH]])
// Buffer caching
// CHECK-DAG:   %[[CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[PACKED_BUF]] : !mgp.cpu.buffer
// Chain merge to preserve control dependencies
// CHECK-DAG:   %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH]])
// CHECK-NEXT:  mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT:} execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x2xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x2xf32>) {
// Inserted buffer lookup
// CHECK-DAG:   %[[PACKED_BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer
// Ops that didn't get lifted, but references to chain and buffer got updated.
// CHECK-DAG:   %[[ARG2_BUF:.*]], %[[ARG2_SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x2xf32>
// CHECK-DAG:   %[[ARG2_DIM0:.*]] = mgp.cpu.tensor_spec.dims.dynamic %[[ARG2_SPEC]] : !mgp.cpu.tensor_spec<?x2xf32>
// CHECK-DAG:   %[[OUT_SPEC:.*]] = mgp.cpu.tensor_spec.create %[[ARG2_DIM0]] : !mgp.cpu.tensor_spec<?x2xf32>
// CHECK-DAG:   %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH0]], %[[OUT_SPEC]], align 16 : !mgp.cpu.tensor_spec<?x2xf32>
// CHECK-DAG:   %[[CH2:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (in %[[ARG2_BUF]] : {{.*}}, in %[[PACKED_BUF]] : {{.*}}, out %[[OUT_BUF]] : {{.*}}), summary "matmul" {numBackOutputBuffers = 1 : ui32}
// CHECK-DAG:   %[[OUT_TENSOR:.*]] = mgp.cpu.tensor.create %[[CH2]], %[[OUT_BUF]], %[[OUT_SPEC]]  : !mgp.cpu.tensor<?x2xf32>
// CHECK-NEXT:  mef.output %[[CH0]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<?x2xf32>
// CHECK-NEXT:}

mgp.model @matmul_packb : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  // Don't bother setting up the compiled kernel context
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %arg2: !mgp.cpu.tensor<?x2xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x2xf32>) {
  %const_buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
  %const_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2x2xf32>

  %res:3 = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (), summary "matmul_pack_shape_func" out_dims 2
  %packed_spec = mgp.cpu.tensor_spec.create %res#1, %res#2 : !mgp.cpu.tensor_spec<?x?xf32>
  %packed_buf = mgp.cpu.buffer.alloc.spec %ch0, %packed_spec, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
  %ch1 = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (in %const_buf : !mgp.cpu.buffer<16>, out %packed_buf : !mgp.cpu.buffer), summary "pack_b_ndbuffer" {numBackOutputBuffers = 1 : ui32}
  %arg2_buf, %arg2_spec = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x2xf32>
  %arg2_dim0 = mgp.cpu.tensor_spec.dims.dynamic %arg2_spec : !mgp.cpu.tensor_spec<?x2xf32>
  %out_spec = mgp.cpu.tensor_spec.create %arg2_dim0 : !mgp.cpu.tensor_spec<?x2xf32>
  %out_buf = mgp.cpu.buffer.alloc.spec %ch1, %out_spec, align 16 : !mgp.cpu.tensor_spec<?x2xf32>
  %ch2 = mgp.cpu.kernel.execute %ch1, %ctx[0, 0], (in %arg2_buf : !mgp.cpu.buffer, in %packed_buf : !mgp.cpu.buffer, out %out_buf : !mgp.cpu.buffer), summary "matmul" {numBackOutputBuffers = 1 : ui32}
  %out_tensor = mgp.cpu.tensor.create %ch2, %out_buf, %out_spec  : !mgp.cpu.tensor<?x2xf32>
  mef.output %ch0, %out_tensor : !mef.chain, !mgp.cpu.tensor<?x2xf32>
}

//
// @non_pure_op
//

// CHECK-LABEL: mgp.model @non_pure_op : mof init {
// CHECK-NEXT:  %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:  %[[CTX:.*]] = mgp.context.create 1
// Trivial chain merge to preserve control dependencies
// CHECK-NEXT:  %[[BUF_ACTION_CH:.*]] = grt.chain.create ()
// Trivial chain merge to preserve control dependencies
// CHECK-NEXT:  %[[OUT_CH:.*]] = grt.chain.create ({{.*}})
// CHECK-NEXT:  mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT:} execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// No op got lifted/recomputed because they have side effects.
// CHECK-DAG:   %[[CONST_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
// CHECK-DAG:   %[[CONST_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2x2xf32>
// CHECK-DAG:   %[[OUT_SPEC:.*]] = "foo.side_effect_op"(%[[CONST_SPEC]]) : (!mgp.cpu.tensor_spec<2x2xf32>) -> !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:   %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH0]], %[[OUT_SPEC]], align 16 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:   %[[PACK_CH:.*]] = "bar.side_effect_op"(%[[CH0]], %[[CONST_BUF]], %[[CONST_SPEC]], %[[OUT_BUF]], %[[OUT_SPEC]]) : (!mef.chain, !mgp.cpu.buffer<16>, !mgp.cpu.tensor_spec<2x2xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x?xf32>) -> !mef.chain
// CHECK-DAG:   %[[OUT_TENSOR:.*]] = mgp.cpu.tensor.create %[[PACK_CH]], %[[OUT_BUF]], %[[OUT_SPEC]]  : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:  mef.output %[[CH0]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:}

mgp.model @non_pure_op : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
  %const_buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
  %const_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2x2xf32>
  %out_spec = "foo.side_effect_op"(%const_spec) : (!mgp.cpu.tensor_spec<2x2xf32>) -> !mgp.cpu.tensor_spec<?x?xf32>
  %out_buf = mgp.cpu.buffer.alloc.spec %ch0, %out_spec, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
  %ch1 = "bar.side_effect_op"(%ch0, %const_buf, %const_spec, %out_buf, %out_spec) : (!mef.chain, !mgp.cpu.buffer<16>, !mgp.cpu.tensor_spec<2x2xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x?xf32>) -> !mef.chain
  %out_tensor = mgp.cpu.tensor.create %ch1, %out_buf, %out_spec  : !mgp.cpu.tensor<?x?xf32>
  mef.output %ch0, %out_tensor : !mef.chain, !mgp.cpu.tensor<?x?xf32>
}

//
// @weight_transpose
//

// CHECK-LABEL: mgp.model @weight_transpose : mof init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 2
// Lifted ops
// CHECK-DAG:    %[[PERM_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16>
// CHECK-DAG:    %[[WEIGHT_BUF:.*]] = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
// CHECK-DAG:    %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 37632, align 16
// CHECK-DAG:    %[[TRANSPOSE_CH:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (in %[[WEIGHT_BUF]] : {{.*}}, in %[[PERM_BUF]] : {{.*}}, out %[[OUT_BUF]] : {{.*}}), summary "transpose" {numBackOutputBuffers = 1 : ui32}
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[BUF_ACTION_CH:.*]] = grt.chain.create (%[[TRANSPOSE_CH]])
// Buffer caching
// CHECK-DAG:    %[[CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[OUT_BUF]] : !mgp.cpu.buffer<37632>
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH]])
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<37632>) {
// Inserted buffer lookup
// CHECK-DAG:    %[[BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer<37632>
// Reference to lifted chain got updated to input chain
// CHECK:        mef.output %[[CH0]], %[[BUF]] : !mef.chain, !mgp.cpu.buffer<37632>
// CHECK-NEXT: }

mgp.model @weight_transpose : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  // Don't bother setting up the compiled kernel context
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<37632>) {
  %perm_buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16>
  %weight_buf = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
  %perm_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<4xsi64>
  %weight_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<64x3x7x7xf32>
  %out_buf = mgp.cpu.buffer.alloc.static %ch0, 37632, align 16
  %out_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<7x7x3x64xf32>
  %out_ch = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (in %weight_buf : !mgp.cpu.buffer<37632>, in %perm_buf : !mgp.cpu.buffer<32>, out %out_buf : !mgp.cpu.buffer<37632>), summary "transpose" {numBackOutputBuffers = 1 : ui32}
  mef.output %out_ch, %out_buf : !mef.chain, !mgp.cpu.buffer<37632>
}

//
// @dont_lift_if_no_computation_saved
//

// CHECK-LABEL: mgp.model @dont_lift_if_no_computation_saved : mof init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 1
// Chain merge to preserve control dependencies (will be cleared by DCE)
// CHECK-NEXT:   %[[BUF_ACTION_CH:.*]] = grt.chain.create ()
// Chain merge to preserve control dependencies
// CHECK-NEXT:   %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]])
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<0>) {
// Nothing changed
// CHECK-NEXT:   %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 0, align 16
// CHECK-NEXT:   mef.output %[[CH0]], %[[OUT_BUF]] : !mef.chain, !mgp.cpu.buffer<0>
// CHECK-NEXT: }

mgp.model @dont_lift_if_no_computation_saved : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<0>) {
  %out_buf = mgp.cpu.buffer.alloc.static %ch0, 0, align 16
  mef.output %ch0, %out_buf : !mef.chain, !mgp.cpu.buffer<0>
}

//
// @intermediate_buffer
//

// CHECK-LABEL: mgp.model @intermediate_buffer : mof init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 2
// Lifted ops
//   constant permutation
// CHECK-DAG:    %[[PERM_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x10000000000000000000000000000000", align 16>
//   constant weights
// CHECK-DAG:    %[[WEIGHT_BUF:.*]] = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<1600, align 16>
//   transpose
// CHECK-DAG:    %[[TRANSPOSE_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 1600, align 16
// CHECK-DAG:    %[[TRANSPOSE_CH:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (in %[[WEIGHT_BUF]] : {{.*}}, in %[[PERM_BUF]] : {{.*}}, out %[[TRANSPOSE_BUF]] : {{.*}}), summary "transpose" {numBackOutputBuffers = 1 : ui32}
//   matmul weights packing
// CHECK-DAG:    %[[IGNORED_CH:.*]], %[[RES:.*]]:2 = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (), summary "matmul_pack_shape_func" out_dims 2 {numBackOutputBuffers = 0 : ui32}
// CHECK-DAG:    %[[PACK_SPEC:.*]] = mgp.cpu.tensor_spec.create %[[RES]]#0, %[[RES]]#1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:    %[[PACK_BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH0]], %[[PACK_SPEC]], align 16 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:    %[[PACK_CH:.*]] = mgp.cpu.kernel.execute %[[TRANSPOSE_CH]], %[[CTX]][0, 0], (in %[[TRANSPOSE_BUF]] : {{.*}}, out %[[PACK_BUF]] : {{.*}}), summary "pack_b_ndbuffer" {numBackOutputBuffers = 1 : ui32}
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[BUF_ACTION_CH:.*]] = grt.chain.create (%[[PACK_CH]])
// Buffer caching (NOTE: transpose buffer isn't cache, since it's used inside init phase only)
// CHECK-DAG:    %[[CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[PACK_BUF]] : !mgp.cpu.buffer
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH]])
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer) {
// Inserted buffer lookup
// CHECK-DAG:    %[[BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer
// Reference to lifted chain got updated to input chain
// CHECK-NEXT:   mef.output %[[CH0]], %[[BUF]] : !mef.chain, !mgp.cpu.buffer
// CHECK-NEXT: }

mgp.model @intermediate_buffer : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  // Don't bother setting up the compiled kernel context
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer) {
  // constant permutation
  %perm_buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x10000000000000000000000000000000", align 16>
  %perm_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2xsi64>

  // constant weights
  %weight_buf = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<1600, align 16>
  %weight_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<10x20xf32>

  // transpose
  %transpose_buf = mgp.cpu.buffer.alloc.static %ch0, 1600, align 16
  %transpose_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<20x10xf32>
  %transpose_ch = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (in %weight_buf : !mgp.cpu.buffer<1600>, in %perm_buf : !mgp.cpu.buffer<16>, out %transpose_buf : !mgp.cpu.buffer<1600>), summary "transpose" {numBackOutputBuffers = 1 : ui32}

  // matmul weights packing
  %res:3 = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (), summary "matmul_pack_shape_func" out_dims 2 {numBackOutputBuffers = 0 : ui32}
  %packed_spec = mgp.cpu.tensor_spec.create %res#1, %res#2 : !mgp.cpu.tensor_spec<?x?xf32>
  %packed_buf = mgp.cpu.buffer.alloc.spec %ch0, %packed_spec, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
  %packed_ch = mgp.cpu.kernel.execute %transpose_ch, %ctx[0, 0], (in %transpose_buf : !mgp.cpu.buffer<1600>, out %packed_buf : !mgp.cpu.buffer), summary "pack_b_ndbuffer" {numBackOutputBuffers = 1 : ui32}

  mef.output %packed_ch, %packed_buf : !mef.chain, !mgp.cpu.buffer
}

//
// @constant_buffer_used_in_both_init_and_exec
//

// CHECK-LABEL: mgp.model @constant_buffer_used_in_both_init_and_exec : mof init {
// Lifted constant buffers
// Old chains and context ops
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 2
// Lifted ops
//   constant permutation (note that this buffer is used in both init and exec)
// CHECK-DAG:    %[[PERM_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x10000000000000000000000000000000", align 16>
//   constant weights
// CHECK-DAG:    %[[WEIGHT_BUF:.*]] = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<1600, align 16>
//   transpose
// CHECK-DAG:    %[[WEIGHT_T_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 1600, align 16
// CHECK-DAG:    %[[WEIGHT_T_CH:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (in %[[WEIGHT_BUF]] : {{.*}}, in %[[PERM_BUF]] : {{.*}}, out %[[WEIGHT_T_BUF]] : {{.*}}), summary "transpose" {numBackOutputBuffers = 1 : ui32}
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[BUF_ACTION_CH:.*]] = grt.chain.create (%[[WEIGHT_T_CH]])
// Buffer caching
// CHECK-DAG:    %[[PERM_CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][0], %[[PERM_BUF]] : !mgp.cpu.buffer<16>
// CHECK-DAG:    %[[WEIGHT_T_CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[WEIGHT_T_BUF]] : !mgp.cpu.buffer<1600>
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[PERM_CACHE_CH]], %[[WEIGHT_T_CACHE_CH]])
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG:.*]]: !mgp.cpu.tensor<1x2xf32>) -> (!mef.chain, !mgp.cpu.buffer<1600>, !mgp.cpu.buffer<8>) {
// CHECK-DAG:    %[[ARG_BUF:.*]], %[[ARG_SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG]] : !mgp.cpu.tensor<1x2xf32>
// Inserted buffer lookup
// CHECK-DAG:    %[[PERM_BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][0] : !mgp.cpu.buffer<16>
// CHECK-DAG:    %[[WEIGHT_T_BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer<1600>
// Old non-constant arg transpose
// Note that all references to lifted chain got updated to input chain
// CHECK-DAG:    %[[ARG_T_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 8, align 16
// CHECK-DAG:    %[[ARG_T_CH:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (in %[[ARG_BUF]] : {{.*}}, in %[[PERM_BUF]] : {{.*}}, out %[[ARG_T_BUF]] : {{.*}}), summary "transpose" {numBackOutputBuffers = 1 : ui32}
// Final output
// CHECK-DAG:    %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[ARG_T_CH]])
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[WEIGHT_T_BUF]], %[[ARG_T_BUF]] : !mef.chain, !mgp.cpu.buffer<1600>, !mgp.cpu.buffer<8>
// CHECK-NEXT: }

mgp.model @constant_buffer_used_in_both_init_and_exec : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 0
  // Don't bother setting up the compiled kernel context
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %arg: !mgp.cpu.tensor<1x2xf32>) -> (!mef.chain, !mgp.cpu.buffer<1600>, !mgp.cpu.buffer<8>) {
  %arg_buf, %arg_spec = mgp.cpu.tensor.extract %arg : !mgp.cpu.tensor<1x2xf32>

  // constant permutation
  %perm_buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x10000000000000000000000000000000", align 16>
  %perm_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2xsi64>

  // constant weights
  %weight_buf = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<1600, align 16>
  %weight_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<10x20xf32>

  // constant weight transpose
  %weight_t_buf = mgp.cpu.buffer.alloc.static %ch0, 1600, align 16
  %weight_t_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<20x10xf32>
  %weight_t_ch = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (in %weight_buf : !mgp.cpu.buffer<1600>, in %perm_buf : !mgp.cpu.buffer<16>, out %weight_t_buf : !mgp.cpu.buffer<1600>), summary "transpose" {numBackOutputBuffers = 1 : ui32}

  // non-constant arg transpose
  %arg_t_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2x1xf32>
  %arg_t_buf = mgp.cpu.buffer.alloc.static %ch0, 8, align 16
  %arg_t_ch = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (in %arg_buf : !mgp.cpu.buffer<8>, in %perm_buf : !mgp.cpu.buffer<16>, out %arg_t_buf : !mgp.cpu.buffer<8>), summary "transpose" {numBackOutputBuffers = 1 : ui32}

  %out_ch = grt.chain.create (%weight_t_ch, %arg_t_ch)
  mef.output %out_ch, %weight_t_buf, %arg_t_buf : !mef.chain, !mgp.cpu.buffer<1600>, !mgp.cpu.buffer<8>
}

//
// @lift_non_trivial_buffer_op_without_liftable_user
//

// CHECK-LABEL: mgp.model @lift_non_trivial_buffer_op_without_liftable_user : mof init {
// Old chains and context ops
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 2
// Lifted ops
// CHECK-DAG:    %[[IGNORED_CH:.*]], %[[RES:.*]]:2 = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (), summary "foo" out_dims 2
// CHECK-DAG:    %[[SPEC:.*]] = mgp.cpu.tensor_spec.create %[[RES]]#0, %[[RES]]#1 : !mgp.cpu.tensor_spec<?x?xf64>
// CHECK-DAG:    %[[BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH0]], %[[SPEC]], align 32 : !mgp.cpu.tensor_spec<?x?xf64>
// (Redundant) Chain merge to preserve control dependencies
// CHECK-DAG:    %[[BUF_ACTION_CH:.*]] = grt.chain.create ()
// Buffer caching
// CHECK-DAG:    %[[CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[BUF]] : !mgp.cpu.buffer
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH]])
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer) {
// Inserted buffer lookup
// CHECK-DAG:    %[[BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer
// CHECK-NEXT:   mef.output %[[CH0]], %[[BUF]] : !mef.chain, !mgp.cpu.buffer
// CHECK-NEXT: }

mgp.model @lift_non_trivial_buffer_op_without_liftable_user : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  // Don't bother setting up the compiled kernel context
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer) {
  %res:3 = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (), summary "foo" out_dims 2
  %spec = mgp.cpu.tensor_spec.create %res#1, %res#2 : !mgp.cpu.tensor_spec<?x?xf64>
  %buf = mgp.cpu.buffer.alloc.spec %ch0, %spec, align 32 : !mgp.cpu.tensor_spec<?x?xf64>
  mef.output %ch0, %buf : !mef.chain, !mgp.cpu.buffer
}

//
// @collect_write_chains_for_lifted_buffer
//

// CHECK-LABEL: mgp.model @collect_write_chains_for_lifted_buffer : mof init {
// Old chains and context ops
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 3
// Lifted ops
// CHECK-DAG:    %[[BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 100, align 16
// CHECK-DAG:    %[[OP_CH:.*]], %[[DIM:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (out %[[BUF]] : {{.*}}), summary "foo" out_dims 1 {numBackOutputBuffers = 1 : ui32}
// CHECK-DAG:    %[[OUT_SPEC:.*]] = mgp.cpu.tensor_spec.create %[[DIM]] : !mgp.cpu.tensor_spec<?xf64>
// CHECK-DAG:    %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH0]], %[[OUT_SPEC]], align 32 : !mgp.cpu.tensor_spec<?xf64>
// Chain merge to preserve control dependencies (NOTE that it took the kernel output chain into account)
// CHECK-DAG:    %[[BUF_ACTION_CH:.*]] = grt.chain.create (%[[OP_CH]])
// Buffer caching
// CHECK-DAG:    %[[CACHE_CH1:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[BUF]] : !mgp.cpu.buffer
// CHECK-DAG:    %[[CACHE_CH2:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][2], %[[OUT_BUF]] : !mgp.cpu.buffer
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[OUT_CH:.*]] = grt.chain.create (%[[CH0]], %[[CACHE_CH1]], %[[CACHE_CH2]])
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<100>, !mgp.cpu.buffer) {
// Inserted buffer lookup
// CHECK-DAG:    %[[BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer<100>
// CHECK-DAG:    %[[OUT_BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][2] : !mgp.cpu.buffer
// CHECK-NEXT:   mef.output %[[CH0]], %[[BUF]], %[[OUT_BUF]] : !mef.chain, !mgp.cpu.buffer<100>, !mgp.cpu.buffer
// CHECK-NEXT: }

mgp.model @collect_write_chains_for_lifted_buffer : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  // Don't bother setting up the compiled kernel context
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<100>, !mgp.cpu.buffer) {
  %buf = mgp.cpu.buffer.alloc.static %ch0, 100, align 16
  %res:2 = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (out %buf : !mgp.cpu.buffer<100>), summary "foo" out_dims 1 {numBackOutputBuffers = 1 : ui32}
  %out_spec = mgp.cpu.tensor_spec.create %res#1 : !mgp.cpu.tensor_spec<?xf64>
  %out_buf = mgp.cpu.buffer.alloc.spec %ch0, %out_spec, align 32 : !mgp.cpu.tensor_spec<?xf64>
  mef.output %ch0, %buf, %out_buf : !mef.chain, !mgp.cpu.buffer<100>, !mgp.cpu.buffer
}


//
// @dont_lift_if_mutated_by_unliftable
//

// CHECK-LABEL: mgp.model @dont_lift_if_mutated_by_unliftable : mof init {
//
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<100>, !mgp.cpu.buffer) {
// Nothing was lifted
// CHECK-DAG:    %[[BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 100, align 16
// CHECK-DAG:    %[[CH1:.*]], %[[DIM0:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][0, 0], (out %[[BUF]] : !mgp.cpu.buffer<100>), summary "foo" out_dims 1 {numBackOutputBuffers = 1 : ui32}
// CHECK-DAG:    %[[CH2:.*]] = "foo.side_effect_op"(%[[DIM0]]) : (index) -> !mef.chain
// CHECK-DAG:    %[[CH3:.*]], %[[DIM1:.*]] = mgp.cpu.kernel.execute %[[CH2]], %[[CTX]][0, 0], (out %[[BUF]] : !mgp.cpu.buffer<100>), summary "foo" out_dims 1 {numBackOutputBuffers = 1 : ui32}
// CHECK-DAG:    %[[OUT_SPEC:.*]] = mgp.cpu.tensor_spec.create %[[DIM1]] : !mgp.cpu.tensor_spec<?xf64>
// CHECK-DAG:    %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH0]], %[[OUT_SPEC]], align 32 : !mgp.cpu.tensor_spec<?xf64>
// CHECK-NEXT:   mef.output %[[CH0]], %[[BUF]], %[[OUT_BUF]] : !mef.chain, !mgp.cpu.buffer<100>, !mgp.cpu.buffer
// CHECK-NEXT: }

mgp.model @dont_lift_if_mutated_by_unliftable : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  // Don't bother setting up the compiled kernel context
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<100>, !mgp.cpu.buffer) {
  %buf = mgp.cpu.buffer.alloc.static %ch0, 100, align 16
  %ch1, %dim0 = mgp.cpu.kernel.execute %ch0, %ctx[0, 0], (out %buf : !mgp.cpu.buffer<100>), summary "foo" out_dims 1 {numBackOutputBuffers = 1 : ui32}
  %ch2 = "foo.side_effect_op"(%dim0) : (index) -> !mef.chain
  %ch3, %dim1 = mgp.cpu.kernel.execute %ch2, %ctx[0, 0], (out %buf : !mgp.cpu.buffer<100>), summary "foo" out_dims 1 {numBackOutputBuffers = 1 : ui32}
  %out_spec = mgp.cpu.tensor_spec.create %dim1 : !mgp.cpu.tensor_spec<?xf64>
  %out_buf = mgp.cpu.buffer.alloc.spec %ch0, %out_spec, align 32 : !mgp.cpu.tensor_spec<?xf64>
  mef.output %ch0, %buf, %out_buf : !mef.chain, !mgp.cpu.buffer<100>, !mgp.cpu.buffer
}

//
// @cloned_need_indirectly
//

// CHECK-LABEL: mgp.model @cloned_needed_indirectly : mof init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create ()
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 2
// CHECK-DAG:    %[[BUF0:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 2359296, align 64
// CHECK-DAG:    %[[CH1:.*]] = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][1, 0], (out %[[BUF0]] : !mgp.cpu.buffer<2359296>), summary "kernelA" {numBackOutputBuffers = 1 : ui32}
// CHECK-DAG:    %[[CH2:.*]], %[[DIMS:.*]]:2 = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][1, 1], (), summary "shapeFunction" out_dims 2 {numBackOutputBuffers = 0 : ui32}
// CHECK-DAG:    %[[SPEC:.*]] = mgp.cpu.tensor_spec.create %[[DIMS]]#0, %[[DIMS]]#1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:    %[[CH3:.*]] = grt.chain.create (%[[CH0]], %[[CH1]])
// CHECK-DAG:    %[[BUF1:.*]] = mgp.cpu.buffer.alloc.spec %[[CH3]], %[[SPEC]], align 64 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:    %[[CH5:.*]] = mgp.cpu.kernel.execute %[[CH1]], %[[CTX]][1, 2], (in %[[BUF0]] : !mgp.cpu.buffer<2359296>, out %[[BUF1]] : !mgp.cpu.buffer), (%outDimensions#0, %outDimensions#1), summary "kernelB" {numBackOutputBuffers = 1 : ui32}
// CHECK-DAG:    %[[CH6:.*]] = grt.chain.create (%[[CH5]])
// CHECK-DAG:    %[[CH4:.*]] = mgp.cpu.buffer.cache %[[CH6]], %[[CTX]][1], %[[BUF1]] : !mgp.cpu.buffer
// CHECK-DAG:    %[[CH7:.*]] = grt.chain.create (%[[CH0]], %[[CH4]])
// CHECK-NEXT:   mef.output %[[CH7]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// Though not directly needed by an unliftable op, this computation nevertheless must be cloned.
// CHECK-DAG:    %[[CH1:.*]], %[[DIMS:.*]]:2 = mgp.cpu.kernel.execute %[[CH0]], %[[CTX]][1, 1], (), summary "shapeFunction" out_dims 2 {numBackOutputBuffers = 0 : ui32}
// CHECK-DAG:    %[[SPEC:.*]] = mgp.cpu.tensor_spec.create %[[DIMS]]#0, %[[DIMS]]#1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:    %[[BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer
// CHECK-DAG:    %[[TEN:.*]] = mgp.cpu.tensor.create %[[CH0]], %[[BUF]], %[[SPEC]] : !mgp.cpu.tensor<?x?xf32>
// CHECK:        mef.output %[[CH0]], %[[TEN]] : !mef.chain, !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT: }

mgp.model @cloned_needed_indirectly : mof init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 1
  // Don't bother setting up the compiled kernel context
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
  %0 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<768x768xf32>
  %1 = mgp.cpu.buffer.alloc.static %arg0, 2359296, align 64
  %outChain = mgp.cpu.kernel.execute %arg0, %arg1[1, 0], (out %1 : !mgp.cpu.buffer<2359296>), summary "kernelA" {numBackOutputBuffers = 1 : ui32}
  %outChain_0, %outDimensions:2 = mgp.cpu.kernel.execute %arg0, %arg1[1, 1], (), summary "shapeFunction" out_dims 2 {numBackOutputBuffers = 0 : ui32}
  %2 = mgp.cpu.tensor_spec.create %outDimensions#0, %outDimensions#1 : !mgp.cpu.tensor_spec<?x?xf32>
  %3 = grt.chain.create (%arg0, %outChain)
  %4 = mgp.cpu.buffer.alloc.spec %3, %2, align 64 : !mgp.cpu.tensor_spec<?x?xf32>
  %outChain_1 = mgp.cpu.kernel.execute %outChain, %arg1[1, 2], (in %1 : !mgp.cpu.buffer<2359296>, out %4 : !mgp.cpu.buffer), (%outDimensions#0, %outDimensions#1), summary "kernelB" {numBackOutputBuffers = 1 : ui32}
  %5 = mgp.cpu.tensor.create %outChain_1, %4, %2 : !mgp.cpu.tensor<?x?xf32>
  %6 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<ui64>
  mef.output %arg0, %5 : !mef.chain, !mgp.cpu.tensor<?x?xf32>
}

//
// @while
//

// CHECK-LABEL: mgp.model @while : mof init {
//
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<32xui8>) {
// Nothing was lifted
// CHECK:   %{{.*}} = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00000004", align 8> : !M.array<4xui8>
// CHECK:   %{{.*}}:10 = grt.while {{.*}} {
// CHECK:     %{{.*}} = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>
// CHECK:   } do {
// CHECK:     %{{.*}} = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00000001", align 8> : !M.array<4xui8>
// CHECK:   }
// CHECK: }

mgp.model @while : mof init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<32xui8>) {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00000004", align 8> : !M.array<4xui8>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<ui32>
  %2 = mgp.cpu.buffer.alloc.static %arg0, 32
  %3 = mgp.cpu.buffer.set %2 : !mgp.cpu.buffer<32>
  %4 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<32xui8>
  %5 = mgp.cpu.buffer.alloc.static %arg0, 32
  %6 = mgp.cpu.buffer.set %5, 1 : !mgp.cpu.buffer<32>
  %7:10 = grt.while (%arg0 as %arg2: !mef.chain, %arg0 as %arg3: !mef.chain, %0 as %arg4: !mgp.cpu.buffer<4>, %1 as %arg5: !mgp.cpu.tensor_spec<ui32>, %3 as %arg6: !mef.chain, %2 as %arg7: !mgp.cpu.buffer<32>, %4 as %arg8: !mgp.cpu.tensor_spec<32xui8>, %6 as %arg9: !mef.chain, %5 as %arg10: !mgp.cpu.buffer<32>, %4 as %arg11: !mgp.cpu.tensor_spec<32xui8>) {
    %9 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00000000", align 8> : !M.array<4xui8>
    %10 = mgp.cpu.buffer.alloc.static %arg2, 1, align 32
    %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<bool>
    %12 = "mop.kernels.greater"(%arg3, %arg4, %arg5, %9, %arg5, %10, %11) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<1>, !mgp.cpu.tensor_spec<bool>) -> !mef.chain
    %13 = mgp.to_grt_bool(%12, %10) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
    mef.output %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %13 : !mef.chain, !mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>, !mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>, i1
  } do {
    %9 = mgp.cpu.buffer.alloc.static %arg2, 32
    %10 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<32xui8>
    %11 = grt.chain.create (%arg6, %arg9)
    %12 = "mop.kernels.add"(%11, %arg7, %arg8, %arg10, %arg11, %9, %10) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mef.chain
    %13 = "some.op"(%12, %9, %10) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
    %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %13 : !mgp.cpu.tensor<32xui8>
    %14 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00000001", align 8> : !M.array<4xui8>
    %15 = mgp.cpu.buffer.alloc.static %arg2, 4, align 32
    %16 = "mop.kernels.sub"(%arg3, %arg4, %arg5, %14, %arg5, %15, %arg5) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>) -> !mef.chain
    mef.output %arg2, %16, %15, %arg5, %arg2, %bufferResult, %tensorSpecResult, %arg9, %arg10, %arg11 : !mef.chain, !mef.chain, !mgp.cpu.buffer<4>, !mgp.cpu.tensor_spec<ui32>, !mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>, !mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>
  }
  %8 = mgp.cpu.tensor.create %7#4, %7#5, %7#6 : !mgp.cpu.tensor<32xui8>
  mef.output %7#0, %8 : !mef.chain, !mgp.cpu.tensor<32xui8>
}
