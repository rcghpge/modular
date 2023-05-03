// RUN: mgp-opt %s --allow-unregistered-dialect --exec-invariant-code-motion | FileCheck %s

//
// @kgen_execute_kernel
//

// CHECK-LABEL: mgp.model @kgen_execute_kernel : tf init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 2
// Dummy chain
// CHECK-NEXT:   %{{.*}} = grt.chain.create
// Lifted ops
// CHECK-NEXT:   %[[PERM_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16>
// CHECK-NEXT:   %[[WEIGHT_BUF:.*]] = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
// CHECK-NEXT:   %[[PERM_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<4xsi64>
// CHECK-NEXT:   %[[WEIGHT_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<64x3x7x7xf32>
// CHECK-DAG:    %[[OUT_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<7x7x3x64xf32>
// CHECK-DAG:    %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 37632, align 16
// CHECK-DAG:    %[[TRANSPOSE_CH:.*]] = "mop.kgen.execute_kernel"(%[[CH0]], %[[WEIGHT_BUF]], %[[WEIGHT_SPEC]], %[[PERM_BUF]], %[[PERM_SPEC]], %[[OUT_BUF]], %[[OUT_SPEC]]) : (!mef.chain, !mgp.cpu.buffer<37632>, !mgp.cpu.tensor_spec<64x3x7x7xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<4xsi64>, !mgp.cpu.buffer<37632>, !mgp.cpu.tensor_spec<7x7x3x64xf32>) -> !mef.chain
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[BUF_ACTION_CH:.*]] = grt.chain.merge %[[TRANSPOSE_CH]]
// Buffer caching
// CHECK-DAG:    %[[CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[OUT_BUF]] : !mgp.cpu.buffer<37632>
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[OUT_CH:.*]] = grt.chain.merge %[[CH0]], %[[CACHE_CH]]
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> !mef.chain {
// Inserted buffer lookup
// CHECK-NEXT:   %[[BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer<37632>
// Reference to lifted chain got updated to input chain
// CHECK-NEXT:   mef.output %[[CH0]] : !mef.chain
// CHECK-NEXT: }

mgp.model @kgen_execute_kernel : tf init {
  %ch0 = grt.chain.create
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain) {
  %perm_buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16>
  %weight_buf = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
  %perm_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<4xsi64>
  %weight_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<64x3x7x7xf32>
  %out_buf = mgp.cpu.buffer.alloc.static %ch0, 37632, align 16
  %out_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<7x7x3x64xf32>
  %out_ch = "mop.kgen.execute_kernel"(%ch0, %weight_buf, %weight_spec, %perm_buf, %perm_spec, %out_buf, %out_spec) : (!mef.chain, !mgp.cpu.buffer<37632>, !mgp.cpu.tensor_spec<64x3x7x7xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<4xsi64>, !mgp.cpu.buffer<37632>, !mgp.cpu.tensor_spec<7x7x3x64xf32>) -> !mef.chain
  mef.output %out_ch : !mef.chain
}

//
// @matmul_packb
//

// CHECK-LABEL: mgp.model @matmul_packb : tf init {
// CHECK-NEXT:  %[[CH0:.*]] = grt.chain.create
// CHECK-NEXT:  %[[CTX:.*]] = mgp.context.create 2
// CHECK-NEXT:  %[[CH1:.*]] = "mop.create_jit_context"(%[[CH0]], %[[CTX]]) {slot = 0 : ui32} : (!mef.chain, !mgp.context) -> !mef.chain
// Dummy chain
// CHECK-NEXT:  %{{.*}} = grt.chain.create
// Lifted ops
// CHECK-NEXT:  %[[CONST_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
// CHECK-NEXT:  %[[CONST_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2x2xf32>
// CHECK-NEXT:  %[[PACKED_SPEC:.*]] = "mop.shape.pack_matmul_b"(%[[CONST_SPEC]]) : (!mgp.cpu.tensor_spec<2x2xf32>) -> !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:  %[[PACKED_BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH1]], %[[PACKED_SPEC]], align 16 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-DAG:  %[[PACK_CH:.*]] = "mop.kernels.pack_matmul_b"(%[[CH1]], %[[CONST_BUF]], %[[CONST_SPEC]], %[[PACKED_BUF]], %[[PACKED_SPEC]]) : (!mef.chain, !mgp.cpu.buffer<16>, !mgp.cpu.tensor_spec<2x2xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x?xf32>) -> !mef.chain
// Chain merge to preserve control dependencies
// CHECK-DAG:  %[[BUF_ACTION_CH:.*]] = grt.chain.merge %[[PACK_CH]]
// Buffer caching
// CHECK-DAG:  %[[CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[PACKED_BUF]] : !mgp.cpu.buffer
// Chain merge to preserve control dependencies
// CHECK-DAG:  %[[OUT_CH:.*]] = grt.chain.merge %[[CH1]], %[[CACHE_CH]]
// CHECK-DAG:  mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT:} execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x2xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x2xf32>) {
// Inserted recomputation
// CHECK-NEXT:  %[[CONST_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2x2xf32>
// CHECK-NEXT:  %[[PACKED_SPEC:.*]] = "mop.shape.pack_matmul_b"(%[[CONST_SPEC]]) : (!mgp.cpu.tensor_spec<2x2xf32>) -> !mgp.cpu.tensor_spec<?x?xf32>
// Inserted buffer lookup
// CHECK-NEXT:  %[[PACKED_BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer
// Ops that didn't get lifted, but references to chain and buffer got updated.
// CHECK-NEXT:  %[[ARG2_BUF:.*]], %[[ARG2_SPEC:.*]] = mgp.cpu.tensor.extract %[[ARG2]] : !mgp.cpu.tensor<?x2xf32>
// CHECK-NEXT:  %[[OUT_SPEC:.*]] = "mop.shape.matmul_packb"(%[[ARG2_SPEC]]) : (!mgp.cpu.tensor_spec<?x2xf32>) -> !mgp.cpu.tensor_spec<?x2xf32>
// CHECK-NEXT:  %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH0]], %[[OUT_SPEC]], align 16 : !mgp.cpu.tensor_spec<?x2xf32>
// CHECK-NEXT:  %[[CH2:.*]] = "mop.kernels.matmul_packb"(%[[CH0]], %[[ARG2_BUF]], %[[ARG2_SPEC]], %[[PACKED_BUF]], %[[PACKED_SPEC]], %[[OUT_BUF]], %[[OUT_SPEC]])
// CHECK-SAME:    : (!mef.chain, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x2xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x?xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x2xf32>) -> !mef.chain
// CHECK-NEXT:  %[[OUT_TENSOR:.*]] = mgp.cpu.tensor.create %[[CH2]], %[[OUT_BUF]], %[[OUT_SPEC]]  : !mgp.cpu.tensor<?x2xf32>
// CHECK-NEXT:  mef.output %[[CH0]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<?x2xf32>
// CHECK-NEXT:}

mgp.model @matmul_packb : tf init {
  %ch0 = grt.chain.create
  %ctx = mgp.context.create 1
  %ch1 = "mop.create_jit_context"(%ch0, %ctx) {slot = 0 : ui32} : (!mef.chain, !mgp.context) -> !mef.chain
  mef.output %ch1, %ctx : !mef.chain, !mgp.context

} execute (%ch0: !mef.chain, %ctx: !mgp.context, %arg2: !mgp.cpu.tensor<?x2xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x2xf32>) {
  %const_buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
  %const_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2x2xf32>
  %packed_spec = "mop.shape.pack_matmul_b"(%const_spec) : (!mgp.cpu.tensor_spec<2x2xf32>) -> !mgp.cpu.tensor_spec<?x?xf32>
  %packed_buf = mgp.cpu.buffer.alloc.spec %ch0, %packed_spec, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
  %ch1 = "mop.kernels.pack_matmul_b"(%ch0, %const_buf, %const_spec, %packed_buf, %packed_spec) : (!mef.chain, !mgp.cpu.buffer<16>, !mgp.cpu.tensor_spec<2x2xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x?xf32>) -> !mef.chain

  %arg2_buf, %arg2_spec = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x2xf32>
  %out_spec = "mop.shape.matmul_packb"(%arg2_spec) : (!mgp.cpu.tensor_spec<?x2xf32>) -> !mgp.cpu.tensor_spec<?x2xf32>
  %out_buf = mgp.cpu.buffer.alloc.spec %ch1, %out_spec, align 16 : !mgp.cpu.tensor_spec<?x2xf32>
  %ch2 = "mop.kernels.matmul_packb"(%ch1, %arg2_buf, %arg2_spec, %packed_buf, %packed_spec, %out_buf, %out_spec)
  : (!mef.chain, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x2xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x?xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x2xf32>) -> !mef.chain
  %out_tensor = mgp.cpu.tensor.create %ch2, %out_buf, %out_spec  : !mgp.cpu.tensor<?x2xf32>
  mef.output %ch0, %out_tensor : !mef.chain, !mgp.cpu.tensor<?x2xf32>
}

//
// @non_pure_op
//

// CHECK-LABEL: mgp.model @non_pure_op : tf init {
// CHECK-NEXT:  %[[CH0:.*]] = grt.chain.create
// CHECK-NEXT:  %[[CTX:.*]] = mgp.context.create 1
// CHECK-NEXT:  %[[CH1:.*]] = "mop.create_jit_context"(%[[CH0]], %[[CTX]]) {slot = 0 : ui32} : (!mef.chain, !mgp.context) -> !mef.chain
// Dummy chain
// CHECK-NEXT:  %{{.*}} = grt.chain.create
// Redundant lift, will be cleaned by canonicalization
// CHECK-NEXT:  %{{.*}} = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2x2xf32>
// Trivial chain merge to preserve control dependencies
// CHECK-NEXT:  %[[BUF_ACTION_CH:.*]] = grt.chain.merge
// Trivial chain merge to preserve control dependencies
// CHECK-NEXT:  %[[OUT_CH:.*]] = grt.chain.merge %[[CH1]]
// CHECK-NEXT:  mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
//
// CHECK-NEXT:} execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// No op got lifted/recomputed because they have side effects.
// CHECK-NEXT:  %[[CONST_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000803F0000803F0000803F", align 16>
// CHECK-NEXT:  %[[CONST_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<2x2xf32>
// CHECK-NEXT:  %[[OUT_SPEC:.*]] = "foo.side_effect_op"(%[[CONST_SPEC]]) : (!mgp.cpu.tensor_spec<2x2xf32>) -> !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:  %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.spec %[[CH0]], %[[OUT_SPEC]], align 16 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:  %[[PACK_CH:.*]] = "bar.side_effect_op"(%[[CH0]], %[[CONST_BUF]], %[[CONST_SPEC]], %[[OUT_BUF]], %[[OUT_SPEC]]) : (!mef.chain, !mgp.cpu.buffer<16>, !mgp.cpu.tensor_spec<2x2xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?x?xf32>) -> !mef.chain
// CHECK-NEXT:  %[[OUT_TENSOR:.*]] = mgp.cpu.tensor.create %[[PACK_CH]], %[[OUT_BUF]], %[[OUT_SPEC]]  : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:  mef.output %[[CH0]], %[[OUT_TENSOR]] : !mef.chain, !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:}

mgp.model @non_pure_op : tf init {
  %ch0 = grt.chain.create
  %ctx = mgp.context.create 1
  %ch1 = "mop.create_jit_context"(%ch0, %ctx) {slot = 0 : ui32} : (!mef.chain, !mgp.context) -> !mef.chain
  mef.output %ch1, %ctx : !mef.chain, !mgp.context

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

// CHECK-LABEL: mgp.model @weight_transpose : tf init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 2
// Dummy chain
// CHECK-NEXT:   %{{.*}} = grt.chain.create
// Lifted ops
// CHECK-NEXT:   %[[PERM_BUF:.*]] = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16>
// CHECK-NEXT:   %[[WEIGHT_BUF:.*]] = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
// CHECK-NEXT:   %[[PERM_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<4xsi64>
// CHECK-NEXT:   %[[WEIGHT_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<64x3x7x7xf32>
// CHECK-DAG:    %[[OUT_SPEC:.*]] = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<7x7x3x64xf32>
// CHECK-DAG:    %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 37632, align 16
// CHECK-DAG:    %[[TRANSPOSE_CH:.*]] = "mop.kernels.transpose"(%[[CH0]], %[[WEIGHT_BUF]], %[[WEIGHT_SPEC]], %[[PERM_BUF]], %[[PERM_SPEC]], %[[OUT_BUF]], %[[OUT_SPEC]]) : (!mef.chain, !mgp.cpu.buffer<37632>, !mgp.cpu.tensor_spec<64x3x7x7xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<4xsi64>, !mgp.cpu.buffer<37632>, !mgp.cpu.tensor_spec<7x7x3x64xf32>) -> !mef.chain
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[BUF_ACTION_CH:.*]] = grt.chain.merge %[[TRANSPOSE_CH]]
// Buffer caching
// CHECK-DAG:    %[[CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[OUT_BUF]] : !mgp.cpu.buffer<37632>
// Chain merge to preserve control dependencies
// CHECK-DAG:    %[[OUT_CH:.*]] = grt.chain.merge %[[CH0]], %[[CACHE_CH]]
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> !mef.chain {
// Inserted buffer lookup
// CHECK-NEXT:   %[[BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer<37632>
// Reference to lifted chain got updated to input chain
// CHECK-NEXT:   mef.output %[[CH0]] : !mef.chain
// CHECK-NEXT: }

mgp.model @weight_transpose : tf init {
  %ch0 = grt.chain.create
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain) {
  %perm_buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0200000000000000030000000000000001000000000000000000000000000000", align 16>
  %weight_buf = mgp.cpu.buffer.constant dense_resource<onnx__Conv_497> : !M.aligned_bytes<37632, align 16>
  %perm_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<4xsi64>
  %weight_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<64x3x7x7xf32>
  %out_buf = mgp.cpu.buffer.alloc.static %ch0, 37632, align 16
  %out_spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<7x7x3x64xf32>
  %out_ch = "mop.kernels.transpose"(%ch0, %weight_buf, %weight_spec, %perm_buf, %perm_spec, %out_buf, %out_spec) : (!mef.chain, !mgp.cpu.buffer<37632>, !mgp.cpu.tensor_spec<64x3x7x7xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<4xsi64>, !mgp.cpu.buffer<37632>, !mgp.cpu.tensor_spec<7x7x3x64xf32>) -> !mef.chain
  mef.output %out_ch : !mef.chain
}

//
// @zero_size_buf
//

// CHECK-LABEL: mgp.model @zero_size_buf  : tf init {
// CHECK-NEXT:   %[[CH0:.*]] = grt.chain.create
// CHECK-NEXT:   %[[CTX:.*]] = mgp.context.create 2
// Dummy chain
// CHECK-NEXT:   %{{.*}} = grt.chain.create
// Lifted ops
// CHECK-DAG:    %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 0, align 16
// Chain merge to preserve control dependencies (will be cleared by DCE)
// CHECK-DAG:    %[[BUF_ACTION_CH:.*]] = grt.chain.merge
// Buffer caching
// CHECK-DAG:    %[[CACHE_CH:.*]] = mgp.cpu.buffer.cache %[[BUF_ACTION_CH]], %[[CTX]][1], %[[OUT_BUF]] : !mgp.cpu.buffer<0>
// Chain merge to preserve control dependencies
// CHECK-NEXT:   %[[OUT_CH:.*]] = grt.chain.merge %[[CH0]], %[[CACHE_CH]]
// CHECK-NEXT:   mef.output %[[OUT_CH]], %[[CTX]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<0>) {
// Zero-sized buffer isn't lifted because we can't cache it
// CHECK-NEXT:   %[[OUT_BUF:.*]] = mgp.cpu.buffer.get_cached %[[CH0]], %[[CTX]][1] : !mgp.cpu.buffer<0>
// CHECK-NEXT:   mef.output %[[CH0]], %[[OUT_BUF]] : !mef.chain, !mgp.cpu.buffer<0>
// CHECK-NEXT: }

mgp.model @zero_size_buf : tf init {
  %ch0 = grt.chain.create
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.buffer<0>) {
  %out_buf = mgp.cpu.buffer.alloc.static %ch0, 0, align 16
  mef.output %ch0, %out_buf : !mef.chain, !mgp.cpu.buffer<0>
}
