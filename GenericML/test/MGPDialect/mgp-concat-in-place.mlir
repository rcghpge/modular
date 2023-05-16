// RUN: mgp-opt %s --allow-unregistered-dialect --concat-in-place | FileCheck %s

//
// @static_concat
//

// CHECK-LABEL: mgp.model @static_concat
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[MGP_CTX:.*]]: !mgp.context)
// CHECK: %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 64
// CHECK: %[[IN_BUFS:.*]]:2 = mgp.cpu.buffer.split.borrow %[[OUT_BUF]] : !mgp.cpu.buffer<64>  -> (!mgp.cpu.buffer<32>, !mgp.cpu.buffer<32>)
// CHECK: %[[CH1:.*]] = "mop.kernels.add"(%[[CH0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUFS]]#0, %{{.*}})
// CHECK: %[[CH2:.*]] = "mop.kernels.sub"(%[[CH0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUFS]]#1, %{{.*}})
// CHECK: %[[CH3:.*]] = grt.chain.merge (%[[CH1]], %[[CH2]])
// CHECK: %[[OUT_TENSOR:.*]] = mgp.cpu.tensor.create %[[CH3]], %[[OUT_BUF]], %{{.*}}

mgp.model @static_concat : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %mgp_ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<16xf32>) {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %2 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %3 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %4 = mgp.cpu.buffer.alloc.static %ch0, 32
  %5 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %6 = "mop.kernels.add"(%ch0, %0, %1, %2, %3, %4, %5) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %10 = mgp.cpu.buffer.alloc.static %ch0, 32
  %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %12 = "mop.kernels.sub"(%ch0, %0, %1, %2, %3, %10, %11) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %14 = mgp.cpu.buffer.alloc.static %ch0, 64
  %15 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<16xf32>
  %16 = grt.chain.merge (%6, %12)
  %17 = mgp.cpu.buffer.concat %16, %14, %4, %10 : (!mgp.cpu.buffer<32>, !mgp.cpu.buffer<32>) -> !mgp.cpu.buffer<64>
  %18 = mgp.cpu.tensor.create %17, %14, %15 : !mgp.cpu.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.cpu.tensor<16xf32>
}

//
// @dynamic_concat
//

// CHECK-LABEL: mgp.model @dynamic_concat
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[MGP_CTX:.*]]: !mgp.context)
// *byte size of first result of split, ie 32*
// CHECK: %[[SPLIT_SIZE:.*]] = mgp.cpu.tensor_spec.size %{{.*}}
// *dimension of first tensor, ie 8, not significant in our test*
// CHECK: %{{.*}} = mgp.cpu.tensor_spec.size %{{.*}}
// *byte size of buffer we are splitting, ie 64*
// CHECK: %[[TOTAL_SIZE:.*]] = mgp.cpu.tensor_spec.size %{{.*}}
// CHECK: %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.dynamic %[[CH0]], %[[TOTAL_SIZE]]
// CHECK: %[[IN_BUFS:.*]]:2 = mgp.cpu.buffer.split.borrow %[[OUT_BUF]], %[[SPLIT_SIZE]] : !mgp.cpu.buffer  -> (!mgp.cpu.buffer, !mgp.cpu.buffer<32>)
// CHECK: %[[CH1:.*]] = "mop.kernels.add"(%[[CH0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUFS]]#0, %{{.*}})
// CHECK: %[[CH2:.*]] = "mop.kernels.sub"(%[[CH0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUFS]]#1, %{{.*}})
// CHECK: %[[CH3:.*]] = grt.chain.merge (%[[CH1]], %[[CH2]])
// CHECK: %[[OUT_TENSOR:.*]] = mgp.cpu.tensor.create %[[CH3]], %[[OUT_BUF]], %{{.*}}

mgp.model @dynamic_concat : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %mgp_ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<16xf32>) {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %2 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %3 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %const_32 = mgp.cpu.tensor_spec.size %3 : !mgp.cpu.tensor_spec<8xf32>
  %4 = mgp.cpu.buffer.alloc.dynamic %ch0, %const_32
  %dummy = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xui8>
  %const_8 = mgp.cpu.tensor_spec.size %dummy : !mgp.cpu.tensor_spec<8xui8>
  %5 = mgp.cpu.tensor_spec.create %const_8 : !mgp.cpu.tensor_spec<?xf32>
  %6 = "mop.kernels.add"(%ch0, %0, %1, %2, %3, %4, %5) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?xf32>) -> !mef.chain
  %10 = mgp.cpu.buffer.alloc.static %ch0, 32
  %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %12 = "mop.kernels.sub"(%ch0, %0, %1, %2, %3, %10, %11) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %14 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<16xf32>
  %const_64 = mgp.cpu.tensor_spec.size %14 : !mgp.cpu.tensor_spec<16xf32>
  %15 = mgp.cpu.buffer.alloc.dynamic %ch0, %const_64
  %16 = grt.chain.merge (%6, %12)
  %17 = mgp.cpu.buffer.concat %16, %15, %4, %10 : (!mgp.cpu.buffer, !mgp.cpu.buffer<32>) -> !mgp.cpu.buffer
  %18 = mgp.cpu.tensor.create %17, %15, %14 !mgp.cpu.buffer : !mgp.cpu.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.cpu.tensor<16xf32>
}

//
// @concat_with_shared_input_buffer
//

// CHECK-LABEL: mgp.model @concat_with_shared_input_buffer
// CHECK: %{{.*}} = mgp.cpu.buffer.split.borrow

mgp.model @concat_with_shared_input_buffer : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %mgp_ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<16xf32>) {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %2 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %3 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %4 = mgp.cpu.buffer.alloc.static %ch0, 32
  %5 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %6 = "mop.kernels.add"(%ch0, %0, %1, %2, %3, %4, %5) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %10 = mgp.cpu.buffer.alloc.static %ch0, 32
  %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %12 = "mop.kernels.sub"(%6, %4, %5, %2, %3, %10, %11) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %14 = mgp.cpu.buffer.alloc.static %ch0, 64
  %15 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<16xf32>
  %16 = grt.chain.merge (%6, %12)
  %17 = mgp.cpu.buffer.concat %16, %14, %4, %10 : (!mgp.cpu.buffer<32>, !mgp.cpu.buffer<32>) -> !mgp.cpu.buffer<64>
  %18 = mgp.cpu.tensor.create %17, %14, %15 : !mgp.cpu.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.cpu.tensor<16xf32>
}

//
// @dont_trigger_since_non_triv_chains
//

// CHECK-LABEL: mgp.model @dont_trigger_since_non_triv_chains
// CHECK: %{{.*}} = mgp.cpu.buffer.concat %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}

mgp.model @dont_trigger_since_non_triv_chains : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %mgp_ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<16xf32>) {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %2 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %3 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %4 = mgp.cpu.buffer.alloc.static %ch0, 32
  %5 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %6 = "mop.kernels.add"(%ch0, %0, %1, %2, %3, %4, %5) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %10 = mgp.cpu.buffer.alloc.static %6, 32
  %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %12 = "mop.kernels.sub"(%6, %0, %1, %2, %3, %10, %11) {numBacOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %14 = mgp.cpu.buffer.alloc.static %12, 64
  %15 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<16xf32>
  %17 = mgp.cpu.buffer.concat %12, %14, %4, %10 : (!mgp.cpu.buffer<32>, !mgp.cpu.buffer<32>) -> !mgp.cpu.buffer<64>
  %18 = mgp.cpu.tensor.create %17, %14, %15 : !mgp.cpu.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.cpu.tensor<16xf32>
}
