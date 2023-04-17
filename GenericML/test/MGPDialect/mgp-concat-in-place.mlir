// RUN: mgp-opt %s --allow-unregistered-dialect --concat-in-place | FileCheck %s

//
// @static_concat
//

// CHECK-LABEL: mgp.model @static_concat
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[MGP_CTX:.*]]: !mgp.context)
// CHECK: %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 64
// CHECK: %[[IN_BUFS:.*]]:2 = mgp.cpu.buffer.split.mutable_borrow %[[OUT_BUF]] : !mgp.cpu.buffer<64>  -> (!mgp.cpu.buffer<32>, !mgp.cpu.buffer<32>)
// CHECK: %[[CH1:.*]] = "mop.kernels.add"(%[[CH0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUFS]]#0, %{{.*}})
// CHECK: %[[CH2:.*]] = "mop.kernels.sub"(%[[CH0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUFS]]#1, %{{.*}})
// CHECK: %[[CH3:.*]] = grt.chain.merge %[[CH1]], %[[CH2]]
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
  %16 = grt.chain.merge %6, %12
  %17 = mgp.cpu.buffer.concat %16, %14, %4, %10 : (!mgp.cpu.buffer<32>, !mgp.cpu.buffer<32>) -> !mgp.cpu.buffer<64>
  %18 = mgp.cpu.tensor.create %17, %14, %15 : !mgp.cpu.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.cpu.tensor<16xf32>
}

//
// @dynamic_concat
//

// CHECK-LABEL: mgp.model @dynamic_concat
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[MGP_CTX:.*]]: !mgp.context)
// CHECK: %[[SIZE:.*]] = mgp.cpu.tensor_spec.size %{{.*}}
// CHECK: %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 64
// CHECK: %[[IN_BUFS:.*]]:2 = mgp.cpu.buffer.split.mutable_borrow %[[OUT_BUF]], %[[SIZE]] : !mgp.cpu.buffer<64>  -> (!mgp.cpu.buffer, !mgp.cpu.buffer<32>)
// CHECK: %[[CH1:.*]] = "mop.kernels.add"(%[[CH0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUFS]]#0, %{{.*}})
// CHECK: %[[CH2:.*]] = "mop.kernels.sub"(%[[CH0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUFS]]#1, %{{.*}})
// CHECK: %[[CH3:.*]] = grt.chain.merge %[[CH1]], %[[CH2]]
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
  %size = mgp.cpu.tensor_spec.size %3 : !mgp.cpu.tensor_spec<8xf32>
  %4 = mgp.cpu.buffer.alloc.dynamic %ch0, %size
  %5 = mgp.cpu.tensor_spec.create %size : !mgp.cpu.tensor_spec<?xf32>
  %6 = "mop.kernels.add"(%ch0, %0, %1, %2, %3, %4, %5) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer, !mgp.cpu.tensor_spec<?xf32>) -> !mef.chain
  %10 = mgp.cpu.buffer.alloc.static %ch0, 32
  %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %12 = "mop.kernels.sub"(%ch0, %0, %1, %2, %3, %10, %11) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %14 = mgp.cpu.buffer.alloc.static %ch0, 64
  %15 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<16xf32>
  %16 = grt.chain.merge %6, %12
  %17 = mgp.cpu.buffer.concat %16, %14, %4, %10 : (!mgp.cpu.buffer, !mgp.cpu.buffer<32>) -> !mgp.cpu.buffer<64>
  %18 = mgp.cpu.tensor.create %17, %14, %15 : !mgp.cpu.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.cpu.tensor<16xf32>
}

//
// @dont_trigger_since_shared
//

// CHECK-LABEL: mgp.model @dont_trigger_since_shared
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[MGP_CTX:.*]]: !mgp.context)
// CHECK: %[[IN_BUF0:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 32
// CHECK: %[[CH1:.*]] = "mop.kernels.add"(%[[CH0]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUF0]], %{{.*}})
// CHECK: %[[IN_BUF1:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 32
// CHECK: %[[CH2:.*]] = "mop.kernels.sub"(%[[CH1]], %[[IN_BUF0]], %{{.*}}, %{{.*}}, %{{.*}}, %[[IN_BUF1]], %{{.*}})
// CHECK: %[[OUT_BUF:.*]] = mgp.cpu.buffer.alloc.static %[[CH0]], 64
// CHECK: %[[CH3:.*]] = grt.chain.merge %[[CH1]], %[[CH2]]
// CHECK: %[[CH4:.*]] = mgp.cpu.buffer.concat %[[CH3]], %[[OUT_BUF]], %[[IN_BUF0]], %[[IN_BUF1]]
// CHECK: %[[OUT_TENSOR:.*]] = mgp.cpu.tensor.create %[[CH4]], %[[OUT_BUF]], %{{.*}}

mgp.model @dont_trigger_since_shared : tf init {
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
  %16 = grt.chain.merge %6, %12
  %17 = mgp.cpu.buffer.concat %16, %14, %4, %10 : (!mgp.cpu.buffer<32>, !mgp.cpu.buffer<32>) -> !mgp.cpu.buffer<64>
  %18 = mgp.cpu.tensor.create %17, %14, %15 : !mgp.cpu.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.cpu.tensor<16xf32>
}
