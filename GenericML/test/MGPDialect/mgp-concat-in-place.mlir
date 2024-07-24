// RUN: mo-opt %s --allow-unregistered-dialect --concat-in-place --canonicalize --cse | FileCheck %s

//
// @static_concat
//

// CHECK-LABEL: mgp.model @static_concat
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK: %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[OUT_BUF:.*]] = mgp.buffer.alloc.static %[[CH1]], %[[CTX]][1] : !mgp.buffer<64>
// CHECK: %[[IN_BUFS:.*]]:2 = mgp.buffer.split.borrow %[[CH1]], %[[CTX]][1], %[[OUT_BUF]] : !mgp.buffer<64>  -> (!mgp.buffer<32>, !mgp.buffer<32>)
// CHECK: %[[CH2:.*]] = "add"(%[[CH1]], %{{.*}}, %{{.*}}, %[[IN_BUFS]]#0)
// CHECK: %[[CH3:.*]] = "sub"(%[[CH1]], %{{.*}}, %{{.*}}, %[[IN_BUFS]]#1)
// CHECK: %[[CH4:.*]] = mgp.chain.create %[[CTX]][1], (%[[CH2]], %[[CH3]])
// CHECK: %[[TEN:.*]] = mgp.tensor.create %[[CH4]], %[[CTX]][1], %[[OUT_BUF]], %{{.*}}
// CHECK: mef.output %[[CH0]], %[[TEN]]

mgp.model @static_concat : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<16xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %2 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %4 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %6 = "add"(%ch1, %0, %2, %4) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %10 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %12 = "sub"(%ch1, %0, %2, %10) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %14 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<64>
  %15 = mgp.tensor_spec.create : !mgp.tensor_spec<16xf32>
  %16 = mgp.chain.create %ctx[1], (%6, %12)
  %17 = mgp.buffer.concat %16, %ctx[1], %14, %4, %10 : (!mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.buffer<64>
  %18 = mgp.tensor.create %17, %ctx[1], %14, %15 : !mgp.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.tensor<16xf32>
}

//
// @dynamic_concat
//

// CHECK-LABEL: mgp.model @dynamic_concat
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context
// CHECK: %[[TOTAL_SIZE:.*]] = mip.constant.index 64
// CHECK: %[[SPLIT_SIZE:.*]] = mip.constant.index 32
// CHECK: %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[OUT_BUF:.*]] = mgp.buffer.alloc.dynamic %[[CH1]], %[[CTX]][1], %[[TOTAL_SIZE]]
// CHECK: %[[IN_BUFS:.*]]:2 = mgp.buffer.split.borrow %[[CH1]], %[[CTX]][1], %[[OUT_BUF]], %[[SPLIT_SIZE]] : !mgp.buffer<>  -> (!mgp.buffer<>, !mgp.buffer<32>)
// CHECK: %[[CH2:.*]] = "add"(%[[CH1]], %{{.*}}, %{{.*}}, %[[IN_BUFS]]#0)
// CHECK: %[[CH3:.*]] = "sub"(%[[CH1]], %{{.*}}, %{{.*}}, %[[IN_BUFS]]#1)
// CHECK: %[[CH4:.*]] = mgp.chain.create %[[CTX]][1], (%[[CH2]], %[[CH3]])
// CHECK: %[[RESIZED_BUF:.*]] = mgp.buffer.resize %[[OUT_BUF]] : !mgp.buffer<64>
// CHECK: %[[TEN:.*]] = mgp.tensor.create %[[CH4]], %[[CTX]][1], %[[RESIZED_BUF]], %{{.*}}
// CHECK: mef.output %[[CH0]], %[[TEN]]

mgp.model @dynamic_concat : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<16xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %2 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %const_32 = mip.constant.index 32
  %4 = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %const_32 : !mgp.buffer<>
  %const_8 = mip.constant.index 8
  %5 = mgp.tensor_spec.create %const_8 : !mgp.tensor_spec<?xf32>
  %6 = "add"(%ch1, %0, %2, %4) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<>) -> !mgp.chain
  %10 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %12 = "sub"(%ch1, %0, %2, %10) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %14 = mgp.tensor_spec.create : !mgp.tensor_spec<16xf32>
  %const_64 = mip.constant.index 64
  %15 = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %const_64 : !mgp.buffer<>
  %16 = mgp.chain.create %ctx[1], (%6, %12)
  %17 = mgp.buffer.concat %16, %ctx[1], %15, %4, %10 : (!mgp.buffer<>, !mgp.buffer<32>) -> !mgp.buffer<>
  %resized_buf = mgp.buffer.resize %15 : !mgp.buffer<64>
  %18 = mgp.tensor.create %17, %ctx[1], %resized_buf, %14 : !mgp.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.tensor<16xf32>
}

//
// @concat_with_shared_input_buffer
//

// CHECK-LABEL: mgp.model @concat_with_shared_input_buffer
// CHECK: %{{.*}} = mgp.buffer.split.borrow

mgp.model @concat_with_shared_input_buffer : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<16xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %2 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %4 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %6 = "add"(%ch1, %0, %2, %4) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %10 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %12 = "sub"(%6, %4, %2, %10) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %14 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<64>
  %15 = mgp.tensor_spec.create : !mgp.tensor_spec<16xf32>
  %16 = mgp.chain.create %ctx[1], (%6, %12)
  %17 = mgp.buffer.concat %16, %ctx[1], %14, %4, %10 : (!mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.buffer<64>
  %18 = mgp.tensor.create %17, %ctx[1], %14, %15 : !mgp.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.tensor<16xf32>
}

//
// @non_trivial_chains
//

// CHECK-LABEL: mgp.model @non_trivial_chains
// CHECK: %{{.*}} = mgp.buffer.split.borrow

mgp.model @non_trivial_chains : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<16xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %2 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %4 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %6 = "add"(%ch1, %0, %2, %4) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %10 = mgp.buffer.alloc.static %6, %ctx[1] : !mgp.buffer<32>
  %12 = "sub"(%6, %0, %2, %10) {numBacOutputBuffers = 1 : ui32} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %14 = mgp.buffer.alloc.static %12, %ctx[1] : !mgp.buffer<64>
  %15 = mgp.tensor_spec.create : !mgp.tensor_spec<16xf32>
  %17 = mgp.buffer.concat %12, %ctx[1], %14, %4, %10 : (!mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.buffer<64>
  %18 = mgp.tensor.create %17, %ctx[1], %14, %15 : !mgp.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.tensor<16xf32>
}

//
// @dont_trigger_since_duplicate_buffers
//

// CHECK-LABEL: mgp.model @dont_trigger_since_duplicate_buffers
// CHECK: %{{.*}} = mgp.buffer.concat

mgp.model @dont_trigger_since_duplicate_buffers : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque) -> (!mef.chain, !mgp.tensor<16xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
  %2 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
  %4 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %6 = "add"(%ch1, %0, %2, %4) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %14 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<64>
  %15 = mgp.tensor_spec.create : !mgp.tensor_spec<16xf32>
  %17 = mgp.buffer.concat %6, %ctx[1], %14, %4, %4 : (!mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.buffer<64>
  %18 = mgp.tensor.create %17, %ctx[1], %14, %15 : !mgp.tensor<16xf32>
  mef.output %ch0, %18 : !mef.chain, !mgp.tensor<16xf32>
}
