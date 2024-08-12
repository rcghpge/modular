// RUN: mo-opt %s --allow-unregistered-dialect --fixup-overhangs | FileCheck %s

// This test case is based on mgp-check-overhangs-errors.mlir
mgp.model @if : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  %2 = grt.chain.create (%0, %1 : !mgp.context)
  mef.output %2, %1, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.tensor<32xui8>) {
  %0 = mgp.buffer.constant %arg0, %arg1[1], #M.aligned_bytes<"0x01", align 8> : !M.array<1xui8>
  %2 = mgp.buffer.alloc.static %arg0, %arg1[1] : !mgp.buffer<32>
  %const0 = mip.constant.index 0
  %3 = mgp.buffer.set_with_index %arg1[1], %2, %const0 : !mgp.buffer<32>
  %4 = mgp.tensor_spec.create  : !mgp.tensor_spec<32xui8>
  %5 = mgp.buffer.alloc.static %arg0, %arg1[1] : !mgp.buffer<32>
  %const1 = mip.constant.index 1
  %6 = mgp.buffer.set_with_index %arg1[1], %5, %const1 : !mgp.buffer<32>
  %7 = mgp.buffer.to_bool %arg0, %arg1[1], %0 : !mgp.buffer<1>
  // CHECK: %[[IF_OUTS:.*]]:4 = grt.if
  %ch1, %ch2, %buf, %spec = grt.if %7 (%arg1 as %ctx : !mgp.context, %arg0 as %arg2: !mef.chain, %3 as %arg3: !mef.chain, %2 as %arg4: !mgp.buffer<32>, %4 as %arg5: !mgp.tensor_spec<32xui8>, %6 as %arg6: !mef.chain, %5 as %arg7: !mgp.buffer<32>, %4 as %arg8: !mgp.tensor_spec<32xui8>) -> (!mef.chain, !mef.chain, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) {
    %11 = mgp.buffer.alloc.static %arg2, %ctx[1] : !mgp.buffer<32>
    %12 = grt.chain.create (%arg3, %arg6)
    %dead = grt.chain.create (%arg3, %arg6)
    // Note here though the op's chain result appears to be used by the
    // mef.output, the corresponding grt.if result has no uses.
    %13 = "some.add"(%12, %arg4, %arg5, %arg7, %arg8, %11, %arg8) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mef.chain
    mef.output %13, %13, %11, %arg8 : !mef.chain, !mef.chain, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>
  } else {
    %11 = mgp.buffer.alloc.static %arg2, %ctx[1] : !mgp.buffer<32>
    %12 = grt.chain.create (%arg3, %arg6)
    %13 = "some.sub"(%12, %arg4, %arg5, %arg7, %arg8, %11, %arg8) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mef.chain
    mef.output %13, %13, %11, %arg8 : !mef.chain, !mef.chain, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>
  }
  // CHECK: %[[OUT:[^ ]+]] = mgp.tensor.create {{.*}} %[[IF_OUTS]]#2, %[[IF_OUTS]]#3 : !mgp.tensor<32xui8>
  %10 = mgp.tensor.create %arg0, %arg1[1], %buf, %spec : !mgp.tensor<32xui8>
  // CHECK: %[[MERGED_CHAIN:[^ ]+]] = grt.chain.create (%arg0, %[[IF_OUTS]]#0, %[[IF_OUTS]]#1)
  // CHECK-NEXT: mef.output %[[MERGED_CHAIN]], %[[OUT]] : !mef.chain, !mgp.tensor<32xui8>
  mef.output %arg0, %10 : !mef.chain, !mgp.tensor<32xui8>
}
