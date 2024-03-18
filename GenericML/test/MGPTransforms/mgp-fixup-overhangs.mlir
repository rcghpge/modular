// RUN: mo-opt %s --allow-unregistered-dialect --fixup-overhangs --split-input-file --verify-diagnostics | FileCheck %s

// This test case is based on mgp-check-overhangs-errors.mlir
mgp.model @if : mof init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  %2 = grt.chain.create (%0, %1 : !mgp.context)
  mef.output %2, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<32xui8>) {
  %0 = mgp.cpu.buffer.constant.deprecated #M.aligned_bytes<"0x01", align 8> : !M.array<1xui8>
  %2 = mgp.cpu.buffer.alloc.static.deprecated %arg0, 32
  %3 = mgp.cpu.buffer.set.deprecated %2 : !mgp.cpu.buffer<32>
  %4 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<32xui8>
  %5 = mgp.cpu.buffer.alloc.static.deprecated %arg0, 32
  %6 = mgp.cpu.buffer.set.deprecated %5, 1 : !mgp.cpu.buffer<32>
  %7 = mgp.to_grt_bool.deprecated(%arg0, %0) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
  // CHECK: %[[IF_OUTS:[^ ]+]]:4 = grt.if %6 (%arg0 as %arg2: !mef.chain, %2 as %arg3: !mef.chain, %1 as %arg4: !mgp.cpu.buffer<32>, %3 as %arg5: !mgp.tensor_spec<32xui8>, %5 as %arg6: !mef.chain, %4 as %arg7: !mgp.cpu.buffer<32>, %3 as %arg8: !mgp.tensor_spec<32xui8>) -> (!mef.chain, !mef.chain, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>) {
  %ch1, %ch2, %buf, %spec = grt.if %7 (%arg0 as %arg2: !mef.chain, %3 as %arg3: !mef.chain, %2 as %arg4: !mgp.cpu.buffer<32>, %4 as %arg5: !mgp.tensor_spec<32xui8>, %6 as %arg6: !mef.chain, %5 as %arg7: !mgp.cpu.buffer<32>, %4 as %arg8: !mgp.tensor_spec<32xui8>) -> (!mef.chain, !mef.chain, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>) {
    %11 = mgp.cpu.buffer.alloc.static.deprecated %arg2, 32
    %12 = grt.chain.create (%arg3, %arg6)
    %dead = grt.chain.create (%arg3, %arg6)
    // Note here though the op's chain result appears to be used by the
    // mef.output, the corresponding grt.if result has no uses.
    %13 = "some.add"(%12, %arg4, %arg5, %arg7, %arg8, %11, %arg8) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mef.chain
    mef.output %13, %13, %11, %arg8 : !mef.chain, !mef.chain, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>
  } else {
    %11 = mgp.cpu.buffer.alloc.static.deprecated %arg2, 32
    %12 = grt.chain.create (%arg3, %arg6)
    %13 = "some.sub"(%12, %arg4, %arg5, %arg7, %arg8, %11, %arg8) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mef.chain
    mef.output %13, %13, %11, %arg8 : !mef.chain, !mef.chain, !mgp.cpu.buffer<32>, !mgp.tensor_spec<32xui8>
  }
  // CHECK:     %[[OUT:[^ ]+]] = mgp.cpu.tensor.create.deprecated %arg0, %[[IF_OUTS]]#2, %[[IF_OUTS]]#3 : !mgp.cpu.tensor<32xui8>
  %10 = mgp.cpu.tensor.create.deprecated %arg0, %buf, %spec : !mgp.cpu.tensor<32xui8>
  // CHECK:     %[[MERGED_CHAIN:[^ ]+]] = grt.chain.create (%arg0, %[[IF_OUTS]]#0, %[[IF_OUTS]]#1)
  // CHECK-NEXT: mef.output %[[MERGED_CHAIN]], %[[OUT]] : !mef.chain, !mgp.cpu.tensor<32xui8>
  mef.output %arg0, %10 : !mef.chain, !mgp.cpu.tensor<32xui8>
}
