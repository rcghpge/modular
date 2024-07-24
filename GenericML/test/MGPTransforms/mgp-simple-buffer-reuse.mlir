// RUN: mo-opt %s --allow-unregistered-dialect --simple-buffer-reuse --split-input-file --verify-diagnostics | FileCheck %s

// This test case covers the case of serial execution commonly found in models
//
//     execute(in %inBuffer, out %alloc0)
//  -> execute(%alloc0, %alloc1)
//  -> execute(%alloc1, %alloc2)
//
// Since %alloc2's first use occurs after the last use of %alloc0, we can safely reuse %alloc0
//
//     execute(in %inBuffer, out %alloc0)
//  -> execute(in %alloc0, out %alloc1)
//  -> execute(in %alloc1, out %alloc0)

mgp.model @serial : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %ch = mgp.chain.host_to_device %arg0, %arg1[1]
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer {{.*}}
  %inBuffer = mgp.tensor.extract.buffer %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inDims:2 = mgp.tensor_spec.dims.dynamic %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = mip.nary.mul(%inDims#0, %inDims#1)

  // We only expect 2 remaining allocs after the pass

  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  %alloc0 = mgp.buffer.alloc.dynamic %ch, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc1 = mgp.buffer.alloc.dynamic %ch, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc2 = mgp.buffer.alloc.dynamic %ch, %arg1[1], %allocSize, align 16 : !mgp.buffer<>


  // CHECK-NEXT: %[[CH0:.*]] = mgp.kernel.execute {{.*}} (in %[[INBUFFER]] : !mgp.buffer<>, out %[[ALLOC0]] : !mgp.buffer<>)
  %outChain_0 = mgp.kernel.execute %ch, %arg1[1, 1], (0, 1), (in %inBuffer : !mgp.buffer<>, out %alloc0 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-NEXT: %[[CH1:.*]] = mgp.kernel.execute {{.*}} (in %[[ALLOC0]] : !mgp.buffer<>, out %[[ALLOC1]] : !mgp.buffer<>)
  %outChain_1 = mgp.kernel.execute %outChain_0, %arg1[1, 1], (0, 1), (in %alloc0 : !mgp.buffer<>, out %alloc1 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true

  // Check to make sure all uses of %alloc2 are replaced with %alloc0
  // CHECK-NEXT: %[[CH2:.*]] = mgp.kernel.execute %[[CH1]], {{.*}} (in %[[ALLOC1]] : !mgp.buffer<>, out %[[ALLOC0]] : !mgp.buffer<>)
  %outChain_2 = mgp.kernel.execute %outChain_1, %arg1[1, 1], (0, 1), (in %alloc1 : !mgp.buffer<>, out %alloc2 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-NEXT: %[[OUTTENSOR:.*]] = mgp.tensor.create %[[CH2]], {{.*}}
  %outTensor = mgp.tensor.create %outChain_2, %arg1[1], %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

// This test case covers the same case of serial execution as above.
// Even though the lifetimes are the same as above and indicate that the buffer
// can be reused, the sizes are different so the simple buffer reuse pass will
// not reuse.
mgp.model @serial_negative : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %arg0 : !mef.chain, !mgp.context, !mgp.opaque
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %ch = mgp.chain.host_to_device %arg0, %arg1[1]
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer {{.*}}
  %inBuffer = mgp.tensor.extract.buffer %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inDims:2 = mgp.tensor_spec.dims.dynamic %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = mip.nary.mul(%inDims#0, %inDims#1)
  %allocSizeDifferent = mip.add %inDims#0, %inDims#1


  // We still expect 3 remaining allocs after the pass

  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  %alloc0 = mgp.buffer.alloc.dynamic %ch, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc1 = mgp.buffer.alloc.dynamic %ch, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc2 = mgp.buffer.alloc.dynamic %ch, %arg1[1], %allocSizeDifferent, align 16 : !mgp.buffer<>


  // CHECK-NEXT: %[[CH0:.*]] = mgp.kernel.execute {{.*}} (in %[[INBUFFER]] : !mgp.buffer<>, out %[[ALLOC0]] : !mgp.buffer<>)
  %outChain_0 = mgp.kernel.execute %ch, %arg1[1, 1], (0, 1), (in %inBuffer : !mgp.buffer<>, out %alloc0 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-NEXT: %[[CH1:.*]] = mgp.kernel.execute {{.*}} (in %[[ALLOC0]] : !mgp.buffer<>, out %[[ALLOC1]] : !mgp.buffer<>)
  %outChain_1 = mgp.kernel.execute %outChain_0, %arg1[1, 1], (0, 1), (in %alloc0 : !mgp.buffer<>, out %alloc1 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-NEXT: %[[CH2:.*]] = mgp.kernel.execute %[[CH1]], {{.*}} (in %[[ALLOC1]] : !mgp.buffer<>, out %[[ALLOC2]] : !mgp.buffer<>)
  %outChain_2 = mgp.kernel.execute %outChain_1, %arg1[1, 1], (0, 1), (in %alloc1 : !mgp.buffer<>, out %alloc2 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-NEXT: %[[OUTTENSOR:.*]] = mgp.tensor.create %[[CH2]], {{.*}}
  %outTensor = mgp.tensor.create %outChain_2, %arg1[1], %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}
