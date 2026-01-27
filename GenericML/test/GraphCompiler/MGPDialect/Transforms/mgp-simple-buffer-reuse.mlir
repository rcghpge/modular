// RUN: mo-opt %s --allow-unregistered-dialect --simple-buffer-reuse --split-input-file | FileCheck %s

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

mgp.model @serial : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer
  %inBuffer = mgp.tensor.extract.buffer %inTensor : !mgp.tensor<?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %inTensor : <?x?xf32>
  %inDims0 = mgp.tensor_spec.get_dim[0] %inTensorSpec : !mgp.tensor_spec<?x?xf32>
  %inDims1 = mgp.tensor_spec.get_dim[1] %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = index.mul %inDims0, %inDims1

  // We only expect 2 remaining allocs after the pass

  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc {{.*}}
  %alloc0 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc1 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc2 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>


  // CHECK-NEXT: %[[CH0:.*]] = "kernel_execute"(%{{.*}}, %[[INBUFFER]], %[[ALLOC0]])
  %outChain_0 = "kernel_execute"(%arg0, %inBuffer, %alloc0) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // CHECK-NEXT: %[[CH1:.*]] = "kernel_execute"(%{{.*}}, %[[ALLOC0]], %[[ALLOC1]])
  %outChain_1 = "kernel_execute"(%outChain_0, %alloc0, %alloc1) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // Check to make sure all uses of %alloc2 are replaced with %alloc0
  // CHECK-NEXT: %[[CH2:.*]] = "kernel_execute"(%[[CH1]], %[[ALLOC1]], %[[ALLOC0]])
  %outChain_2 = "kernel_execute"(%outChain_1, %alloc1, %alloc2) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain

  // CHECK-NEXT: %[[OUTTENSOR:.*]] = mgp.tensor.create %[[CH2]]
  %outTensor = mgp.tensor.create %outChain_2, %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

// This test case covers the same case of serial execution as above.
// Even though the lifetimes are the same as above and indicate that the buffer
// can be reused, the sizes are different so the simple buffer reuse pass will
// not reuse.
mgp.model @serial_negative : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer
  %inBuffer = mgp.tensor.extract.buffer %inTensor : !mgp.tensor<?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %inTensor : <?x?xf32>
  %inDims0 = mgp.tensor_spec.get_dim[0] %inTensorSpec : !mgp.tensor_spec<?x?xf32>
  %inDims1 = mgp.tensor_spec.get_dim[1] %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = index.mul %inDims0, %inDims1
  %allocSizeDifferent = index.add %inDims0, %inDims1


  // We still expect 3 remaining allocs after the pass

  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc {{.*}}
  // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc {{.*}}
  %alloc0 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc1 = mgp.buffer.alloc %arg0, %allocSize, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>
  %alloc2 = mgp.buffer.alloc %arg0, %allocSizeDifferent, %cpu, align 16 : !mgp.buffer, !mgp.device_context<<"cpu", 0>>


  // CHECK-NEXT: %[[CH0:.*]] = "kernel_execute"({{.*}}, %[[INBUFFER]], %[[ALLOC0]])
  %outChain_0 = "kernel_execute"(%arg0, %inBuffer, %alloc0) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // CHECK-NEXT: %[[CH1:.*]] = "kernel_execute"({{.*}}, %[[ALLOC0]], %[[ALLOC1]])
  %outChain_1 = "kernel_execute"(%outChain_0, %alloc0, %alloc1) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain
  // CHECK-NEXT: %[[CH2:.*]] = "kernel_execute"(%[[CH1]], %[[ALLOC1]], %[[ALLOC2]])
  %outChain_2 = "kernel_execute"(%outChain_1, %alloc1, %alloc2) : (!mef.chain, !mgp.buffer, !mgp.buffer) -> !mef.chain

  // CHECK-NEXT: %[[OUTTENSOR:.*]] = mgp.tensor.create %[[CH2]]
  %outTensor = mgp.tensor.create %outChain_2, %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}
