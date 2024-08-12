// RUN: mo-opt %s --allow-unregistered-dialect --buffer-lifetimes  | FileCheck %s

mgp.model @serial : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer {{.*}}
  %inBuffer = mgp.tensor.extract.buffer %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inDims:2 = mgp.tensor_spec.dims.dynamic %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = mip.nary.mul(%inDims#0, %inDims#1)
  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc.dynamic {{.*}}

  %alloc0 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc1 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc2 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>

  // CHECK-DAG: %[[CH0:.*]] = mgp.kernel.execute {{.*}} (in %[[INBUFFER]] : !mgp.buffer<>, {{.*}})
  %outChain_0 = mgp.kernel.execute %arg0, %arg1[1, 1], (0, 1), (in %inBuffer : !mgp.buffer<>, out %alloc0 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-DAG: %[[CH1:.*]] = mgp.kernel.execute {{.*}} (in %[[ALLOC0]] : !mgp.buffer<>, {{.*}})
  %outChain_1 = mgp.kernel.execute %outChain_0, %arg1[1, 1], (0, 1), (in %alloc0 : !mgp.buffer<>, out %alloc1 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-DAG: %[[FREE0:.*]] = mgp.buffer.free %[[CH1]], %[[ALLOC0]] : !mgp.buffer<>
  // CHECK-DAG: %[[CH2:.*]] = mgp.kernel.execute %[[FREE0]], {{.*}} (in %[[ALLOC1]] : !mgp.buffer<>, {{.*}})
  %outChain_2 = mgp.kernel.execute %outChain_1, %arg1[1, 1], (0, 1), (in %alloc1 : !mgp.buffer<>, out %alloc2 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-DAG: %[[FREE1:.*]] = mgp.buffer.free %[[CH2]], %[[ALLOC1]] : !mgp.buffer<>
  // CHECK-NEXT: %[[OUTTENSOR:.*]] = mgp.tensor.create %[[FREE1]], {{.*}}
  %outTensor = mgp.tensor.create %outChain_2, %arg1[1], %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

mgp.model @parallel : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer {{.*}}
  %inBuffer = mgp.tensor.extract.buffer %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inDims:2 = mgp.tensor_spec.dims.dynamic %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = mip.nary.mul(%inDims#0, %inDims#1)
  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc.dynamic {{.*}}

  %alloc0 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc1 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc2 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>

  // CHECK-DAG: %[[CH0:.*]] = mgp.kernel.execute {{.*}} (in %[[INBUFFER]] : !mgp.buffer<>, out %[[ALLOC0]] {{.*}})
  %outChain_0 = mgp.kernel.execute %arg0, %arg1[1, 1], (0, 1), (in %inBuffer : !mgp.buffer<>, out %alloc0 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-DAG: %[[CH1:.*]] = mgp.kernel.execute {{.*}} (in %[[INBUFFER]] : !mgp.buffer<>, out %[[ALLOC1]] {{.*}})
  %outChain_1 = mgp.kernel.execute %outChain_0, %arg1[1, 1], (0, 1), (in %inBuffer : !mgp.buffer<>, out %alloc1 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  %merged = grt.chain.create (%outChain_0, %outChain_1)
  // CHECK-DAG: %[[MERGED:.*]] = grt.chain.create (%[[CH0]], %[[CH1]])

  // CHECK-DAG: %[[CH2:.*]] = mgp.kernel.execute {{.*}} (in %[[ALLOC0]] : !mgp.buffer<>, in %[[ALLOC1]] : !mgp.buffer<>, {{.*}})
  %outChain_2 = mgp.kernel.execute %merged, %arg1[1, 1], (0, 1), (in %alloc0 : !mgp.buffer<>, in %alloc1 : !mgp.buffer<>, out %alloc2 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-DAG: %[[FREE1:.*]] = mgp.buffer.free %[[CH2]], %[[ALLOC1]] : !mgp.buffer<>
  // CHECK-DAG: %[[FREE0:.*]] = mgp.buffer.free %[[FREE1]], %[[ALLOC0]] : !mgp.buffer<>
  // CHECK-NEXT: %[[OUTTENSOR:.*]] = mgp.tensor.create %[[FREE0]], {{.*}}
  %outTensor = mgp.tensor.create %outChain_2, %arg1[1], %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

mgp.model @tensor_create_aliased : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer {{.*}}
  %inBuffer = mgp.tensor.extract.buffer %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inDims:2 = mgp.tensor_spec.dims.dynamic %inTensorSpec : !mgp.tensor_spec<?x?xf32>

  %allocSize = mip.nary.mul(%inDims#0, %inDims#1)
  // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc.dynamic {{.*}}
  // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc.dynamic {{.*}}

  %alloc0 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc1 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  %alloc2 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
  // CHECK-DAG: %[[CH0:.*]] = mgp.kernel.execute {{.*}} (in %[[INBUFFER]] : !mgp.buffer<>, {{.*}})
  %outChain_0 = mgp.kernel.execute %arg0, %arg1[1, 1], (0, 1), (in %inBuffer : !mgp.buffer<>, out %alloc0 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true

  %alias0_tensor = mgp.tensor.create %outChain_0, %arg1[1], %alloc0, %inTensorSpec : !mgp.tensor<?x?xf32>
  // CHECK-DAG: %[[ALIAS0:.*]] = mgp.tensor.extract.buffer {{.*}} : !mgp.tensor<?x?xf32>
  %alias0 = mgp.tensor.extract.buffer %arg1[1], %alias0_tensor : !mgp.tensor<?x?xf32>

  // CHECK-DAG: %[[CH1:.*]] = mgp.kernel.execute {{.*}} (in %[[ALLOC0]] : !mgp.buffer<>, {{.*}})
  %outChain_1 = mgp.kernel.execute %outChain_0, %arg1[1, 1], (0, 1), (in %alloc0 : !mgp.buffer<>, out %alloc1 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-DAG: %[[CH2:.*]] = mgp.kernel.execute {{.*}} (in %[[ALLOC1]] : !mgp.buffer<>, {{.*}})
  %outChain_2 = mgp.kernel.execute %outChain_1, %arg1[1, 1], (0, 1), (in %alloc1 : !mgp.buffer<>, out %alloc2 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-DAG: mgp.buffer.free %[[CH2]], %[[ALLOC1]] : !mgp.buffer<>
  // CHECK-DAG: %[[CH3:.*]] = mgp.kernel.execute {{.*}} (in %[[ALIAS0]] : !mgp.buffer<>, {{.*}})
  %outChain_3 = mgp.kernel.execute %outChain_2, %arg1[1, 1], (0, 1), (in %alias0 : !mgp.buffer<>, out %alloc2 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
  // CHECK-DAG: mgp.buffer.free %[[CH3]], %[[ALLOC0]] : !mgp.buffer<>

  %outTensor = mgp.tensor.create %outChain_3, %arg1[1], %alloc2, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}

// -----

mgp.model @nested_block : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">

} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">, %inTensor: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  // CHECK: %[[INBUFFER:.*]] = mgp.tensor.extract.buffer {{.*}}
  %inBuffer = mgp.tensor.extract.buffer %arg1[1], %inTensor : !mgp.tensor<?x?xf32>
  %inTensorSpec = mgp.tensor.extract.tensor_spec %arg1[1], %inTensor : !mgp.tensor<?x?xf32>

  %buf1 = mgp.buffer.constant %arg0, %arg1[1], #M.dense_array<1> : !M.array<1xui8>
  %b1 = mgp.buffer.to_bool %arg0, %arg1[1], %buf1 : !mgp.buffer<1>

  %ifchain, %outbuf = grt.if %b1 (%arg0: !mef.chain, %arg1: !mgp.context, %inBuffer: !mgp.buffer<>, %inTensorSpec: !mgp.tensor_spec<?x?xf32>) -> (!mef.chain, !mgp.buffer<>) {
    %inDims:2 = mgp.tensor_spec.dims.dynamic %inTensorSpec : !mgp.tensor_spec<?x?xf32>

    %allocSize = mip.nary.mul(%inDims#0, %inDims#1)

    // CHECK: %[[ALLOC0:.*]] = mgp.buffer.alloc.dynamic {{.*}}
    // CHECK-NEXT: %[[ALLOC1:.*]] = mgp.buffer.alloc.dynamic {{.*}}
    // CHECK-NEXT: %[[ALLOC2:.*]] = mgp.buffer.alloc.dynamic {{.*}}
    %alloc0 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
    %alloc1 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>
    %alloc2 = mgp.buffer.alloc.dynamic %arg0, %arg1[1], %allocSize, align 16 : !mgp.buffer<>

    // CHECK-DAG: %[[CH0:.*]] = mgp.kernel.execute {{.*}} (in %[[INBUFFER]] : !mgp.buffer<>, {{.*}})
    %outChain_0 = mgp.kernel.execute %arg0, %arg1[1, 1], (0, 1), (in %inBuffer : !mgp.buffer<>, out %alloc0 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
    // CHECK-DAG: %[[CH1:.*]] = mgp.kernel.execute {{.*}} (in %[[ALLOC0]] : !mgp.buffer<>, {{.*}})
    %outChain_1 = mgp.kernel.execute %outChain_0, %arg1[1, 1], (0, 1), (in %alloc0 : !mgp.buffer<>, out %alloc1 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
    // CHECK-DAG: mgp.buffer.free %[[CH1]], %[[ALLOC0]] : !mgp.buffer<>
    // CHECK-DAG: %[[CH2:.*]] = mgp.kernel.execute {{.*}} (in %[[ALLOC1]] : !mgp.buffer<>, {{.*}})
    %outChain_2 = mgp.kernel.execute %outChain_1, %arg1[1, 1], (0, 1), (in %alloc1 : !mgp.buffer<>, out %alloc2 : !mgp.buffer<>), (%inDims#0, %inDims#1),  summary "" is_async true
    // CHECK-DAG: mgp.buffer.free %[[CH2]], %[[ALLOC1]] : !mgp.buffer<>
    mef.output %outChain_2, %alloc2 : !mef.chain, !mgp.buffer<>
  } else {
    mef.output %arg0, %inBuffer : !mef.chain, !mgp.buffer<>
  }
  %outTensor = mgp.tensor.create %ifchain, %arg1[1], %outbuf, %inTensorSpec : !mgp.tensor<?x?xf32>
  mef.output %arg0, %outTensor : !mef.chain, !mgp.tensor<?x?xf32>
}
