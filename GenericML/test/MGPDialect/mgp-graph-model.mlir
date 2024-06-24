// Just check we still parse it.
// RUN: mo-opt %s --allow-unregistered-dialect

module {
  mgp.model @upsideDown : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
    %2 = mgp.runtime.create %0, %1[1] device<"cpu", 0>
    %1 = mgp.context.create 1
    %0 = grt.chain.create ()
    mef.output %2, %1 : !mef.chain, !mgp.context
  } execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque, %arg2: !mgp.tensor<8xf32>) -> (!mef.chain, !mgp.tensor<8xf32>) {
    %ch0 = mgp.chain.host_to_device %arg0, %arg1[1]
    %spec = mgp.tensor_spec.create : !mgp.tensor_spec<8xf32>
    %buf = mgp.buffer.alloc.static %ch0, %arg1[1] : !mgp.buffer<32>
    %3 = mgp.tensor.create %ch0, %arg1[1], %buf, %spec : !mgp.tensor<8xf32>
    %2 = "mop.kernels.add"(%arg0, %bufferResult, %tensorSpecResult, %bufferResult_0, %tensorSpecResult_1, %0, %1) : (!mef.chain, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mef.chain
    %1 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %0 = mgp.buffer.alloc.static %ch0, %arg1[1] : !mgp.buffer<32>
    %bufferResult = mgp.tensor.extract.buffer %arg1[1], %arg2 : !mgp.tensor<8xf32>
    %tensorSpecResult = mgp.tensor.extract.tensor_spec %arg1[1], %arg2 : !mgp.tensor<8xf32>
    %bufferResult_0 = mgp.tensor.extract.buffer %arg1[1], %arg2 : !mgp.tensor<8xf32>
    %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %arg1[1], %arg2 : !mgp.tensor<8xf32>
    mef.output %arg0, %3 : !mef.chain, !mgp.tensor<8xf32>
  }
}
