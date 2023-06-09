// Just check we still parse it.
// RUN: mgp-opt %s --allow-unregistered-dialect

module {
  mgp.model @upsideDown : tf init {
    %2 = "mop.create_jit_context"(%0, %1) {slot = 0 : ui32} : (!mef.chain, !mgp.context) -> !mef.chain
    %1 = mgp.context.create 1
    %0 = grt.chain.create ()
    mef.output %2, %1 : !mef.chain, !mgp.context
  } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<8xf32>) -> (!mef.chain, !mgp.cpu.tensor<8xf32>) {
    %3 = mgp.cpu.tensor.create %2, %0, %1 : !mgp.cpu.tensor<8xf32>
    %2 = "mop.kernels.add"(%arg0, %bufferResult, %tensorSpecResult, %bufferResult_0, %tensorSpecResult_1, %0, %1) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
    %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
    %0 = mgp.cpu.buffer.alloc.static %arg0, 32 {bRawAlign = 32 : ui64}
    %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<8xf32>
    %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<8xf32>
    mef.output %arg0, %3 : !mef.chain, !mgp.cpu.tensor<8xf32>
  }
}
