// RUN: mgp-opt %s --allow-unregistered-dialect --cse | FileCheck %s

// Original graph
// mo.graph @oneOp() -> (!mo.tensor<[8], f32>) {
//  %iota = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<8xf32>} : () -> !mo.tensor<[8],f32>
//  %tens = "mo.constant"() {value = #M.dense_array<10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0> : tensor<8xf32>} : () -> !mo.tensor<[8], f32>  %1 = mo.mul(%iota, %tens) : !mo.tensor<[8], f32>
//  %2 = mo.mul(%1, %tens) : !mo.tensor<[8], f32>
//  %3 = mo.add(%2, %tens) : !mo.tensor<[8], f32>
//  mo.output %3 : !mo.tensor<[8], f32>
//}

module {
  mgp.model @basicMoGraph : mof init {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 0
    mef.output %0, %1 : !mef.chain, !mgp.context
  } execute (%arg0: !mef.chain, %arg1: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<8xf32>) {
    %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
    %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
    %2 = mgp.cpu.tensor.create %arg0, %0 , %1 : !mgp.cpu.tensor<8xf32>
    %3 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
    %4 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
    %5 = mgp.cpu.tensor.create %arg0, %3 , %4 : !mgp.cpu.tensor<8xf32>
    %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %2 : !mgp.cpu.tensor<8xf32>
    %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
    %6 = mgp.cpu.buffer.alloc.static %arg0, 32
    %7 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
    %8 = "mop.kernels.mul"(%arg0, %bufferResult, %tensorSpecResult, %bufferResult_0, %tensorSpecResult_1, %6, %7) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
    %9 = mgp.cpu.tensor.create %8, %6 , %7 : !mgp.cpu.tensor<8xf32>
    %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %9 : !mgp.cpu.tensor<8xf32>
    %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
    %10 = mgp.cpu.buffer.alloc.static %arg0, 32
    %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
    %12 = "mop.kernels.mul"(%8, %bufferResult_2, %tensorSpecResult_3, %bufferResult_4, %tensorSpecResult_5, %10, %11) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
    %13 = mgp.cpu.tensor.create %12, %10 , %11 : !mgp.cpu.tensor<8xf32>
    %bufferResult_6, %tensorSpecResult_7 = mgp.cpu.tensor.extract %13 : !mgp.cpu.tensor<8xf32>
    %bufferResult_8, %tensorSpecResult_9 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
    %14 = mgp.cpu.buffer.alloc.static %arg0, 32
    %15 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
    %16 = "mop.kernels.add"(%12, %bufferResult_6, %tensorSpecResult_7, %bufferResult_8, %tensorSpecResult_9, %14, %15) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
    %17 = mgp.cpu.tensor.create %16, %14 , %15 : !mgp.cpu.tensor<8xf32>
    mef.output %16, %17 : !mef.chain, !mgp.cpu.tensor<8xf32>
  }
}

// Check that we remove all but one of the spec creates.

// CHECK: mgp.model @basicMoGraph
// CHECK: mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
// CHECK-NOT: mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
