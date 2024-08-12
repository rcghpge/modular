// RUN: mo-opt %s --allow-unregistered-dialect --cse --split-input-file | FileCheck %s

// Original graph
// mo.graph @oneOp() -> (!mo.tensor<[8], f32>) {
//  %iota = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0> : tensor<8xf32>} : () -> !mo.tensor<[8],f32>
//  %tens = "mo.constant"() {value = #M.dense_array<10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0> : tensor<8xf32>} : () -> !mo.tensor<[8], f32>  %1 = mo.mul(%iota, %tens) : !mo.tensor<[8], f32>
//  %2 = mo.mul(%1, %tens) : !mo.tensor<[8], f32>
//  %3 = mo.add(%2, %tens) : !mo.tensor<[8], f32>
//  mo.output %3 : !mo.tensor<[8], f32>
//}

module {
  mgp.model @basicMoGraph : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 0
    mef.output %0, %1, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">
  } execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">) -> (!mef.chain, !mgp.tensor<8xf32>) {
    %0 = mgp.buffer.constant %arg0, %arg1[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32>
    %1 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %2 = mgp.tensor.create %arg0, %arg1[1], %0 , %1 : !mgp.tensor<8xf32>
    %3 = mgp.buffer.constant %arg0, %arg1[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32>
    %4 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %5 = mgp.tensor.create %arg0, %arg1[1], %3 , %4 : !mgp.tensor<8xf32>
    %bufferResult = mgp.tensor.extract.buffer %arg1[1], %2 : !mgp.tensor<8xf32>
    %tensorSpecResult = mgp.tensor.extract.tensor_spec %arg1[1], %2 : !mgp.tensor<8xf32>
    %bufferResult_0 = mgp.tensor.extract.buffer %arg1[1], %5 : !mgp.tensor<8xf32>
    %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %arg1[1], %5 : !mgp.tensor<8xf32>
    %6 = mgp.buffer.alloc.static %arg0, %arg1[1] : !mgp.buffer<32>
    %7 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %8 = "mop.kernels.mul"(%arg0, %bufferResult, %tensorSpecResult, %bufferResult_0, %tensorSpecResult_1, %6, %7) : (!mef.chain, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mef.chain
    %9 = mgp.tensor.create %8, %arg1[1], %6 , %7 : !mgp.tensor<8xf32>
    %bufferResult_2 = mgp.tensor.extract.buffer %arg1[1], %9 : !mgp.tensor<8xf32>
    %tensorSpecResult_3 = mgp.tensor.extract.tensor_spec %arg1[1], %9 : !mgp.tensor<8xf32>
    %bufferResult_4 = mgp.tensor.extract.buffer %arg1[1], %5 : !mgp.tensor<8xf32>
    %tensorSpecResult_5 = mgp.tensor.extract.tensor_spec %arg1[1], %5 : !mgp.tensor<8xf32>
    %10 = mgp.buffer.alloc.static %arg0, %arg1[1] : !mgp.buffer<32>
    %11 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %12 = "mop.kernels.mul"(%8, %bufferResult_2, %tensorSpecResult_3, %bufferResult_4, %tensorSpecResult_5, %10, %11) : (!mef.chain, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mef.chain
    %13 = mgp.tensor.create %12, %arg1[1], %10 , %11 : !mgp.tensor<8xf32>
    %bufferResult_6 = mgp.tensor.extract.buffer %arg1[1], %13 : !mgp.tensor<8xf32>
    %tensorSpecResult_7 = mgp.tensor.extract.tensor_spec %arg1[1], %13 : !mgp.tensor<8xf32>
    %bufferResult_8 = mgp.tensor.extract.buffer %arg1[1], %5 : !mgp.tensor<8xf32>
    %tensorSpecResult_9 = mgp.tensor.extract.tensor_spec %arg1[1], %5 : !mgp.tensor<8xf32>
    %14 = mgp.buffer.alloc.static %arg0, %arg1[1] : !mgp.buffer<32>
    %15 = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
    %16 = "mop.kernels.add"(%12, %bufferResult_6, %tensorSpecResult_7, %bufferResult_8, %tensorSpecResult_9, %14, %15) : (!mef.chain, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mef.chain
    %17 = mgp.tensor.create %16, %arg1[1], %14 , %15 : !mgp.tensor<8xf32>
    mef.output %16, %17 : !mef.chain, !mgp.tensor<8xf32>
  }
}

// Check that we remove all but one of the spec creates.

// CHECK: mgp.model @basicMoGraph
// CHECK: mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
// CHECK-NOT: mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>

// -----

mgp.model @tensor_spec_size : mof devices [#M.device_ref<"cpu", 0>] init (%cpu: !mgp.opaque<"cpu0_context">) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  mef.output %0, %1, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"cpu0_context">
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %cpu: !mgp.opaque<"cpu0_context">, %inTensor: !mgp.tensor<1234x5678xf32>) -> (!mef.chain, !mgp.tensor<1234x5678xf32>) {
  %spec = mgp.tensor.extract.tensor_spec %arg1[1], %inTensor: !mgp.tensor<1234x5678xf32>
  %size0 = mgp.tensor_spec.size %spec : !mgp.tensor_spec<1234x5678xf32>
  %size1 = mgp.tensor_spec.size %spec : !mgp.tensor_spec<1234x5678xf32>
  %ch0 = mip.print.index %size0, %arg0
  %ch1 = mip.print.index %size1, %ch0
  mef.output %ch1, %inTensor : !mef.chain, !mgp.tensor<1234x5678xf32>
}

// CHECK: mgp.model @tensor_spec_size
// CHECK: mgp.tensor_spec.size %{{.*}} : !mgp.tensor_spec<1234x5678xf32>
// CHECK-NOT: mgp.tensor_spec.size %{{.*}} : !mgp.tensor_spec<1234x5678xf32>
