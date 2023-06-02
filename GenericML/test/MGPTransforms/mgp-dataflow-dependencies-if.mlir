// RUN: mgp-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test | FileCheck %s

//
// Simple grt.if
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
 //             ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "some.op"(%arg0) : (!mef.chain) -> i1
// CHECK-NEXT:  %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %2 = "some.op"(%arg0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  %3:3 = "grt.if"(%0, %arg0, %1, %2, %2) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.tensor<f32>, %arg4: !mgp.cpu.tensor<f32>, %arg5: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    %5 = "some.op"(%arg2) : (!mef.chain) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:    %6 = "some.op"(%arg4) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:    "mef.output"(%5, %6, %arg5) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:  }, {
//              ### Block 3 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.tensor<f32>, %arg4: !mgp.cpu.tensor<f32>, %arg5: !mgp.cpu.tensor<f32>):
// CHECK-NEXT:    %5 = "some.op"(%arg4) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:    "mef.output"(%arg3, %5, %arg5) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:  }) : (i1, !mef.chain, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>)
// CHECK-NEXT:  %4 = "some.op"(%3#0, %3#1, %3#2) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
// CHECK-NEXT:  "mef.output"(%arg0, %4) : (!mef.chain, !mgp.cpu.tensor<f32>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<f32>, sym_name = "simple"} : () -> ()

// CHECK: sameAs(%arg4[2], %arg5[2])
// CHECK: sameAs(%arg4[2], %arg5[3])
// CHECK: sameAs(%3#2, %2)

// CHECK: impliesReady(%3#0, %arg0)
// CHECK: impliesReady(%3#1, %2)
// CHECK: impliesReady(%3#2, %2)

mgp.model @simple : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<f32>) {
  %cond = "some.op"(%ch0) : (!mef.chain) -> i1
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
  %i, %j, %k = grt.if %cond (%ch0: !mef.chain, %x: !mgp.cpu.tensor<f32>, %y: !mgp.cpu.tensor<f32>, %y as %z: !mgp.cpu.tensor<f32>)
    -> (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) {
    %c = "some.op"(%ch0) : (!mef.chain) -> !mgp.cpu.tensor<f32>
    %d = "some.op"(%y) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
    mef.output %c, %d, %z : !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>
  } else {
    %e = "some.op"(%y) : (!mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
    mef.output %x, %e, %z : !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>, !mgp.cpu.tensor<f32>) -> !mgp.cpu.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.cpu.tensor<f32>
}

//
// Lots of redundant grt.chain.merge operands.
//

// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "grt.chain.merge"(%arg0) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %1 = "grt.chain.merge"(%0, %0) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %2 = "some.op"(%1) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %3 = "grt.chain.merge"(%arg0, %2) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %4 = "some.op"(%3) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %5 = "some.op"(%3) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %6 = "some.op"(%3) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %7 = "grt.chain.merge"(%4, %5) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %8 = "grt.chain.merge"(%3, %7, %6) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %9 = "grt.chain.merge"() : () -> !mef.chain
// CHECK-NEXT:  %10 = "grt.chain.merge"(%9, %8) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %11 = "some.op"() : () -> i1
// CHECK-NEXT:  %12 = "grt.if"(%11, %9) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain):
// CHECK-NEXT:    "mef.output"(%arg2) : (!mef.chain) -> ()
// CHECK-NEXT:  }, {
//              ### Block 3 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain):
// CHECK-NEXT:    %14 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:    "mef.output"(%14) : (!mef.chain) -> ()
// CHECK-NEXT:  }) : (i1, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %13 = "grt.chain.merge"(%9, %10, %12) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  "mef.output"(%13) : (!mef.chain) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> (), sym_name = "redundant"} : () -> ()

// CHECK: isAlwaysReady(%9)
// CHECK: isAlwaysReady(%arg2[2])
// CHECK: isAlwaysReady(%arg2[3])
// CHECK: isAlwaysReady(%14[3])
// CHECK: isAlwaysReady(%12)

mgp.model @redundant : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch1 = grt.chain.merge (%ch0)
  %ch2 = grt.chain.merge (%ch1, %ch1)
  %ch3 = "some.op"(%ch2) : (!mef.chain) -> !mef.chain
  %ch4 = grt.chain.merge (%ch0, %ch3)
  %ch5 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch6 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch7 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch8 = grt.chain.merge (%ch5, %ch6)
  %ch9 = grt.chain.merge (%ch4, %ch8, %ch7)
  %ch10 = grt.chain.merge ()
  %ch11 = grt.chain.merge (%ch10, %ch9)
  %c = "some.op"() : () -> i1
  %ch12 = grt.if %c (%ch10 : !mef.chain) -> !mef.chain {
    mef.output %ch10 : !mef.chain
  } else {
    %ch13 = grt.chain.create
    mef.output %ch13 : !mef.chain
  }
  %ch14 = grt.chain.merge (%ch10, %ch11, %ch12)
  mef.output %ch14 : !mef.chain
}

//
// Realistic grt.if with tensors in and out
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
//             ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "mgp.cpu.buffer.constant"() {aValue = #M.aligned_bytes<"0x01", align 8> : !M.array<1xui8>, bRawAlign = 8 : ui64} : () -> !mgp.cpu.buffer<1>
// CHECK-NEXT:  %1 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 1 : ui8} : () -> !mgp.cpu.tensor_spec<bool>
// CHECK-NEXT:  %2 = "mgp.cpu.tensor.create"(%arg0, %0, %1) : (!mef.chain, !mgp.cpu.buffer<1>, !mgp.cpu.tensor_spec<bool>) -> !mgp.cpu.tensor<bool>
// CHECK-NEXT:  %3 = "mgp.cpu.buffer.alloc.static"(%arg0) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %4 = "mgp.cpu.buffer.set"(%3) {value = 0 : ui8} : (!mgp.cpu.buffer<32>) -> !mef.chain
// CHECK-NEXT:  %5 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.cpu.tensor_spec<32xui8>
// CHECK-NEXT:  %6 = "mgp.cpu.tensor.create"(%4, %3, %5) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
// CHECK-NEXT:  %7 = "mgp.cpu.buffer.alloc.static"(%arg0) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %8 = "mgp.cpu.buffer.set"(%7) {value = 1 : ui8} : (!mgp.cpu.buffer<32>) -> !mef.chain
// CHECK-NEXT:  %9 = "mgp.cpu.tensor.create"(%8, %7, %5) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
// CHECK-NEXT:  %10:2 = "mgp.cpu.tensor.extract"(%2) : (!mgp.cpu.tensor<bool>) -> (!mgp.cpu.buffer<1>, !mgp.cpu.tensor_spec<bool>)
// CHECK-NEXT:  %11 = "mgp.to_grt_bool"(%arg0, %10#0) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
// CHECK-NEXT:  %12:2 = "grt.if"(%11, %arg0, %6, %9) ({
//              ### Block 2 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.tensor<32xui8>, %arg4: !mgp.cpu.tensor<32xui8>):
// CHECK-NEXT:    %15:2 = "mgp.cpu.tensor.extract"(%arg4) : (!mgp.cpu.tensor<32xui8>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>)
// CHECK-NEXT:    %16 = "mgp.cpu.tensor.create"(%arg2, %15#0, %15#1) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
// CHECK-NEXT:    "mef.output"(%arg2, %16) : (!mef.chain, !mgp.cpu.tensor<32xui8>) -> ()
// CHECK-NEXT:  }, {
//              ### Block 3 ###
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.tensor<32xui8>, %arg4: !mgp.cpu.tensor<32xui8>):
// CHECK-NEXT:    %15:2 = "mgp.cpu.tensor.extract"(%arg3) : (!mgp.cpu.tensor<32xui8>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>)
// CHECK-NEXT:    %16 = "mgp.cpu.tensor.create"(%arg2, %15#0, %15#1) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
// CHECK-NEXT:    "mef.output"(%arg2, %16) : (!mef.chain, !mgp.cpu.tensor<32xui8>) -> ()
// CHECK-NEXT:  }) : (i1, !mef.chain, !mgp.cpu.tensor<32xui8>, !mgp.cpu.tensor<32xui8>) -> (!mef.chain, !mgp.cpu.tensor<32xui8>)
// CHECK-NEXT:  %13:2 = "mgp.cpu.tensor.extract"(%12#1) : (!mgp.cpu.tensor<32xui8>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>)
// CHECK-NEXT:  %14 = "mgp.cpu.tensor.create"(%12#0, %13#0, %13#1) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<32xui8>) -> !mgp.cpu.tensor<32xui8>
// CHECK-NEXT:  "mef.output"(%12#0, %14) : (!mef.chain, !mgp.cpu.tensor<32xui8>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<32xui8>, sym_name = "if"} : () -> ()

// CHECK: sameAs(%arg0, %arg0)
// CHECK: sameAs(%arg0, %arg2[2])
// CHECK: sameAs(%arg0, %arg2[3])
// CHECK: sameAs(%arg0, %12#0)

// CHECK: sameAs(%arg3[2], %6)
// CHECK: sameAs(%arg3[2], %arg3[2])
// CHECK: sameAs(%arg3[2], %arg3[3])

// CHECK: sameAs(%arg4[2], %9)
// CHECK: sameAs(%arg4[2], %arg4[2])
// CHECK: sameAs(%arg4[2], %arg4[3])

// CHECK: sameAs(%arg3[3], %6)
// CHECK: sameAs(%arg3[3], %arg3[2])
// CHECK: sameAs(%arg3[3], %arg3[3])

// CHECK: sameAs(%arg4[3], %9)
// CHECK: sameAs(%arg4[3], %arg4[2])
// CHECK: sameAs(%arg4[3], %arg4[3])

// CHECK: impliesReady(%16[2], %arg0)
// CHECK: impliesReady(%16[2], %5)
// CHECK: impliesReady(%16[2], %7)
// CHECK: impliesReady(%16[2], %8)
// CHECK: impliesReady(%16[2], %9)
// CHECK: impliesReady(%16[2], %arg2[2])
// CHECK: impliesReady(%16[2], %arg4[2])
// CHECK: impliesReady(%16[2], %15#0[2])
// CHECK: impliesReady(%16[2], %15#1[2])
// CHECK: impliesReady(%16[2], %16[2])
// CHECK: impliesReady(%16[2], %arg2[3])
// CHECK: impliesReady(%16[2], %arg4[3])
// This looks weird but is because sameAs(%12#0, %arg0)
// CHECK: impliesReady(%16[2], %12#0)

// CHECK: impliesReady(%16[3], %arg0)
// CHECK: impliesReady(%16[3], %3)
// CHECK: impliesReady(%16[3], %4)
// CHECK: impliesReady(%16[3], %5)
// CHECK: impliesReady(%16[3], %6)
// CHECK: impliesReady(%16[3], %arg2[2])
// CHECK: impliesReady(%16[3], %arg3[2])
// CHECK: impliesReady(%16[3], %arg2[3])
// CHECK: impliesReady(%16[3], %arg3[3])
// CHECK: impliesReady(%16[3], %15#0[3])
// CHECK: impliesReady(%16[3], %15#1[3])
// CHECK: impliesReady(%16[3], %16[3])
// CHECK: impliesReady(%16[3], %12#0)

// CHECK: impliesReady(%12#1, %arg0)
// CHECK: impliesReady(%12#1, %5)
// CHECK: impliesReady(%12#1, %arg2[2])
// CHECK: impliesReady(%12#1, %arg2[3])
// CHECK: impliesReady(%12#1, %12#0)
// CHECK: impliesReady(%12#1, %12#1)

mgp.model @if : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<32xui8>) {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x01", align 8> : !M.array<1xui8>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<bool>
  %2 = mgp.cpu.tensor.create %arg0, %0, %1  : !mgp.cpu.tensor<bool>
  %3 = mgp.cpu.buffer.alloc.static %arg0, 32
  %4 = mgp.cpu.buffer.set %3 : !mgp.cpu.buffer<32>
  %5 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<32xui8>
  %6 = mgp.cpu.tensor.create %4, %3, %5  : !mgp.cpu.tensor<32xui8>
  %7 = mgp.cpu.buffer.alloc.static %arg0, 32
  %8 = mgp.cpu.buffer.set %7, 1 : !mgp.cpu.buffer<32>
  %9 = mgp.cpu.tensor.create %8, %7, %5  : !mgp.cpu.tensor<32xui8>
  %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %2 : !mgp.cpu.tensor<bool>
  %10 = mgp.to_grt_bool(%arg0, %bufferResult) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
  %11:2 = grt.if %10 (%arg0: !mef.chain, %6: !mgp.cpu.tensor<32xui8>, %9: !mgp.cpu.tensor<32xui8>)  -> (!mef.chain, !mgp.cpu.tensor<32xui8>) {
    %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %9 : !mgp.cpu.tensor<32xui8>
    %13 = mgp.cpu.tensor.create %arg0, %bufferResult_2, %tensorSpecResult_3  : !mgp.cpu.tensor<32xui8>
    mef.output %arg0, %13 : !mef.chain, !mgp.cpu.tensor<32xui8>
  } else {
    %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %6 : !mgp.cpu.tensor<32xui8>
    %13 = mgp.cpu.tensor.create %arg0, %bufferResult_2, %tensorSpecResult_3  : !mgp.cpu.tensor<32xui8>
    mef.output %arg0, %13 : !mef.chain, !mgp.cpu.tensor<32xui8>
  }
  %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %11#1 : !mgp.cpu.tensor<32xui8>
  %12 = mgp.cpu.tensor.create %11#0, %bufferResult_0, %tensorSpecResult_1  : !mgp.cpu.tensor<32xui8>
  mef.output %11#0, %12 : !mef.chain, !mgp.cpu.tensor<32xui8>
}
