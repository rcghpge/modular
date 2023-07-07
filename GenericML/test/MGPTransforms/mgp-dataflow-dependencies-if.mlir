// RUN: mgp-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test --check-buffer-read-write | FileCheck %s

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

// CHECK: sameAs begin
// CHECK-NEXT: %arg0 |-> {%0[0]}
// CHECK-NEXT: %arg1 |-> {%1[0]}
// CHECK-NEXT: %arg2[2] |-> {%0[0], %arg0}
// CHECK-NEXT: %arg3[2] |-> {%1}
// CHECK-NEXT: %arg4[2] |-> {%2}
// CHECK-NEXT: %arg5[2] |-> {%2}
// CHECK-NEXT: %arg2[3] |-> {%0[0], %arg0}
// CHECK-NEXT: %arg3[3] |-> {%1}
// CHECK-NEXT: %arg4[3] |-> {%2}
// CHECK-NEXT: %arg5[3] |-> {%2}
// CHECK-NEXT: %3#2 |-> {%2}
// CHECK-NEXT: sameAs end

// Notice if results now capture both arms
// CHECK: maybeSameAs begin
// CHECK-NEXT:  %arg0 |-> {%0[0]}
// CHECK-NEXT:  %arg1 |-> {%1[0]}
// CHECK-NEXT:  %arg2[2] |-> {%0[0], %arg0}
// CHECK-NEXT:  %arg3[2] |-> {%1}
// CHECK-NEXT:  %arg4[2] |-> {%2}
// CHECK-NEXT:  %arg5[2] |-> {%2}
// CHECK-NEXT:  %arg2[3] |-> {%0[0], %arg0}
// CHECK-NEXT:  %arg3[3] |-> {%1}
// CHECK-NEXT:  %arg4[3] |-> {%2}
// CHECK-NEXT:  %arg5[3] |-> {%2}
// CHECK-NEXT:  %3#0 |-> {%1, %5[2], %arg3[3]}
// CHECK-NEXT:  %3#1 |-> {%6[2], %5[3]}
// CHECK-NEXT:  %3#2 |-> {%2, %arg5[2], %arg5[3]}
// CHECK-NEXT: maybeSameAs end

// CHECK: isAlwaysReady({%0[0], %arg0, %arg2[2], %arg2[3]})

// impliesReady is actually 'implies ready or never bound', so
// %3#1 captures that either one or other of the if branches will
// be executed.
// Note that though we know grt.if is strict on the condition, there's
// nothing in the grt.if result that tells us when the grt.if itself has
// completed.
// CHECK: impliesReady begin
// CHECK: %3#0 |-> {%0[0], %arg0, %5[2]}
// CHECK: %3#1 |-> {%0[0], %arg0, %2, %6[2], %5[3]}
// CHECK: %3#2 |-> {%0[0], %arg0, %2}
// CHECK: impliesReady end

// Ops in the if branches are structurally disjoint
// CHECK: structurallyDisjoint begin
// CHECK-NEXT: %5[2] |-> {%5[3]}
// CHECK-NEXT: %6[2] |-> {%5[3]}
// CHECK-NEXT: %5[3] |-> {%5[2], %6[2]}
// CHECK-NEXT: structurallyDisjoint end

// runsLater is actually 'runs later or never runs'.
// CHECK: runsLater begin
// CHECK-NEXT: %0[0] |-> {%0, %1, %2, %3#0, %5[2], %6[2], %5[3], %4}
// CHECK-NEXT: %0 |-> {%3#0}
// CHECK-NEXT: %2 |-> {%6[2], %5[3], %4}
// CHECK-NEXT: %3#0 |-> {%4}
// CHECK-NEXT: %5[2] |-> {%4}
// CHECK-NEXT: %6[2] |-> {%4}
// CHECK-NEXT: %5[3] |-> {%4}
// CHECK: runsLater end

// exclusive now accounts for nesting and the disjointness of the if branches
// CHECK: exclusive begin
// CHECK-NEXT: %0 |-> {%4}
// CHECK-NEXT: %5[2] |-> {%5[3]}
// CHECK-NEXT: %6[2] |-> {%5[3]}
// CHECK-NEXT: %5[3] |-> {%5[2], %6[2]}
// CHECK-NEXT: %4 |-> {%0}
// CHECK-NEXT: exclusive end

mgp.model @simple : tf init {
  %0 = grt.chain.create ()
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
// Lots of redundant grt.chain.create operands.
//

// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "grt.chain.create"(%arg0) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %1 = "grt.chain.create"(%0, %0) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %2 = "some.op"(%1) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %3 = "grt.chain.create"(%arg0, %2) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %4 = "some.op"(%3) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %5 = "some.op"(%3) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %6 = "some.op"(%3) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %7 = "grt.chain.create"(%4, %5) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %8 = "grt.chain.create"(%3, %7, %6) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %9 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %10 = "grt.chain.create"(%9, %8) : (!mef.chain, !mef.chain) -> !mef.chain
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
// CHECK-NEXT:  %13 = "grt.chain.create"(%9, %10, %12) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  "mef.output"(%13) : (!mef.chain) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> (), sym_name = "redundant"} : () -> ()

// CHECK: sameAs begin
// CHECK: %arg2[2] |-> {%9}
// CHECK: %arg2[3] |-> {%9}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK: %12 |-> {%9, %arg2[2], %14[3]}
// CHECK: maybeSameAs end

// CHECK: isAlwaysReady({%0[0], %arg0, %0, %1, %9, %arg2[2], %arg2[3], %14[3], %12})

// CHECK: impliesReady begin
// CHECK: %2 |-> {%0[0], %arg0, %0, %1}
// CHECK: impliesReady end

// CHECK: runsLater begin
// CHECK: %6 |-> {%8, %10, %13}
// CHECK: runsLater end

// CHECK: exclusive begin
// CHECK: %6 |-> {%0[0], %arg0, %0, %1, %2, %10, %13}
// CHECK: exclusive end

mgp.model @redundant : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %ch1 = grt.chain.create (%ch0)
  %ch2 = grt.chain.create (%ch1, %ch1)
  %ch3 = "some.op"(%ch2) : (!mef.chain) -> !mef.chain
  %ch4 = grt.chain.create (%ch0, %ch3)
  %ch5 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch6 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch7 = "some.op"(%ch4) : (!mef.chain) -> !mef.chain
  %ch8 = grt.chain.create (%ch5, %ch6)
  %ch9 = grt.chain.create (%ch4, %ch8, %ch7)
  %ch10 = grt.chain.create ()
  %ch11 = grt.chain.create (%ch10, %ch9)
  %c = "some.op"() : () -> i1
  %ch12 = grt.if %c (%ch10 : !mef.chain) -> !mef.chain {
    mef.output %ch10 : !mef.chain
  } else {
    %ch13 = grt.chain.create ()
    mef.output %ch13 : !mef.chain
  }
  %ch14 = grt.chain.create (%ch10, %ch11, %ch12)
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

// CHECK: sameAs begin
// CHECK: %arg3[2] |-> {%6}
// CHECK: %arg4[2] |-> {%9}
// CHECK: %12#0 |-> {%0[0], %arg0}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK: %12#0 |-> {%0[0], %arg0, %arg2[2], %arg2[3]}
// CHECK: %12#1 |-> {%16[2], %16[3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK: %16[2] |-> {%0[0], %arg0, %5, %7, %8, %9, %arg2[2], %arg4[2], %15#0[2], %15#1[2]}
// CHECK: %16[3] |-> {%0[0], %arg0, %3, %4, %5, %6, %arg2[3], %arg3[3], %15#0[3], %15#1[3]}
// CHECK: impliesReady end

// CHECK: runsLater begin
// CHECK: %5 |-> {%6, %9, %15#0[2], %16[2], %15#0[3], %16[3], %13#0, %14}
// CHECK: runsLater end

// CHECK: exclusive begin
// CHECK: %16[2] |-> {%7, %8, %9, %arg4[2], %arg4[3], %15#0[3], %15#1[3], %16[3], %14}
// CHECK: %12#1 |-> {%0, %1, %2, %3, %4, %6, %7, %8, %9, %10#0, %arg3[2], %arg4[2], %arg3[3], %arg4[3], %14}
// CHECK: exclusive end

mgp.model @if : tf init {
  %0 = grt.chain.create ()
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

//
// Use of if result
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "mgp.cpu.buffer.constant"() {aValue = #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>, bRawAlign = 32 : ui64} : () -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %1 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.cpu.tensor_spec<8xf32>
// CHECK-NEXT:  %2 = "mgp.cpu.buffer.constant"() {aValue = #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>, bRawAlign = 32 : ui64} : () -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %3 = "mgp.cpu.buffer.constant"() {aValue = #M.aligned_bytes<"0x00", align 32> : !M.array<1xui8>, bRawAlign = 32 : ui64} : () -> !mgp.cpu.buffer<1>
// CHECK-NEXT:  %4 = "mgp.to_grt_bool"(%arg0, %3) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
// CHECK-NEXT:  %5:2 = "grt.if"(%4, %arg0, %0, %1, %2) ({
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.buffer<32>, %arg4: !mgp.cpu.tensor_spec<8xf32>, %arg5: !mgp.cpu.buffer<32>):
// CHECK-NEXT:    %11 = "mgp.cpu.buffer.alloc.static"(%arg2) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:    %12 = "mop.kernels.mul"(%arg2, %arg3, %arg4, %arg5, %arg4, %11, %arg4) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
// CHECK-NEXT:    "mef.output"(%12, %11) : (!mef.chain, !mgp.cpu.buffer<32>) -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^bb0(%arg2: !mef.chain, %arg3: !mgp.cpu.buffer<32>, %arg4: !mgp.cpu.tensor_spec<8xf32>, %arg5: !mgp.cpu.buffer<32>):
// CHECK-NEXT:    %11 = "mgp.cpu.buffer.alloc.static"(%arg2) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:    %12 = "mop.kernels.sub"(%arg2, %arg3, %arg4, %arg5, %arg4, %11, %arg4) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
// CHECK-NEXT:    "mef.output"(%12, %11) : (!mef.chain, !mgp.cpu.buffer<32>) -> ()
// CHECK-NEXT:  }) : (i1, !mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>) -> (!mef.chain, !mgp.cpu.buffer<32>)
// CHECK-NEXT:  %6 = "mgp.cpu.buffer.alloc.static"(%5#0) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %7 = "mop.kernels.sub"(%5#0, %5#1, %1, %2, %1, %6, %1) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
// CHECK-NEXT:  %8 = "mgp.cpu.buffer.alloc.static"(%7) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %9 = "mop.kernels.add"(%7, %6, %1, %2, %1, %8, %1) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
// CHECK-NEXT:  %10 = "mgp.cpu.tensor.create"(%9, %8, %1) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mgp.cpu.tensor<8xf32>
// CHECK-NEXT:  "mef.output"(%9, %10) : (!mef.chain, !mgp.cpu.tensor<8xf32>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<8xf32>, sym_name = "conditional_c"} : () -> ()

// CHECK: exclusive begin
// CHECK: %6 |-> {%0, %3, %4, %arg3[2], %arg3[3], %10}
// CHECK: exclusive end

mgp.model @conditional_c : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<8xf32>) {
  %a = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>
  %spec = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %b = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>
  %zero = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00", align 32> : !M.array<1xui8>
  %cond = mgp.to_grt_bool(%ch0, %zero) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
  %ch1, %c = grt.if %cond (%ch0 : !mef.chain, %a : !mgp.cpu.buffer<32>, %spec : !mgp.cpu.tensor_spec<8xf32>, %b : !mgp.cpu.buffer<32>) -> (!mef.chain, !mgp.cpu.buffer<32>) {
    %c = mgp.cpu.buffer.alloc.static %ch0, 32
    %ch1 = "mop.kernels.mul"(%ch0, %a, %spec, %b, %spec, %c, %spec) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
    mef.output %ch1, %c : !mef.chain, !mgp.cpu.buffer<32>
  } else {
    %c = mgp.cpu.buffer.alloc.static %ch0, 32
    %ch1 = "mop.kernels.sub"(%ch0, %a, %spec, %b, %spec, %c, %spec) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
    mef.output %ch1, %c : !mef.chain, !mgp.cpu.buffer<32>
  }
  %d = mgp.cpu.buffer.alloc.static %ch1, 32
  %ch2 = "mop.kernels.sub"(%ch1, %c, %spec, %b, %spec, %d, %spec) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %e = mgp.cpu.buffer.alloc.static %ch2, 32
  %ch3 = "mop.kernels.add"(%ch2, %d, %spec, %b, %spec, %e, %spec) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %out = mgp.cpu.tensor.create %ch3, %e, %spec  : !mgp.cpu.tensor<8xf32>
  mef.output %ch3, %out : !mef.chain, !mgp.cpu.tensor<8xf32>
}

