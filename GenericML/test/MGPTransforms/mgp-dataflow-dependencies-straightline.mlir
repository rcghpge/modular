// RUN: mgp-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test | FileCheck %s

//
// Simple graph with no control flow
//

// For reference here's the model with vars matching the checks:
// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "some.op"(%arg0) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %2 = "some.op"(%1, %0) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %3 = "some.op"(%arg0) : (!mef.chain) -> !mef.chain
// CHECK-NEXT:  %4 = "some.op"(%3, %0) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  %5 = "some.op"(%2, %4) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK-NEXT:  "mef.output"(%5) : (!mef.chain) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> (), sym_name = "simple"} : () -> ()

// sameAs is just the identity
// CHECK: sameAs(%arg0, %arg0) = true
// CHECK: sameAs(%arg0, %arg1) = false
// CHECK: sameAs(%arg0, %0) = false
// CHECK: sameAs(%arg1, %arg0) = false
// CHECK: sameAs(%arg1, %arg1) = true
// CHECK: sameAs(%arg1, %0) = false
// CHECK: sameAs(%0, %arg0) = false
// CHECK: sameAs(%0, %arg1) = false
// CHECK: sameAs(%0, %0) = true
// etc

// Only %0 is always ready
// CHECK: isAlwaysReady(%arg0) = false
// CHECK: isAlwaysReady(%arg1) = false
// CHECK: isAlwaysReady(%0) = true
// etc

// impliesReady is mostly identity, with transitive closure for some.op args
// CHECK: impliesReady(%arg0, %arg0) = true
// CHECK: impliesReady(%arg0, %arg1) = false
// CHECK: impliesReady(%arg0, %0) = false
// CHECK: impliesReady(%arg1, %arg0) = false
// CHECK: impliesReady(%arg1, %arg1) = true
// CHECK: impliesReady(%arg1, %0) = false
// CHECK: impliesReady(%0, %arg0) = false
// CHECK: impliesReady(%0, %arg1) = false
// CHECK: impliesReady(%0, %0) = true

// CHECK: impliesReady(%1, %arg0) = true

// CHECK: impliesReady(%2, %arg0) = true
// CHECK: impliesReady(%2, %0) = true
// CHECK: impliesReady(%2, %1) = true

// CHECK: impliesReady(%3, %arg0) = true

// CHECK: impliesReady(%4, %arg0) = true
// CHECK: impliesReady(%4, %0) = true
// CHECK: impliesReady(%4, %2) = false
// CHECK: impliesReady(%4, %3) = true

// CHECK: impliesReady(%5, %arg0) = true
// CHECK: impliesReady(%5, %0) = true
// CHECK: impliesReady(%5, %1) = true
// CHECK: impliesReady(%5, %2) = true
// CHECK: impliesReady(%5, %3) = true
// CHECK: impliesReady(%5, %4) = true

mgp.model @simple : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> !mef.chain {
  %ch1 = grt.chain.create
  %ch2 = "some.op"(%ch0) : (!mef.chain) -> !mef.chain
  %ch3 = "some.op"(%ch2, %ch1) : (!mef.chain, !mef.chain) -> !mef.chain
  %ch4 = "some.op"(%ch0) : (!mef.chain) -> !mef.chain
  %ch5 = "some.op"(%ch4, %ch1) : (!mef.chain, !mef.chain) -> !mef.chain
  %ch6 = "some.op"(%ch3, %ch5) : (!mef.chain, !mef.chain) -> !mef.chain
  mef.output %ch6 : !mef.chain
}

//
// Realistic graph with no control flow.
//

// For reference here's the model with vars matching the checks:
// CHECK: "mgp.model"() ({
//              ### Block 0 ###
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
//            ### Block 1 ###
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "mgp.cpu.buffer.constant"() {aValue = #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>, bRawAlign = 32 : ui64} : () -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %1 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.cpu.tensor_spec<8xf32>
// CHECK-NEXT:  %2 = "mgp.cpu.tensor.create"(%arg0, %0, %1) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mgp.cpu.tensor<8xf32>
// CHECK-NEXT:  %3 = "mgp.cpu.buffer.constant"() {aValue = #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>, bRawAlign = 32 : ui64} : () -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %4 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.cpu.tensor_spec<8xf32>
// CHECK-NEXT:  %5 = "mgp.cpu.tensor.create"(%arg0, %3, %4) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mgp.cpu.tensor<8xf32>
// CHECK-NEXT:  %6:2 = "mgp.cpu.tensor.extract"(%2) : (!mgp.cpu.tensor<8xf32>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>)
// CHECK-NEXT:  %7:2 = "mgp.cpu.tensor.extract"(%5) : (!mgp.cpu.tensor<8xf32>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>)
// CHECK-NEXT:  %8 = "mgp.cpu.buffer.alloc.static"(%arg0) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %9 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.cpu.tensor_spec<8xf32>
// CHECK-NEXT:  %10 = "mop.kernels.mul"(%arg0, %6#0, %6#1, %7#0, %7#1, %8, %9) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
// CHECK-NEXT:  %11 = "mgp.cpu.tensor.create"(%10, %8, %9) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mgp.cpu.tensor<8xf32>
// CHECK-NEXT:  %12:2 = "mgp.cpu.tensor.extract"(%11) : (!mgp.cpu.tensor<8xf32>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>)
// CHECK-NEXT:  %13:2 = "mgp.cpu.tensor.extract"(%5) : (!mgp.cpu.tensor<8xf32>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>)
// CHECK-NEXT:  %14 = "mgp.cpu.buffer.alloc.static"(%arg0) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %15 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.cpu.tensor_spec<8xf32>
// CHECK-NEXT:  %16 = "mop.kernels.sub"(%arg0, %12#0, %12#1, %13#0, %13#1, %14, %15) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
// CHECK-NEXT:  %17 = "mgp.cpu.tensor.create"(%16, %14, %15) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mgp.cpu.tensor<8xf32>
// CHECK-NEXT:  %18:2 = "mgp.cpu.tensor.extract"(%17) : (!mgp.cpu.tensor<8xf32>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>)
// CHECK-NEXT:  %19:2 = "mgp.cpu.tensor.extract"(%11) : (!mgp.cpu.tensor<8xf32>) -> (!mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>)
// CHECK-NEXT:  %20 = "mgp.cpu.buffer.alloc.static"(%arg0) {aSize = 32 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<32>
// CHECK-NEXT:  %21 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.cpu.tensor_spec<8xf32>
// CHECK-NEXT:  %22 = "mop.kernels.add"(%arg0, %18#0, %18#1, %19#0, %19#1, %20, %21) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
// CHECK-NEXT:  %23 = "mgp.cpu.tensor.create"(%22, %20, %21) : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mgp.cpu.tensor<8xf32>
// CHECK-NEXT:  "mef.output"(%arg0, %23) : (!mef.chain, !mgp.cpu.tensor<8xf32>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<8xf32>, sym_name = "straightline"} : () -> ()

// sameAs is just the identity, just spot check.
// CHECK: sameAs(%arg0, %arg0) = true
// CHECK: sameAs(%arg1, %arg1) = true
// CHECK: sameAs(%0, %0) = true
// CHECK: sameAs(%7#0, %7#0) = true
// CHECK: sameAs(%7#1, %7#1) = true

// Just spot check impliesReady.
// CHECK: impliesReady(%10, %arg0) = true
// CHECK: impliesReady(%10, %0) = true
// CHECK: impliesReady(%10, %1) = true
// CHECK: impliesReady(%10, %2) = true
// CHECK: impliesReady(%10, %3) = true
// CHECK: impliesReady(%10, %4) = true
// CHECK: impliesReady(%10, %5) = true
// CHECK: impliesReady(%10, %6#0) = true
// CHECK: impliesReady(%10, %6#1) = true
// CHECK: impliesReady(%10, %7#0) = true
// CHECK: impliesReady(%10, %7#1) = true
// CHECK: impliesReady(%10, %8) = true
// CHECK: impliesReady(%10, %9) = true
// CHECK: impliesReady(%10, %10) = true

mgp.model @straightline : tf init {
  %0 = grt.chain.create
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<8xf32>) {
  %0 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %2 = mgp.cpu.tensor.create %arg0, %0, %1  : !mgp.cpu.tensor<8xf32>
  %3 = mgp.cpu.buffer.constant #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>
  %4 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %5 = mgp.cpu.tensor.create %arg0, %3, %4  : !mgp.cpu.tensor<8xf32>
  %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %2 : !mgp.cpu.tensor<8xf32>
  %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
  %6 = mgp.cpu.buffer.alloc.static %arg0, 32
  %7 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %8 = "mop.kernels.mul"(%arg0, %bufferResult, %tensorSpecResult, %bufferResult_0, %tensorSpecResult_1, %6, %7) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %9 = mgp.cpu.tensor.create %8, %6, %7  : !mgp.cpu.tensor<8xf32>
  %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %9 : !mgp.cpu.tensor<8xf32>
  %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %5 : !mgp.cpu.tensor<8xf32>
  %10 = mgp.cpu.buffer.alloc.static %arg0, 32
  %11 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %12 = "mop.kernels.sub"(%arg0, %bufferResult_2, %tensorSpecResult_3, %bufferResult_4, %tensorSpecResult_5, %10, %11) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %13 = mgp.cpu.tensor.create %12, %10, %11  : !mgp.cpu.tensor<8xf32>
  %bufferResult_6, %tensorSpecResult_7 = mgp.cpu.tensor.extract %13 : !mgp.cpu.tensor<8xf32>
  %bufferResult_8, %tensorSpecResult_9 = mgp.cpu.tensor.extract %9 : !mgp.cpu.tensor<8xf32>
  %14 = mgp.cpu.buffer.alloc.static %arg0, 32
  %15 = mgp.cpu.tensor_spec.create  : !mgp.cpu.tensor_spec<8xf32>
  %16 = "mop.kernels.add"(%arg0, %bufferResult_6, %tensorSpecResult_7, %bufferResult_8, %tensorSpecResult_9, %14, %15) {numBackOutputBuffers = 1 : ui32} : (!mef.chain, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>, !mgp.cpu.buffer<32>, !mgp.cpu.tensor_spec<8xf32>) -> !mef.chain
  %17 = mgp.cpu.tensor.create %16, %14, %15  : !mgp.cpu.tensor<8xf32>
  mef.output %arg0, %17 : !mef.chain, !mgp.cpu.tensor<8xf32>
}
