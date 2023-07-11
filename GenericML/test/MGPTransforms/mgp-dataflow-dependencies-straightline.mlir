// RUN: mgp-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test --check-buffer-read-write | FileCheck %s

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

// sameAs only ties the 'init' results to the 'execute args
// CHECK: sameAs begin
// CHECK-NEXT: %arg0 |-> {%0[0]}
// CHECK-NEXT: %arg1 |-> {%1[0]}
// CHECK-NEXT: sameAs end

// maybeSameAs == sameAs
// CHECK: maybeSameAs begin
// CHECK-NEXT: %arg0 |-> {%0[0]}
// CHECK-NEXT: %arg1 |-> {%1[0]}
// CHECK-NEXT: maybeSameAs end

// The two always ready chains (this is up to sameAs equivalence)
// CHECK: isAlwaysReady({%0[0], %arg0, %0})

// impliesReady captures transitive operand dependencies (plus identity)
// CHECK: impliesReady begin
// CHECK-NEXT: %arg0 |-> {%0[0]}
// CHECK-NEXT: %arg1 |-> {%1[0]}
// CHECK-NEXT: %1 |-> {%0[0], %arg0}
// CHECK-NEXT: %2 |-> {%0[0], %arg0, %0, %1}
// CHECK-NEXT: %3 |-> {%0[0], %arg0}
// CHECK-NEXT: %4 |-> {%0[0], %arg0, %0, %3}
// CHECK-NEXT: %5 |-> {%0[0], %arg0, %0, %1, %2, %3, %4}
// CHECK-NEXT: impliesReady end

// Nothing is structurally disjoint since no control flow
// CHECK: structurallyDisjoint begin
// CHECK-NEXT: structurallyDisjoint end

// runsLater for which ops (identified by their result values) run after which
// CHECK: runsLater begin
// CHECK-NEXT: %0[0] |-> {%1, %2, %3, %4, %5}
// CHECK-NEXT: %0 |-> {%2, %4, %5}
// CHECK-NEXT: %1 |-> {%2, %5}
// CHECK-NEXT: %2 |-> {%5}
// CHECK-NEXT: %3 |-> {%4, %5}
// CHECK-NEXT: %4 |-> {%5}
// CHECK-NEXT: runsLater end

// exclusive tells us which values can't have overlapping lifetimes.
// CHECK: exclusive begin
// CHECK-NEXT: %0[0] |-> {%5}
// CHECK-NEXT: %arg0 |-> {%5}
// CHECK-NEXT: %0 |-> {%5}
// CHECK-NEXT: %1 |-> {%5}
// CHECK-NEXT: %3 |-> {%5}
// CHECK-NEXT: %5 |-> {%0[0], %arg0, %0, %1, %3}
// CHECK-NEXT: exclusive end

mgp.model @simple : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> !mef.chain {
  %ch1 = grt.chain.create ()
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

// Spot check impliesReady
// CHECK: impliesReady begin
// CHECK: %10 |-> {%0[0], %arg0, %0, %1, %2, %3, %4, %5, %6#0, %6#1, %7#0, %7#1, %8, %9}
// CHECK: impliesReady end

// Spot check runsLater
// CHECK: runsLater begin
// CHECK: %12#0 |-> {%16, %17, %18#0, %22, %23}
// CHECK: runsLater end

// Spot check exclusive. Note that %5 -> {%7, %13}, and %16 is the earliest
// point which implies both %7 and %13 are ready.
// CHECK: exclusive begin
// CHECK: %5 |-> {%16, %17, %18#0, %18#1, %22, %23}
// CHECK: exclusive end

mgp.model @straightline : tf init {
  %0 = grt.chain.create ()
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

//
// mgp.cpu.buffer.split.borrow implicitly takes reference to argument
//

// For reference here's the model with vars matching the checks:
// CHECK: "mgp.model"() ({
// CHECK-NEXT:  %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK-NEXT:  %1 = "mgp.context.create"() {numSlots = 0 : ui64} : () -> !mgp.context
// CHECK-NEXT:  "mef.output"(%0, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK-NEXT:}, {
// CHECK-NEXT:^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK-NEXT:  %0 = "mgp.cpu.buffer.alloc.static"(%arg0) {aSize = 6 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<6>
// CHECK-NEXT:  %1 = "mgp.cpu.buffer.set"(%0) {value = 42 : ui8} : (!mgp.cpu.buffer<6>) -> !mef.chain
// CHECK-NEXT:  %2:2 = "mgp.cpu.buffer.split.borrow"(%arg0, %0) {outBufferRawSizes = #M.dense_array<2, 4> : !M.array<2xui64>} : (!mef.chain, !mgp.cpu.buffer<6>) -> (!mgp.cpu.buffer<2>, !mgp.cpu.buffer<4>)
// CHECK-NEXT:  %3 = "mgp.debug.cpu.buffer.print"(%1, %2#0) : (!mef.chain, !mgp.cpu.buffer<2>) -> !mef.chain
// CHECK-NEXT:  %4 = "mgp.debug.cpu.buffer.print"(%3, %2#1) : (!mef.chain, !mgp.cpu.buffer<4>) -> !mef.chain
// CHECK-NEXT:  %5 = "mgp.cpu.buffer.alloc.static"(%4) {aSize = 6 : ui64, bRawAlign = 18446744073709551615 : ui64} : (!mef.chain) -> !mgp.cpu.buffer<6>
// CHECK-NEXT:  %6 = "mgp.cpu.buffer.concat"(%4, %5, %2#0, %2#1) : (!mef.chain, !mgp.cpu.buffer<6>, !mgp.cpu.buffer<2>, !mgp.cpu.buffer<4>) -> !mef.chain
// CHECK-NEXT:  %7 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<6> : !M.array<1xui64>, bRawDType = 135 : ui8} : () -> !mgp.cpu.tensor_spec<6xsi8>
// CHECK-NEXT:  %8 = "mgp.cpu.tensor.create"(%6, %5, %7) : (!mef.chain, !mgp.cpu.buffer<6>, !mgp.cpu.tensor_spec<6xsi8>) -> !mgp.cpu.tensor<6xsi8>
// CHECK-NEXT:  "mef.output"(%6, %8) : (!mef.chain, !mgp.cpu.tensor<6xsi8>) -> ()
// CHECK-NEXT:}) {frameworkName = "tf", inputOutputTypes = () -> !mgp.cpu.tensor<6xsi8>, sym_name = "safe_split_borrow"} : () -> ()

// CHECK: maybeSameAs begin
// Note %2#* is considered maybe the same as %0.
// CHECK: %2#0 |-> {%0[0], %arg0, %0}
// CHECK: %2#1 |-> {%0[0], %arg0, %0}
// CHECK: maybeSameAs end

// CHECK: exclusive begin
// Note %0 could overlap with %3, %4, %5 and %6 if it were not for
// the implicit lifetime extension.
// CHECK: %0 |-> {%8}
// CHECK: exclusive end

mgp.model @safe_split_borrow : tf init {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1 : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.cpu.tensor<6xsi8>) {
  %in_buf = mgp.cpu.buffer.alloc.static %ch0, 6
  %ch1 = mgp.cpu.buffer.set %in_buf, 42 : !mgp.cpu.buffer<6>
  %in_buf_borrows:2 = mgp.cpu.buffer.split.borrow %ch0, %in_buf
    : !mgp.cpu.buffer<6> -> (!mgp.cpu.buffer<2>, !mgp.cpu.buffer<4>)
  %ch2 = mgp.debug.cpu.buffer.print %ch1, %in_buf_borrows#0 : !mgp.cpu.buffer<2>
  %ch3 = mgp.debug.cpu.buffer.print %ch2, %in_buf_borrows#1 : !mgp.cpu.buffer<4>
  %out_buf = mgp.cpu.buffer.alloc.static %ch3, 6
  %ch4 = mgp.cpu.buffer.concat %ch3, %out_buf, %in_buf_borrows#0, %in_buf_borrows#1 : (!mgp.cpu.buffer<2>, !mgp.cpu.buffer<4>) -> !mgp.cpu.buffer<6>
  %out_spec = mgp.cpu.tensor_spec.create : !mgp.cpu.tensor_spec<6xsi8>
  %ten = mgp.cpu.tensor.create %ch4, %out_buf, %out_spec : !mgp.cpu.tensor<6xsi8>
  mef.output %ch4, %ten  : !mef.chain, !mgp.cpu.tensor<6xsi8>
}

