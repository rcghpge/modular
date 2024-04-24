// RUN: mo-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test | FileCheck %s

//
// Simple grt.if
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "some.op"(%arg0) : (!mef.chain) -> i1
// CHECK:   %1 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %2 = "some.op"(%arg0) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:   %3:3 = "grt.if"(%0, %arg0, %1, %2, %2) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mef.chain, %arg3: !mgp.tensor<f32>, %arg4: !mgp.tensor<f32>, %arg5: !mgp.tensor<f32>):
// CHECK:     %5 = "some.op"(%arg2) : (!mef.chain) -> !mgp.tensor<f32>
// CHECK:     %6 = "some.op"(%arg4) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%5, %6, %arg5) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg2: !mef.chain, %arg3: !mgp.tensor<f32>, %arg4: !mgp.tensor<f32>, %arg5: !mgp.tensor<f32>):
// CHECK:     %5 = "some.op"(%arg4) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:     "mef.output"(%arg3, %5, %arg5) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> ()
// CHECK:   }) : (i1, !mef.chain, !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>)
// CHECK:   %4 = "some.op"(%3#0, %3#1, %3#2) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
// CHECK:   "mef.output"(%arg0, %4) : (!mef.chain, !mgp.tensor<f32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<f32>, sym_name = "simple"} : () -> ()

// Notice if results now capture both arms
// CHECK: maybeSameAs begin
// CHECK-NEXT:  %arg0 |-> {%3[0]}
// CHECK-NEXT:  %arg1 |-> {%1[0]}
// CHECK-NEXT:  %arg2[2] |-> {%3[0], %arg0}
// CHECK-NEXT:  %arg3[2] |-> {%1}
// CHECK-NEXT:  %arg4[2] |-> {%2}
// CHECK-NEXT:  %arg5[2] |-> {%2}
// CHECK-NEXT:  %arg2[3] |-> {%3[0], %arg0}
// CHECK-NEXT:  %arg3[3] |-> {%1}
// CHECK-NEXT:  %arg4[3] |-> {%2}
// CHECK-NEXT:  %arg5[3] |-> {%2}
// CHECK-NEXT:  %3#0 |-> {%1, %5[2], %arg3[3]}
// CHECK-NEXT:  %3#1 |-> {%6[2], %5[3]}
// CHECK-NEXT:  %3#2 |-> {%2, %arg5[2], %arg5[3]}
// CHECK-NEXT: maybeSameAs end

// CHECK: isAlwaysReady({%0[0]})

// impliesReady is actually 'implies ready or never bound', so
// %3#1 captures that either one or other of the if branches will
// be executed.
// We know the if condition is ready if any of its results are ready.
// CHECK: impliesReady begin
// CHECK:  %3#0 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %0, %5[2]}
// CHECK:  %3#1 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %0, %2, %6[2], %5[3]}
// CHECK:  %3#2 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %0, %2}
// CHECK: impliesReady end

// Ops in the if branches are structurally disjoint
// CHECK: structurallyAtMostOne begin
// CHECK-NEXT: %5[2] |-> {%5[3]}
// CHECK-NEXT: %6[2] |-> {%5[3]}
// CHECK-NEXT: %5[3] |-> {%5[2], %6[2]}
// CHECK-NEXT: structurallyAtMostOne end

// CHECK: structurallyCannotOverlapRelation begin
// CHECK-NEXT: %5[2] |-> {%5[3]}
// CHECK-NEXT: %6[2] |-> {%5[3]}
// CHECK-NEXT: %5[3] |-> {%5[2], %6[2]}
// CHECK-NEXT: structurallyCannotOverlapRelation end

// runsLater is actually 'runs later or never runs'.
// CHECK: runsLater begin
// CHECK-NEXT:  %0[0] |-> {%2[0], %3[0], %0, %1, %2, %3#0, %5[2], %6[2], %5[3], %4}
// CHECK-NEXT:  %1[0] |-> {%2[0], %3[0], %0, %1, %2, %3#0, %5[2], %6[2], %5[3], %4}
// CHECK-NEXT:  %2[0] |-> {%3[0], %0, %1, %2, %3#0, %5[2], %6[2], %5[3], %4}
// CHECK-NEXT:  %3[0] |-> {%0, %1, %2, %3#0, %5[2], %6[2], %5[3], %4}
// CHECK-NEXT:  %0 |-> {%3#0, %4}
// CHECK-NEXT:  %2 |-> {%6[2], %5[3], %4}
// CHECK-NEXT:  %3#0 |-> {%4}
// CHECK-NEXT:  %5[2] |-> {%4}
// CHECK-NEXT:  %6[2] |-> {%4}
// CHECK-NEXT:  %5[3] |-> {%4}
// CHECK: runsLater end

// exclusive now accounts for nesting and the disjointness of the if branches
// CHECK: exclusive begin
// CHECK:  %0 |-> {%0[0], %2[0], %4}
// CHECK:  %5[2] |-> {%0[0], %2[0], %5[3]}
// CHECK:  %6[2] |-> {%0[0], %2[0], %5[3]}
// CHECK:  %5[3] |-> {%0[0], %2[0], %5[2], %6[2]}
// CHECK:  %4 |-> {%0[0], %2[0], %0}
// CHECK: exclusive end

mgp.model @simple : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %cxt: !mgp.context) -> (!mef.chain, !mgp.tensor<f32>) {
  %cond = "some.op"(%ch0) : (!mef.chain) -> i1
  %x = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %y = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
  %i, %j, %k = grt.if %cond (%ch0: !mef.chain, %x: !mgp.tensor<f32>, %y: !mgp.tensor<f32>, %y as %z: !mgp.tensor<f32>)
    -> (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) {
    %c = "some.op"(%ch0) : (!mef.chain) -> !mgp.tensor<f32>
    %d = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %c, %d, %z : !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  } else {
    %e = "some.op"(%y) : (!mgp.tensor<f32>) -> !mgp.tensor<f32>
    mef.output %x, %e, %z : !mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>
  }
  %l = "some.op"(%i, %j, %k) : (!mgp.tensor<f32>, !mgp.tensor<f32>, !mgp.tensor<f32>) -> !mgp.tensor<f32>
  mef.output %ch0, %l : !mef.chain, !mgp.tensor<f32>
}

//
// Lots of redundant grt.chain.create operands.
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// ### Block 0 ###
// CHECK: "mgp.model"() ({
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "grt.chain.create"(%arg0) : (!mef.chain) -> !mef.chain
// CHECK:   %1 = "grt.chain.create"(%0, %0) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %2 = "some.op"(%1) : (!mef.chain) -> !mef.chain
// CHECK:   %3 = "grt.chain.create"(%arg0, %2) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %4 = "some.op"(%3) : (!mef.chain) -> !mef.chain
// CHECK:   %5 = "some.op"(%3) : (!mef.chain) -> !mef.chain
// CHECK:   %6 = "some.op"(%3) : (!mef.chain) -> !mef.chain
// CHECK:   %7 = "grt.chain.create"(%4, %5) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %8 = "grt.chain.create"(%3, %7, %6) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %9 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %10 = "grt.chain.create"(%9, %8) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %11 = "some.op"() : () -> i1
// CHECK:   %12 = "grt.if"(%11, %9) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mef.chain):
// CHECK:     "mef.output"(%arg2) : (!mef.chain) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg2: !mef.chain):
// CHECK:     %14 = "grt.chain.create"() : () -> !mef.chain
// CHECK:     "mef.output"(%14) : (!mef.chain) -> ()
// CHECK:   }) : (i1, !mef.chain) -> !mef.chain
// CHECK:   %13 = "grt.chain.create"(%9, %10, %12) : (!mef.chain, !mef.chain, !mef.chain) -> !mef.chain
// CHECK:   "mef.output"(%13) : (!mef.chain) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> (), sym_name = "redundant"} : () -> ()

// CHECK: sameAs begin
// CHECK: %arg2[2] |-> {%9}
// CHECK: %arg2[3] |-> {%9}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK: %12 |-> {%9, %arg2[2], %14[3]}
// CHECK: maybeSameAs end

// CHECK: isAlwaysReady({%0[0], %9, %arg2[2], %arg2[3], %14[3], %12})

// CHECK: impliesReady begin
// CHECK: %2 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %0, %1}
// CHECK: impliesReady end

// CHECK: runsLater begin
// CHECK: %6 |-> {%8, %10, %13}
// CHECK: runsLater end

// CHECK: exclusive begin
// CHECK: %6 |-> {%0[0], %2[0], %0, %1, %2, %10, %13}
// CHECK: exclusive end

mgp.model @redundant : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> !mef.chain {
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
// ### Block 0 ###
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "mgp.chain.host_to_device"(%arg0, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %1 = "mgp.buffer.constant"
// CHECK:   %2 = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<> : !M.array<0xui64>, bRawDType = 1 : ui8} : () -> !mgp.tensor_spec<bool>
// CHECK:   %3 = "mgp.tensor.create"(%0, %arg1, %1, %2) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>, !mgp.tensor_spec<bool>) -> !mgp.tensor<bool>
// CHECK:   %4 = "mgp.buffer.alloc.static"(%0, %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %5 = "initialize"(%4) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
// CHECK:   %6 = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:   %7 = "mgp.tensor.create"(%5, %arg1, %4, %6) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %8 = "mgp.buffer.alloc.static"(%0, %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %9 = "initialize"(%8) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
// CHECK:   %10 = "mgp.tensor.create"(%9, %arg1, %8, %6) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %11 = "mgp.tensor.extract.buffer"(%arg1, %3) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<bool>) -> !mgp.buffer<1>
// CHECK:   %12 = "mgp.buffer.to_bool"(%0, %arg1, %11) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>) -> i1
// CHECK:   %13:2 = "grt.if"(%12, %0, %7, %10, %arg1) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mgp.chain, %arg3: !mgp.tensor<32xui8>, %arg4: !mgp.tensor<32xui8>, %arg5: !mgp.context):
// CHECK:     %18 = "mgp.tensor.extract.buffer"(%arg5, %arg4) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:     %19 = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:     %20 = "mgp.tensor.create"(%arg2, %arg5, %18, %19) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:     "mef.output"(%arg2, %20) : (!mgp.chain, !mgp.tensor<32xui8>) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg2: !mgp.chain, %arg3: !mgp.tensor<32xui8>, %arg4: !mgp.tensor<32xui8>, %arg5: !mgp.context):
// CHECK:     %18 = "mgp.tensor.extract.buffer"(%arg5, %arg3) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:     %19 = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:     %20 = "mgp.tensor.create"(%arg2, %arg5, %18, %19) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:     "mef.output"(%arg2, %20) : (!mgp.chain, !mgp.tensor<32xui8>) -> ()
// CHECK:   }) : (i1, !mgp.chain, !mgp.tensor<32xui8>, !mgp.tensor<32xui8>, !mgp.context) -> (!mgp.chain, !mgp.tensor<32xui8>)
// CHECK:   %14 = "mgp.tensor.extract.buffer"(%arg1, %13#1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<32xui8>) -> !mgp.buffer<32>
// CHECK:   %15 = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<32> : !M.array<1xui64>, bRawDType = 134 : ui8} : () -> !mgp.tensor_spec<32xui8>
// CHECK:   %16 = "mgp.tensor.create"(%13#0, %arg1, %14, %15) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<32xui8>) -> !mgp.tensor<32xui8>
// CHECK:   %17 = "mgp.chain.device_to_host"(%13#0, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%17, %16) : (!mef.chain, !mgp.tensor<32xui8>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<32xui8>, sym_name = "if"} : () -> ()


// CHECK: sameAs begin
// CHECK: %arg3[2] |-> {%7}
// CHECK: %arg4[2] |-> {%10}
// CHECK: %13#0 |-> {%0}
// CHECK: sameAs end

// CHECK: maybeSameAs begin
// CHECK: %13#0 |-> {%0, %arg2[2], %arg2[3]}
// The second if result may alias via the tensor.{create,extract}s.
// CHECK: %13#1 |-> {%4, %7, %8, %10, %arg4[2], %18[2], %20[2], %arg3[3], %18[3], %20[3]}
// CHECK: maybeSameAs end

// CHECK: impliesReady begin
// CHECK: %20[2] |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %arg1, %0, %6, %8, %9, %10, %arg2[2], %arg4[2], %arg5[2], %18[2], %19[2]}
// CHECK: %20[3] |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %arg1, %0, %4, %5, %6, %7, %arg2[3], %arg3[3], %arg5[3], %18[3], %19[3]}
// CHECK: impliesReady end

// CHECK: runsLater begin
// CHECK: %6 |-> {%7, %10, %18[2], %20[2], %18[3], %20[3], %14, %16}
// CHECK: runsLater end

// CHECK: exclusive begin
// The tensor.{create,extract} aliasing excludes many exclusive entries.
// CHECK: %20[2] |-> {%0[0], %2[0], %9, %18[3], %19[3], %20[3]}
// CHECK: %13#1 |-> {%0[0], %2[0], %1, %2, %3, %5, %9, %11}
// CHECK: exclusive end

mgp.model @if : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.tensor<32xui8>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x01", align 8> : !M.array<1xui8>
  %1 = mgp.tensor_spec.create : !mgp.tensor_spec<bool>
  %2 = mgp.tensor.create %ch1, %ctx[1],  %0, %1  : !mgp.tensor<bool>
  %3 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  // NOTE: Since %11#1 maybe the same as %6 and %9, and those are considered
  //       to be maybe the same as %3 and %7, the CheckBufferReadWritePass will
  //       complain that %3 and %7 may be concurrently mutated. Fixing that
  //       would require a more discriminating CheckBufferReadWritePass.
  %4 = "initialize"(%3) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
  %5 = mgp.tensor_spec.create  : !mgp.tensor_spec<32xui8>
  %6 = mgp.tensor.create %4, %ctx[1],  %3, %5  : !mgp.tensor<32xui8>
  %7 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %8 = "initialize"(%7) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<32>) -> !mgp.chain
  %9 = mgp.tensor.create %8, %ctx[1],  %7, %5  : !mgp.tensor<32xui8>
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<bool>
  %10 = mgp.buffer.to_bool %ch1, %ctx[1], %bufferResult : !mgp.buffer<1>
  %11:2 = grt.if %10 (%ch1: !mgp.chain, %6: !mgp.tensor<32xui8>, %9: !mgp.tensor<32xui8>, %ctx: !mgp.context)  -> (!mgp.chain, !mgp.tensor<32xui8>) {
    %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<32xui8>
    %tensorSpecResult_3 = mgp.tensor_spec.create : !mgp.tensor_spec<32xui8>
    %13 = mgp.tensor.create %ch1, %ctx[1],  %bufferResult_2, %tensorSpecResult_3 : !mgp.tensor<32xui8>
    mef.output %ch1, %13 : !mgp.chain, !mgp.tensor<32xui8>
  } else {
    %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %6 : !mgp.tensor<32xui8>
    %tensorSpecResult_3 = mgp.tensor_spec.create : !mgp.tensor_spec<32xui8>
    %13 = mgp.tensor.create %ch1, %ctx[1],  %bufferResult_2, %tensorSpecResult_3 : !mgp.tensor<32xui8>
    mef.output %ch1, %13 : !mgp.chain, !mgp.tensor<32xui8>
  }
  %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %11#1 : !mgp.tensor<32xui8>
  %tensorSpecResult_1 = mgp.tensor_spec.create : !mgp.tensor_spec<32xui8>
  %12 = mgp.tensor.create %11#0, %ctx[1], %bufferResult_0, %tensorSpecResult_1 : !mgp.tensor<32xui8>
  %ch2 = mgp.chain.device_to_host %11#0, %ctx[1]
  mef.output %ch2, %12 : !mef.chain, !mgp.tensor<32xui8>
}

//
// Use of if result
//

// For reference here's the model with vars matching the checks (modulo the block ids).
// CHECK: "mgp.model"() ({
// ### Block 0 ###
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "mgp.chain.host_to_device"(%arg0, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %1 = "mgp.buffer.constant"
// CHECK:   %2 = "mgp.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %3 = "mgp.buffer.constant"
// CHECK:   %4 = "mgp.buffer.constant"
// CHECK:   %5 = "mgp.buffer.to_bool"(%0, %arg1, %4) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<1>) -> i1
// CHECK:   %6:2 = "grt.if"(%5, %0, %1, %2, %3, %arg1) ({
// ### Block 2 ###
// CHECK:   ^bb0(%arg2: !mgp.chain, %arg3: !mgp.buffer<32>, %arg4: !mgp.tensor_spec<8xf32>, %arg5: !mgp.buffer<32>, %arg6: !mgp.context):
// CHECK:     %13 = "mgp.buffer.alloc.static"(%arg2, %arg6) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:     %14 = "mul"(%arg2, %arg3, %arg5, %13) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:     "mef.output"(%14, %13) : (!mgp.chain, !mgp.buffer<32>) -> ()
// CHECK:   }, {
// ### Block 3 ###
// CHECK:   ^bb0(%arg2: !mgp.chain, %arg3: !mgp.buffer<32>, %arg4: !mgp.tensor_spec<8xf32>, %arg5: !mgp.buffer<32>, %arg6: !mgp.context):
// CHECK:     %13 = "mgp.buffer.alloc.static"(%arg2, %arg6) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:     %14 = "sub"(%arg2, %arg3, %arg5, %13) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:     "mef.output"(%14, %13) : (!mgp.chain, !mgp.buffer<32>) -> ()
// CHECK:   }) : (i1, !mgp.chain, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>, !mgp.buffer<32>, !mgp.context) -> (!mgp.chain, !mgp.buffer<32>)
// CHECK:   %7 = "mgp.buffer.alloc.static"(%6#0, %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %8 = "sub"(%6#0, %6#1, %3, %7) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %9 = "mgp.buffer.alloc.static"(%8, %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %10 = "add"(%8, %7, %3, %9) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %11 = "mgp.tensor.create"(%10, %arg1, %9, %2) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %12 = "mgp.chain.device_to_host"(%10, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%12, %11) : (!mef.chain, !mgp.tensor<8xf32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<8xf32>, sym_name = "conditional_c"} : () -> ()

// CHECK: exclusive begin
// CHECK: %7 |-> {%0[0], %2[0], %0, %1, %4, %5, %arg2[2], %arg3[2], %arg2[3], %arg3[3], %11, %12}
// CHECK: exclusive end

mgp.model @conditional_c : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.tensor<8xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %a = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>
  %spec = mgp.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %b = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>
  %zero = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x00", align 32> : !M.array<1xui8>
  %cond = mgp.buffer.to_bool %ch1, %ctx[1], %zero : !mgp.buffer<1>
  %ch3, %c = grt.if %cond (%ch1 : !mgp.chain, %a : !mgp.buffer<32>, %spec : !mgp.tensor_spec<8xf32>, %b : !mgp.buffer<32>, %ctx : !mgp.context) -> (!mgp.chain, !mgp.buffer<32>) {
    %c = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
    %ch2 = "mul"(%ch1, %a, %b, %c) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
    mef.output %ch2, %c : !mgp.chain, !mgp.buffer<32>
  } else {
    %c = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
    %ch2 = "sub"(%ch1, %a, %b, %c) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
    mef.output %ch2, %c : !mgp.chain, !mgp.buffer<32>
  }
  %d = mgp.buffer.alloc.static %ch3, %ctx[1] : !mgp.buffer<32>
  %ch4 = "sub"(%ch3, %c, %b, %d) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %e = mgp.buffer.alloc.static %ch4, %ctx[1] : !mgp.buffer<32>
  %ch5 = "add"(%ch4, %d, %b, %e) {tempInOutSig=#M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %out = mgp.tensor.create %ch5, %ctx[1],  %e, %spec : !mgp.tensor<8xf32>
  %ch6 = mgp.chain.device_to_host %ch5, %ctx[1]
  mef.output %ch6, %out : !mef.chain, !mgp.tensor<8xf32>
}
