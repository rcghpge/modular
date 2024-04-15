// RUN: mo-opt %s --mlir-disable-threading --allow-unregistered-dialect --async-dataflow-dependencies-test --check-buffer-read-write | FileCheck %s

//
// Simple graph with no control flow
//

// For reference here's the model with vars matching the checks:
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
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "some.op"(%arg0) : (!mef.chain) -> !mef.chain
// CHECK:   %2 = "some.op"(%1, %0) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %3 = "some.op"(%arg0) : (!mef.chain) -> !mef.chain
// CHECK:   %4 = "some.op"(%3, %0) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   %5 = "some.op"(%2, %4) : (!mef.chain, !mef.chain) -> !mef.chain
// CHECK:   "mef.output"(%5) : (!mef.chain) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> (), sym_name = "simple"} : () -> ()

// sameAs only ties the 'init' results to the 'execute args
// CHECK: sameAs begin
// CHECK-NEXT: %arg0 |-> {%3[0]}
// CHECK-NEXT: %arg1 |-> {%1[0]}
// CHECK-NEXT: sameAs end

// maybeSameAs == sameAs
// CHECK: maybeSameAs begin
// CHECK-NEXT: %arg0 |-> {%3[0]}
// CHECK-NEXT: %arg1 |-> {%1[0]}
// CHECK-NEXT: maybeSameAs end

// The two always ready chains (this is up to sameAs equivalence)
// CHECK: isAlwaysReady({%0[0], %0})

// impliesReady captures transitive operand dependencies (plus identity)
// CHECK: impliesReady begin
// CHECK: %2[0] |-> {%0[0], %1[0]}
// CHECK: %3[0] |-> {%0[0], %1[0], %2[0]}
// CHECK: %arg0 |-> {%0[0], %1[0], %2[0], %3[0]}
// CHECK: %arg1 |-> {%1[0]}
// CHECK: %1 |-> {%0[0], %1[0], %2[0], %3[0], %arg0}
// CHECK: %2 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %0, %1}
// CHECK: %3 |-> {%0[0], %1[0], %2[0], %3[0], %arg0}
// CHECK: %4 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %0, %3}
// CHECK: %5 |-> {%0[0], %1[0], %2[0], %3[0], %arg0, %0, %1, %2, %3, %4}
// CHECK-NEXT: impliesReady end

// No control flow so nothing in these relations.
// CHECK: structurallyAtMostOne begin
// CHECK-NEXT: structurallyAtMostOne end
// CHECK: structurallyCannotOverlapRelation begin
// CHECK-NEXT: structurallyCannotOverlapRelation end

// runsLater for which ops (identified by their result values) run after which
// CHECK: runsLater begin
// CHECK-NEXT: %0[0] |-> {%2[0], %3[0], %1, %2, %3, %4, %5}
// CHECK-NEXT: %1[0] |-> {%2[0], %3[0], %1, %2, %3, %4, %5}
// CHECK-NEXT: %2[0] |-> {%3[0], %1, %2, %3, %4, %5}
// CHECK-NEXT: %3[0] |-> {%1, %2, %3, %4, %5}
// CHECK-NEXT: %0 |-> {%2, %4, %5}
// CHECK-NEXT: %1 |-> {%2, %5}
// CHECK-NEXT: %2 |-> {%5}
// CHECK-NEXT: %3 |-> {%4, %5}
// CHECK-NEXT: %4 |-> {%5}
// CHECK-NEXT: runsLater end

// exclusive tells us which values can't have overlapping lifetimes.
// CHECK: exclusive begin
// CHECK-NEXT: %0[0] |-> {%3[0], %arg0, %1, %2, %3, %4, %5}
// CHECK-NEXT: %2[0] |-> {%1, %2, %3, %4, %5}
// CHECK-NEXT: %3[0] |-> {%0[0]}
// CHECK-NEXT: %arg0 |-> {%0[0]}
// CHECK-NEXT: %0 |-> {%5}
// CHECK-NEXT: %1 |-> {%0[0], %2[0], %5}
// CHECK-NEXT: %2 |-> {%0[0], %2[0]}
// CHECK-NEXT: %3 |-> {%0[0], %2[0], %5}
// CHECK-NEXT: %4 |-> {%0[0], %2[0]}
// CHECK-NEXT: %5 |-> {%0[0], %2[0], %0, %1, %3}
// CHECK-NEXT: exclusive end

mgp.model @simple : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
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
// ### Block 0 ###
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// ### Block 1 ###
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %[[V0:.*]] = "mgp.chain.host_to_device"(%arg0, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %[[V1:.*]] = "mgp.buffer.constant"
// CHECK:   %[[V2:.*]] = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[V3:.*]] = "mgp.tensor.create"(%[[V0]], %arg1, %[[V1]], %[[V2]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[V4:.*]] = "mgp.buffer.constant"
// CHECK:   %[[V5:.*]] = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[V6:.*]] = "mgp.tensor.create"(%[[V0]], %arg1, %[[V4]], %[[V5]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[V7:.*]] = "mgp.tensor.extract.buffer"(%arg1, %[[V3]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[V8:.*]] = "mgp.tensor.extract.buffer"(%arg1, %[[V6]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[V9:.*]] = "mgp.buffer.alloc.static"(%[[V0]], %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[V10:.*]] = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[V11:.*]] = "mul"(%[[V0]], %[[V7]], %[[V8]], %[[V9]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[V12:.*]] = "mgp.tensor.create"(%[[V11]], %arg1, %[[V9]], %[[V10]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[V13:.*]] = "mgp.tensor.extract.buffer"(%arg1, %[[V12]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[V14:.*]] = "mgp.tensor.extract.buffer"(%arg1, %[[V6]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[V15:.*]] = "mgp.buffer.alloc.static"(%[[V0]], %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[V16:.*]] = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[V17:.*]] = "sub"(%[[V0]], %[[V13]], %[[V14]], %[[V15]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[V18:.*]] = "mgp.tensor.create"(%[[V17]], %arg1, %[[V15]], %[[V16]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   %[[V19:.*]] = "mgp.tensor.extract.buffer"(%arg1, %[[V18]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[V20:.*]] = "mgp.tensor.extract.buffer"(%arg1, %[[V12]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.context, !mgp.tensor<8xf32>) -> !mgp.buffer<32>
// CHECK:   %[[V21:.*]] = "mgp.buffer.alloc.static"(%[[V0]], %arg1) {aRuntimeSlot = 1 : ui64, bSize = 32 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<32>
// CHECK:   %[[V22:.*]] = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<8> : !M.array<1xui64>, bRawDType = 66 : ui8} : () -> !mgp.tensor_spec<8xf32>
// CHECK:   %[[V23:.*]] = "add"(%[[V0]], %[[V19]], %[[V20]], %[[V21]]) {tempInOutSig = #M.inout_sig<"iiio">} : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
// CHECK:   %[[V24:.*]] = "mgp.tensor.create"(%[[V23]], %arg1, %[[V21]], %[[V22]]) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<32>, !mgp.tensor_spec<8xf32>) -> !mgp.tensor<8xf32>
// CHECK:   "mef.output"(%arg0, %[[V24]]) : (!mef.chain, !mgp.tensor<8xf32>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<8xf32>, sym_name = "straightline"} : () -> ()

// Spot check impliesReady
// CHECK: impliesReady begin
// CHECK: %[[V11]] |-> {%[[V0]][0], %[[V1]][0], %[[V2]][0], %[[V3]][0], %arg0, %arg1, %[[V0]], %[[V1]], %[[V2]], %[[V3]], %[[V4]], %[[V5]], %[[V6]], %[[V7]], %[[V8]], %[[V9]]}
// CHECK: impliesReady end

// Spot check runsLater
// CHECK: runsLater begin
// CHECK: %[[V13]] |-> {%[[V17]], %[[V18]], %[[V19]], %[[V23]], %[[V24]]}
// CHECK: runsLater end

// Spot check exclusive. Note that %[[V6]] flows into %[[V8]] and %[[V14]], and %[[V17]] is the
// earliest point which implies both %[[V8]] and %[[V14]] are ready.
// CHECK: exclusive begin
// CHECK: %[[V6]] |-> {%[[V0]][0], %[[V2]][0], %[[V18]], %[[V19]], %[[V23]], %[[V24]]}
// CHECK: exclusive end

mgp.model @straightline : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.tensor<8xf32>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %0 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000803F0000004000004040000080400000A0400000C0400000E04000000041", align 32> : !M.array<32xui8>
  %1 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %2 = mgp.tensor.create %ch1, %ctx[1],  %0, %1  : !mgp.tensor<8xf32>
  %3 = mgp.buffer.constant %ch0, %ctx[1], #M.aligned_bytes<"0x0000204100002041000020410000204100002041000020410000204100002041", align 32> : !M.array<32xui8>
  %4 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %5 = mgp.tensor.create %ch1, %ctx[1],  %3, %4  : !mgp.tensor<8xf32>
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %2 : !mgp.tensor<8xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %5 : !mgp.tensor<8xf32>
  %6 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %7 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %8 = "mul"(%ch1, %bufferResult, %bufferResult_0, %6) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %9 = mgp.tensor.create %8, %ctx[1],  %6, %7  : !mgp.tensor<8xf32>
  %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<8xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %ctx[1], %5 : !mgp.tensor<8xf32>
  %10 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %11 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %12 = "sub"(%ch1, %bufferResult_2, %bufferResult_4, %10) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %13 = mgp.tensor.create %12, %ctx[1],  %10, %11  : !mgp.tensor<8xf32>
  %bufferResult_6 = mgp.tensor.extract.buffer %ctx[1], %13 : !mgp.tensor<8xf32>
  %bufferResult_8 = mgp.tensor.extract.buffer %ctx[1], %9 : !mgp.tensor<8xf32>
  %14 = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<32>
  %15 = mgp.cpu.tensor_spec.create  : !mgp.tensor_spec<8xf32>
  %16 = "add"(%ch1, %bufferResult_6, %bufferResult_8, %14) { tempInOutSig=#M.inout_sig<"iiio"> } : (!mgp.chain, !mgp.buffer<32>, !mgp.buffer<32>, !mgp.buffer<32>) -> !mgp.chain
  %17 = mgp.tensor.create %16, %ctx[1],  %14, %15  : !mgp.tensor<8xf32>
  mef.output %ch0, %17 : !mef.chain, !mgp.tensor<8xf32>
}

//
// mgp.cpu.buffer.split.borrow implicitly takes reference to argument
//

// For reference here's the model with vars matching the checks:
// CHECK: "mgp.model"() ({
// CHECK:   %0 = "grt.chain.create"() : () -> !mef.chain
// CHECK:   %1 = "mgp.context.create"() {numSlots = 2 : ui64} : () -> !mgp.context
// CHECK:   %2 = "mgp.check_runtime"(%0, %1) {aSlot = 0 : ui64, bDevices = #M.device_spec_collection<host = <"cpu", 0>, devices = [<ref = <"cpu", 0>, target = <triple = "unknown", arch = "">>]>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   %3 = "mgp.runtime.create"(%2, %1) {aRuntimeSlot = 1 : ui64, bDeviceSpecMapSlot = 0 : ui64, cDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%3, %1) : (!mef.chain, !mgp.context) -> ()
// CHECK: }, {
// CHECK: ^bb0(%arg0: !mef.chain, %arg1: !mgp.context):
// CHECK:   %0 = "mgp.chain.host_to_device"(%arg0, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.context) -> !mgp.chain
// CHECK:   %1 = "mgp.buffer.alloc.static"(%0, %arg1) {aRuntimeSlot = 1 : ui64, bSize = 6 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<6>
// CHECK:   %2 = "write_to_buffer"(%1) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<6>) -> !mgp.chain
// CHECK:   %3:2 = "mgp.buffer.split.borrow"(%2, %arg1, %1) {aRuntimeSlot = 1 : ui64, bOutBufferRawSizes = #M.dense_array<2, 4> : !M.array<2xui64>, cDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<6>) -> (!mgp.buffer<2>, !mgp.buffer<4>)
// CHECK:   %4 = "mgp.buffer.alloc.static"(%2, %arg1) {aRuntimeSlot = 1 : ui64, bSize = 6 : ui64, cRawAlign = 18446744073709551615 : ui64, dDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mgp.buffer<6>
// CHECK:   %5 = "concat_buffers"(%2, %4, %3#0, %3#1) {tempInOutSig = #M.inout_sig<"ioii">} : (!mgp.chain, !mgp.buffer<6>, !mgp.buffer<2>, !mgp.buffer<4>) -> !mgp.chain
// CHECK:   %6 = "mgp.cpu.tensor_spec.create"() {aRawDims = #M.dense_array<6> : !M.array<1xui64>, bRawDType = 135 : ui8} : () -> !mgp.tensor_spec<6xsi8>
// CHECK:   %7 = "mgp.tensor.create"(%5, %arg1, %4, %6) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context, !mgp.buffer<6>, !mgp.tensor_spec<6xsi8>) -> !mgp.tensor<6xsi8>
// CHECK:   %8 = "mgp.chain.device_to_host"(%5, %arg1) {aRuntimeSlot = 1 : ui64, bDevice = #M.device_ref<"cpu", 0>} : (!mgp.chain, !mgp.context) -> !mef.chain
// CHECK:   "mef.output"(%8, %7) : (!mef.chain, !mgp.tensor<6xsi8>) -> ()
// CHECK: }) {frameworkName = "mof", inputOutputTypes = () -> !mgp.tensor<6xsi8>, sym_name = "safe_split_borrow"} : () -> ()

// CHECK: maybeSameAs begin
// Note %3#1 is considered maybe the same as %1.
// CHECK: %3#0 |-> {%1}
// CHECK: %3#1 |-> {%1}
// CHECK: maybeSameAs end

// CHECK: exclusive begin
// Note %1 is only exclusive with %7 because it is implicitly lifetime
// extended by %3#{0,1}.
// CHECK: %1 |-> {%0[0], %2[0], %7, %8}
// CHECK: exclusive end

mgp.model @safe_split_borrow : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1, 0]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context) -> (!mef.chain, !mgp.tensor<6xsi8>) {
  %ch1 = mgp.chain.host_to_device %ch0, %ctx[1]
  %in_buf = mgp.buffer.alloc.static %ch1, %ctx[1] : !mgp.buffer<6>
  %ch2 = "write_to_buffer"(%in_buf) {tempInOutSig = #M.inout_sig<"o">} : (!mgp.buffer<6>) -> !mgp.chain
  %in_buf_borrows:2 = mgp.buffer.split.borrow %ch2, %ctx[1], %in_buf : !mgp.buffer<6> -> (!mgp.buffer<2>, !mgp.buffer<4>)
  %out_buf = mgp.buffer.alloc.static %ch2, %ctx[1] : !mgp.buffer<6>
  %ch3 = "concat_buffers"(%ch2, %out_buf, %in_buf_borrows#0, %in_buf_borrows#1) {tempInOutSig = #M.inout_sig<"ioii">} : (!mgp.chain, !mgp.buffer<6>, !mgp.buffer<2>, !mgp.buffer<4>) -> !mgp.chain
  %out_spec = mgp.cpu.tensor_spec.create : !mgp.tensor_spec<6xsi8>
  %ten = mgp.tensor.create %ch3, %ctx[1], %out_buf, %out_spec : !mgp.tensor<6xsi8>
  %ch4 = mgp.chain.device_to_host %ch3, %ctx[1]
  mef.output %ch4, %ten : !mef.chain, !mgp.tensor<6xsi8>
}
